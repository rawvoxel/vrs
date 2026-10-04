use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, vk};
use std::{ffi::c_void, ptr::NonNull, slice};

pub struct Buffer {
    pub size: vk::DeviceSize,
    pub memory_req: vk::MemoryRequirements,
    pub memory: vk::DeviceMemory,
    pub handle: vk::Buffer,
    mapped: Option<NonNull<c_void>>,
}

impl Buffer {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        buffer_info: vk::BufferCreateInfo,
        memory_property_flags: vk::MemoryPropertyFlags,
        buffer_offset: u64,
    ) -> VkResult<Self> {
        unsafe {
            let buffer = logical_device.create_buffer(&buffer_info, None)?;
            let memory_req = logical_device.get_buffer_memory_requirements(buffer);
            let memory =
                alloc::vk_alloc(adapter, logical_device, memory_req, memory_property_flags)?;
            logical_device.bind_buffer_memory(buffer, memory, buffer_offset)?;

            Ok(Self {
                size: buffer_info.size,
                handle: buffer,
                memory,
                memory_req,
                mapped: None,
            })
        }
    }

    /// Persistently maps the whole buffer, which must be in `HOST_VISIBLE` memory.
    pub fn mem_map(&mut self, logical_device: &ash::Device) -> VkResult<()> {
        if self.mapped.is_some() {
            return Ok(());
        }

        let ptr = unsafe {
            logical_device.map_memory(
                self.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        };
        self.mapped = NonNull::new(ptr);
        Ok(())
    }

    /// Copies `data` to the start of the mapped buffer.
    ///
    /// The device must not be accessing the buffer while it's written, and since
    /// nothing flushes the write the memory should be `HOST_COHERENT`.
    ///
    /// # Panics
    /// If the buffer isn't mapped or `data` doesn't fit.
    pub fn mem_write<T: Copy>(&self, data: &[T]) {
        let ptr = self.mapped.expect("Buffer must be mapped before it's written.");
        let byte_count = size_of_val(data);
        assert!(
            byte_count as vk::DeviceSize <= self.size,
            "Writing {byte_count} bytes overflows a {} byte buffer.",
            self.size,
        );

        // SAFETY: The mapping covers the whole buffer and was bounds checked above.
        // Mappings are aligned to at least `minMemoryMapAlignment` (64 bytes).
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                ptr.as_ptr().cast::<u8>(),
                byte_count,
            );
        }
    }

    pub fn mem_unmap(&mut self, logical_device: &ash::Device) {
        if self.mapped.take().is_some() {
            unsafe { logical_device.unmap_memory(self.memory) }
        }
    }

    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    /// Freeing the memory also unmaps it.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_buffer(self.handle, None);
        }
    }
}

/// A `DEVICE_LOCAL` buffer filled from its own persistently mapped staging buffer,
/// which grows whenever written data doesn't fit.
pub struct StagedBuffer {
    pub src: Buffer,
    pub dst: Buffer,
    pub usage: vk::BufferUsageFlags,
    /// Bytes written by the last `write`, which `cmd_copy` copies.
    pub len: vk::DeviceSize,
}

impl StagedBuffer {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        usage: vk::BufferUsageFlags,
        capacity: vk::DeviceSize,
    ) -> VkResult<Self> {
        let src_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let src_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut src = Buffer::new(adapter, logical_device, src_info, src_flags, 0)?;
        src.mem_map(logical_device)?;

        let dst_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let dst = Buffer::new(
            adapter,
            logical_device,
            dst_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            0,
        )?;

        Ok(Self {
            src,
            dst,
            usage,
            len: 0,
        })
    }

    #[inline]
    pub fn capacity(&self) -> vk::DeviceSize {
        self.dst.size
    }

    /// Writes `data` into the staging buffer, growing both buffers to the next
    /// power of two first when it doesn't fit.
    ///
    /// # Safety
    /// The device must not be using either buffer, and a grown `dst` has a new handle.
    pub unsafe fn write<T: Copy>(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        data: &[T],
    ) -> VkResult<()> {
        let byte_count = size_of_val(data) as vk::DeviceSize;
        if byte_count > self.capacity() {
            let grown = Self::new(
                adapter,
                logical_device,
                self.usage,
                byte_count.next_power_of_two(),
            )?;
            unsafe { std::mem::replace(self, grown).destroy(logical_device) };
        }

        self.src.mem_write(data);
        self.len = byte_count;
        Ok(())
    }

    /// Records a copy of the last write from `src` to `dst`. Reads of `dst` still need a barrier.
    pub fn cmd_copy(&self, logical_device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        if self.len == 0 {
            return;
        }

        let region = vk::BufferCopy2::default().size(self.len);
        let copy_info = vk::CopyBufferInfo2::default()
            .src_buffer(self.src.handle)
            .dst_buffer(self.dst.handle)
            .regions(slice::from_ref(&region));
        unsafe { logical_device.cmd_copy_buffer2(cmd_buffer, &copy_info) }
    }

    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            self.src.destroy(logical_device);
            self.dst.destroy(logical_device);
        }
    }
}

#[inline]
pub fn create_buffer_view(
    logical_device: &ash::Device,
    buffer: vk::Buffer,
    format: vk::Format,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
) -> VkResult<vk::BufferView> {
    let info = vk::BufferViewCreateInfo::default()
        .buffer(buffer)
        .format(format)
        .offset(offset)
        .range(range);

    unsafe { logical_device.create_buffer_view(&info, None) }
}
