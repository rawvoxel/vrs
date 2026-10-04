use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, vk};
use std::{ffi::c_void, ptr::NonNull};

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

    /// Copies `data` to the start of the mapped buffer, see `mem_write_at`.
    #[inline]
    pub fn mem_write<T: Copy>(&self, data: &[T]) {
        self.mem_write_at(0, data);
    }

    /// Copies `data` into the mapped buffer, starting `offset` bytes in.
    ///
    /// The device must not be accessing the written bytes, and since
    /// nothing flushes the write the memory should be `HOST_COHERENT`.
    ///
    /// # Panics
    /// If the buffer isn't mapped or `data` doesn't fit.
    pub fn mem_write_at<T: Copy>(&self, offset: vk::DeviceSize, data: &[T]) {
        let ptr = self.mapped.expect("Buffer must be mapped before it's written.");
        let byte_count = size_of_val(data);
        assert!(
            offset.checked_add(byte_count as vk::DeviceSize).is_some_and(|end| end <= self.size),
            "Writing {byte_count} bytes at {offset} overflows a {} byte buffer.",
            self.size,
        );

        // SAFETY: The mapping covers the whole buffer and was bounds checked above.
        // Bytes are copied, so the destination needs no alignment.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                ptr.as_ptr().cast::<u8>().add(offset as usize),
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

/// A buffer which grows whenever it's too small. By default a persistently mapped
/// `HOST_VISIBLE` buffer the device reads directly.
pub struct DynamicBuffer {
    pub buffer: Buffer,
    pub usage: vk::BufferUsageFlags,
    pub memory_flags: vk::MemoryPropertyFlags,
}

impl DynamicBuffer {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        usage: vk::BufferUsageFlags,
        capacity: vk::DeviceSize,
    ) -> VkResult<Self> {
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Self::with_memory(adapter, logical_device, usage, flags, capacity)
    }

    /// Only `HOST_VISIBLE` memory is mapped, and can be written.
    pub fn with_memory(
        adapter: &Adapter,
        logical_device: &ash::Device,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
        capacity: vk::DeviceSize,
    ) -> VkResult<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut buffer = Buffer::new(adapter, logical_device, info, memory_flags, 0)?;
        if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            buffer.mem_map(logical_device)?;
        }

        Ok(Self {
            buffer,
            usage,
            memory_flags,
        })
    }

    #[inline]
    pub fn capacity(&self) -> vk::DeviceSize {
        self.buffer.size
    }

    /// Writes `data` to the start of the buffer, first growing it to the next power of two when it doesn't fit.
    ///
    /// # Safety
    /// The device must not be using the buffer, and a grown buffer has a new handle.
    pub unsafe fn write<T: Copy>(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        data: &[T],
    ) -> VkResult<()> {
        unsafe { self.reserve(adapter, logical_device, size_of_val(data) as vk::DeviceSize)? };
        self.buffer.mem_write(data);
        Ok(())
    }

    /// Grows the buffer to the next power of two when it's smaller than `byte_count`, discarding its contents.
    ///
    /// # Safety
    /// The device must not be using the buffer, and a grown buffer has a new handle.
    pub unsafe fn reserve(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        byte_count: vk::DeviceSize,
    ) -> VkResult<()> {
        if byte_count > self.capacity() {
            let grown = Self::with_memory(
                adapter,
                logical_device,
                self.usage,
                self.memory_flags,
                byte_count.next_power_of_two(),
            )?;
            unsafe { std::mem::replace(self, grown).destroy(logical_device) };
        }
        Ok(())
    }

    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe { self.buffer.destroy(logical_device) }
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
