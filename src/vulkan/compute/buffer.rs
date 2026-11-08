use crate::vulkan::adapter::Adapter;
use ash::{prelude::VkResult, vk};
use bytemuck::{AnyBitPattern, NoUninit};
use std::{ffi::c_void, mem, ptr::NonNull, slice};

/// Device memory a shader reaches through a 64-bit address, for storage, copies and indirect
/// arguments.
pub struct AddressedBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    address: vk::DeviceAddress,
}

impl AddressedBuffer {
    /// `size` bytes of device-local memory, usable as storage, as either end of a copy, and as
    /// indirect arguments, plus whatever `usage` adds.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let usage = usage
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDIRECT_BUFFER;
        let (handle, memory, _) = create(
            adapter,
            device,
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok(Self {
            handle,
            memory,
            size,
            address,
        })
    }

    #[inline]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Its memory, dedicated to this buffer alone and bound at offset 0: the buffer's bytes are
    /// the memory's bytes from its start.
    #[inline]
    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Where shaders find it.
    #[inline]
    pub fn address(&self) -> vk::DeviceAddress {
        self.address
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Which way a [`HostBuffer`]'s data goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAccess {
    /// The CPU writes and the device reads: coherent memory, ideally device-local.
    Upload,
    /// The device writes and the CPU reads: cached memory, so reads aren't slow.
    Readback,
}

/// Memory the CPU maps for as long as the buffer lives, to upload to the device or read back
/// from it. Shaders can reach it by address too.
pub struct HostBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    address: vk::DeviceAddress,
    mapped: NonNull<c_void>,
    /// Whether writes need flushing and reads invalidating.
    coherent: bool,
    non_coherent_atom_size: vk::DeviceSize,
}

// The mapped pointer is only written or read through `&mut self` or `&self` copies of plain data.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// `size` bytes, mapped, usable as storage and as either end of a copy, plus whatever `usage`
    /// adds.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        size: vk::DeviceSize,
        access: HostAccess,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let usage = usage
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let (required, preferred) = match access {
            HostAccess::Upload => (
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ),
            HostAccess::Readback => (
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_CACHED,
            ),
        };
        let (handle, memory, flags) = create(adapter, device, size, usage, required, preferred)?;
        // Frees what was made, if mapping fails.
        let destroy = |device: &ash::Device| unsafe {
            device.destroy_buffer(handle, None);
            device.free_memory(memory, None);
        };
        let mapped = match unsafe {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(pointer) => NonNull::new(pointer).ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED),
            Err(error) => Err(error),
        };
        let mapped = match mapped {
            Ok(mapped) => mapped,
            Err(error) => {
                destroy(device);
                return Err(error);
            }
        };
        let coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok(Self {
            handle,
            memory,
            size,
            address,
            mapped,
            coherent,
            non_coherent_atom_size: adapter.properties().limits.non_coherent_atom_size.max(1),
        })
    }

    #[inline]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Where shaders find it.
    #[inline]
    pub fn address(&self) -> vk::DeviceAddress {
        self.address
    }

    /// Copies `data` in at byte `offset`, and makes it visible to the device.
    ///
    /// # Panics
    /// When it doesn't fit.
    pub fn write<T: NoUninit>(
        &mut self,
        device: &ash::Device,
        offset: vk::DeviceSize,
        data: &[T],
    ) -> VkResult<()> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let size = bytes.len() as vk::DeviceSize;
        assert!(self.fits(offset, size), "the write doesn't fit the buffer");
        unsafe {
            let destination = self.mapped.as_ptr().cast::<u8>().add(offset as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
        if !self.coherent {
            let range = self.atom_range(offset, size);
            unsafe { device.flush_mapped_memory_ranges(slice::from_ref(&range))? };
        }
        Ok(())
    }

    /// Copies out what's at byte `offset`, once the device's writes there are done and visible
    /// to the host.
    ///
    /// # Panics
    /// When `data` reaches past the end.
    pub fn read<T: AnyBitPattern>(
        &self,
        device: &ash::Device,
        offset: vk::DeviceSize,
        data: &mut [T],
    ) -> VkResult<()> {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        assert!(self.fits(offset, size), "the read doesn't fit the buffer");
        if !self.coherent {
            let range = self.atom_range(offset, size);
            unsafe { device.invalidate_mapped_memory_ranges(slice::from_ref(&range))? };
        }
        // Every bit pattern is a valid `T`, so any bytes make valid values.
        unsafe {
            let source = self.mapped.as_ptr().cast::<u8>().add(offset as usize);
            std::ptr::copy_nonoverlapping(source, data.as_mut_ptr().cast::<u8>(), size as usize);
        }
        Ok(())
    }

    /// Whether `size` bytes at byte `offset` lie inside the buffer; an offset so large that the
    /// sum overflows lies outside.
    #[inline]
    fn fits(&self, offset: vk::DeviceSize, size: vk::DeviceSize) -> bool {
        offset.checked_add(size).is_some_and(|end| end <= self.size)
    }

    /// The range to flush or invalidate for `bytes` at `offset`, widened to whole atoms.
    fn atom_range(
        &self,
        offset: vk::DeviceSize,
        bytes: vk::DeviceSize,
    ) -> vk::MappedMemoryRange<'static> {
        let atom = self.non_coherent_atom_size;
        let start = offset / atom * atom;
        let end = (offset + bytes).div_ceil(atom) * atom;
        let size = if end >= self.size {
            vk::WHOLE_SIZE
        } else {
            end - start
        };
        vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(start)
            .size(size)
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A buffer bound to new memory with the `required` properties, and the `preferred` ones too when
/// some memory type has both, and the properties its memory got. The memory can be reached by
/// device address.
fn create(
    adapter: &Adapter,
    device: &ash::Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> VkResult<(vk::Buffer, vk::DeviceMemory, vk::MemoryPropertyFlags)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let handle = unsafe { device.create_buffer(&buffer_info, None)? };
    let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
    let allocate = |memory_type_index: u32| {
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut flags_info);
        unsafe { device.allocate_memory(&allocate_info, None) }
            .map(|memory| (memory, memory_type_index))
    };
    // A type with the preferred properties too, and when its heap is full (such as the small
    // heap of device-local memory the host sees without resizable BAR), one with the required
    // properties alone.
    let preferred_type = find_memory_type(adapter, requirements, required | preferred);
    let required_type = find_memory_type(adapter, requirements, required);
    let allocated = match (preferred_type, required_type) {
        (Some(preferred_type), Some(required_type)) if preferred_type != required_type => {
            allocate(preferred_type).or_else(|error| match error {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    allocate(required_type)
                }
                _ => Err(error),
            })
        }
        (Some(memory_type_index), _) | (None, Some(memory_type_index)) => {
            allocate(memory_type_index)
        }
        (None, None) => Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
    };
    let (memory, memory_type_index) = match allocated {
        Ok(allocated) => allocated,
        Err(error) => {
            unsafe { device.destroy_buffer(handle, None) };
            return Err(error);
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(handle, memory, 0) } {
        unsafe {
            device.destroy_buffer(handle, None);
            device.free_memory(memory, None);
        }
        return Err(error);
    }
    let flags = adapter.memory_properties().memory_types[memory_type_index as usize].property_flags;
    Ok((handle, memory, flags))
}

/// The first memory type with `flags` that the buffer can use and whose heap can hold it.
fn find_memory_type(
    adapter: &Adapter,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let properties = adapter.memory_properties();
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    types
        .iter()
        .enumerate()
        .position(|(index, memory_type)| {
            let heap = &properties.memory_heaps[memory_type.heap_index as usize];
            (1 << index) & requirements.memory_type_bits != 0
                && memory_type.property_flags.contains(flags)
                && requirements.size <= heap.size
        })
        .map(|index| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host buffer of `size` bytes whose memory flushes in `atom`-byte atoms: no device memory
    /// behind it, for the range arithmetic alone.
    fn unbacked(size: vk::DeviceSize, atom: vk::DeviceSize) -> HostBuffer {
        HostBuffer {
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
            address: 0,
            mapped: NonNull::dangling(),
            coherent: false,
            non_coherent_atom_size: atom,
        }
    }

    /// The range to flush or invalidate covers the bytes asked, widened out to whole atoms, and
    /// becomes the whole size where it would reach the buffer's end.
    #[test]
    fn a_range_widens_to_whole_atoms_and_to_the_end() {
        let buffer = unbacked(1024, 64);
        let range = |offset, bytes| {
            let range = buffer.atom_range(offset, bytes);
            (range.offset, range.size)
        };
        assert_eq!(range(0, 64), (0, 64));
        assert_eq!(range(64, 64), (64, 64));
        assert_eq!(range(65, 1), (64, 64));
        assert_eq!(range(60, 8), (0, 128));
        assert_eq!(range(100, 200), (64, 256));
        assert_eq!(range(960, 64), (960, vk::WHOLE_SIZE));
        assert_eq!(range(1000, 24), (960, vk::WHOLE_SIZE));
        // A size that isn't a whole number of atoms: its last atom runs past the end.
        let buffer = unbacked(1000, 64);
        assert_eq!(buffer.atom_range(990, 10).size, vk::WHOLE_SIZE);
        assert_eq!(buffer.atom_range(900, 10).size, 64);
    }
}
