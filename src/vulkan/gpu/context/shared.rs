//! A [`GpuContext`] whose device another Vulkan user (such as wgpu) shares: that user describes
//! the device it needs, the context merges in its own needs, makes the device and owns it.

use super::{GpuContext, GpuOptions, create_immediate, query_support};
use crate::vulkan::{adapter::Adapter, queue::Queue};
use ash::{ext, khr, prelude::VkResult, vk};
use std::{
    ffi::{CStr, c_char, c_void},
    sync::Mutex,
};

impl GpuContext {
    /// Makes the device another Vulkan user asks for, with everything a [`GpuContext`] needs
    /// merged in, and owns it: the other user borrows the device and must be done with it before
    /// the context is dropped.
    ///
    /// `extensions` and `info` are the other user's: its device extensions, and its create info
    /// with its one queue (whose family must have graphics, compute and timestamps), its enabled
    /// features and its chain of feature structs. The context's own needs are merged in:
    /// - a missing extension is appended to `extensions`, which on return lists every extension
    ///   the device has;
    /// - a feature is turned on in the struct of the chain that holds it, the Vulkan 1.2 or 1.3
    ///   struct when the chain has one, or else in a struct of the context's own linked in front
    ///   of the chain. Nothing the other user enabled is turned off.
    ///
    /// Adapters that lack something are refused with [`vk::Result::ERROR_FEATURE_NOT_PRESENT`],
    /// as [`GpuContext::new`] refuses them.
    ///
    /// # Safety
    ///
    /// Every pointer in `info` must be valid, and its chain must hold only structs that extend
    /// `VkDeviceCreateInfo`. Feature structs in the chain have bits set in place, never cleared.
    pub unsafe fn new_shared(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        extensions: &mut Vec<&'static CStr>,
        info: &vk::DeviceCreateInfo<'_>,
    ) -> VkResult<Self> {
        let adapter = Adapter::new(instance, physical_device);
        let support = query_support(instance, &adapter);
        if !support.missing.is_empty() || info.queue_create_info_count != 1 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let queue_info = unsafe { &*info.p_queue_create_infos };
        let queue_family = adapter
            .queue_families()
            .iter()
            .find(|family| family.index() == queue_info.queue_family_index)
            .copied()
            .filter(|family| {
                let properties = family.properties();
                properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && properties.timestamp_valid_bits > 0
                    && queue_info.queue_count >= 1
            })
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;

        if !extensions.contains(&ext::mesh_shader::NAME) {
            extensions.push(ext::mesh_shader::NAME);
        }
        let extension_names: Vec<*const c_char> =
            extensions.iter().map(|name| name.as_ptr()).collect();

        let chain = unsafe { Chain::of(info) };
        let mut merged = *info;
        merged.pp_enabled_extension_names = extension_names.as_ptr();
        merged.enabled_extension_count = extension_names.len() as u32;

        // Vulkan 1.0's features: in a `VkPhysicalDeviceFeatures2` if the chain has one, else in
        // the enabled features, copied so the other user's are left as they were.
        let mut core = if info.p_enabled_features.is_null() {
            vk::PhysicalDeviceFeatures::default()
        } else {
            unsafe { *info.p_enabled_features }
        };
        let enable_core = |features: &mut vk::PhysicalDeviceFeatures| {
            features.shader_int64 = vk::TRUE;
            features.fragment_stores_and_atomics = vk::TRUE;
        };
        match unsafe { chain.find::<vk::PhysicalDeviceFeatures2>() } {
            Some(features) => enable_core(unsafe { &mut (*features).features }),
            None => {
                enable_core(&mut core);
                merged.p_enabled_features = &core;
            }
        }

        let vulkan_12 = unsafe { chain.find::<vk::PhysicalDeviceVulkan12Features>() };
        let vulkan_13 = unsafe { chain.find::<vk::PhysicalDeviceVulkan13Features>() };
        let mut links = Links::new(info.p_next);

        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut atomic_int64 = vk::PhysicalDeviceShaderAtomicInt64Features::default();
        let mut subgroup_extended_types =
            vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default();
        let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut subgroup_size_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
        let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        unsafe {
            match vulkan_12 {
                Some(features) => {
                    let features = &mut *features;
                    features.buffer_device_address = vk::TRUE;
                    features.timeline_semaphore = vk::TRUE;
                    features.shader_buffer_int64_atomics = vk::TRUE;
                    if support.subgroup_extended_types {
                        features.shader_subgroup_extended_types = vk::TRUE;
                    }
                }
                None => {
                    links.enable(&chain, &mut buffer_device_address, |features| {
                        features.buffer_device_address = vk::TRUE;
                    });
                    links.enable(&chain, &mut timeline_semaphore, |features| {
                        features.timeline_semaphore = vk::TRUE;
                    });
                    links.enable(&chain, &mut atomic_int64, |features| {
                        features.shader_buffer_int64_atomics = vk::TRUE;
                    });
                    if support.subgroup_extended_types {
                        links.enable(&chain, &mut subgroup_extended_types, |features| {
                            features.shader_subgroup_extended_types = vk::TRUE;
                        });
                    }
                }
            }
            match vulkan_13 {
                Some(features) => {
                    let features = &mut *features;
                    features.synchronization2 = vk::TRUE;
                    features.dynamic_rendering = vk::TRUE;
                    features.subgroup_size_control = vk::TRUE;
                    features.compute_full_subgroups = vk::TRUE;
                    features.maintenance4 = vk::TRUE;
                }
                None => {
                    links.enable(&chain, &mut synchronization2, |features| {
                        features.synchronization2 = vk::TRUE;
                    });
                    links.enable(&chain, &mut dynamic_rendering, |features| {
                        features.dynamic_rendering = vk::TRUE;
                    });
                    links.enable(&chain, &mut subgroup_size_control, |features| {
                        features.subgroup_size_control = vk::TRUE;
                        features.compute_full_subgroups = vk::TRUE;
                    });
                    links.enable(&chain, &mut maintenance4, |features| {
                        features.maintenance4 = vk::TRUE;
                    });
                }
            }
            links.enable(&chain, &mut mesh_shader, |features| {
                features.task_shader = vk::TRUE;
                features.mesh_shader = vk::TRUE;
                if support.mesh_shader_queries {
                    features.mesh_shader_queries = vk::TRUE;
                }
            });
        }
        merged.p_next = links.head;

        let device = unsafe { instance.create_device(physical_device, &merged, None)? };
        let immediate = match create_immediate(&device, &queue_family) {
            Ok(immediate) => immediate,
            Err(error) => {
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };
        Ok(Self {
            mesh_shader: ext::mesh_shader::Device::new(instance, &device),
            queue: Mutex::new(Queue::new(&queue_family, &device, 0)),
            adapter,
            device,
            queue_family,
            immediate: Mutex::new(immediate),
            subgroup: support.subgroup,
            mesh_limits: support.mesh_limits,
            options: GpuOptions {
                present: extensions.contains(&khr::swapchain::NAME),
            },
        })
    }
}

/// The structs a create info's `p_next` chain holds, as they were when it was read.
struct Chain(Vec<*mut vk::BaseOutStructure<'static>>);

impl Chain {
    /// # Safety
    /// Every struct in `info`'s chain must be valid.
    unsafe fn of(info: &vk::DeviceCreateInfo<'_>) -> Self {
        let mut structs = Vec::new();
        let mut next = info.p_next as *mut vk::BaseOutStructure<'static>;
        while !next.is_null() {
            structs.push(next);
            next = unsafe { (*next).p_next };
        }
        Self(structs)
    }

    /// The chain's `T`, if it has one.
    ///
    /// # Safety
    /// Every struct in the chain must be valid.
    unsafe fn find<T: vk::TaggedStructure>(&self) -> Option<*mut T> {
        self.0
            .iter()
            .find(|base| unsafe { (***base).s_type } == T::STRUCTURE_TYPE)
            .map(|base| *base as *mut T)
    }
}

/// The front of a chain that structs of our own are linked onto.
struct Links {
    head: *const c_void,
}

impl Links {
    fn new(head: *const c_void) -> Self {
        Self { head }
    }

    /// Turns a feature on with `enable`: in the chain's `T` if it has one, else in `own`, which
    /// is linked in front of the chain.
    ///
    /// # Safety
    /// Every struct in `chain` must be valid, and `own` must outlive every use of the chain.
    unsafe fn enable<T: vk::TaggedStructure>(
        &mut self,
        chain: &Chain,
        own: &mut T,
        enable: impl FnOnce(&mut T),
    ) {
        match unsafe { chain.find::<T>() } {
            Some(existing) => enable(unsafe { &mut *existing }),
            None => {
                enable(own);
                let base = own as *mut T as *mut vk::BaseOutStructure<'static>;
                unsafe { (*base).p_next = self.head as *mut vk::BaseOutStructure<'static> };
                self.head = base as *const c_void;
            }
        }
    }
}
