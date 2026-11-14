//! A device made from another Vulkan user's create info, with a [`VulkanDevice`]'s extensions
//! and features merged into it, and the merging both creation paths share: a create info's
//! `p_next` chain read, and added to.

use super::{VulkanDevice, query_support, suits_device};
use crate::vulkan::{
    adapter::Adapter,
    queue::{Queue, QueueFamily},
};
use ash::{ext, khr, nv, prelude::VkResult, vk};
use std::ffi::{CStr, c_char, c_void};

/// Makes the device another Vulkan user (such as wgpu-hal) asks for, with everything a
/// [`VulkanDevice`] needs merged in. The caller owns the device, and destroys it.
///
/// `extensions` and `info` are the other user's: its device extensions, and its create info with
/// its one queue (whose family must have graphics, compute and timestamps), its enabled features
/// and its chain of feature structs. A [`VulkanDevice`]'s own needs are merged in:
/// - a missing extension is appended to `extensions`, which on return lists every extension the
///   device has;
/// - a feature is turned on in the struct of the chain that holds it, the Vulkan 1.2 or 1.3
///   struct when the chain has one, or else in a struct of its own linked in front of the chain.
///   Nothing the other user enabled is turned off.
///
/// Adapters that lack something are refused with [`vk::Result::ERROR_FEATURE_NOT_PRESENT`], as
/// [`create_device`](super::create_device) refuses them.
///
/// # Safety
/// Every pointer in `info` must be valid, and its chain must hold only structs that extend
/// `VkDeviceCreateInfo`. Feature structs in the chain have bits set in place, never cleared. The
/// caller destroys the device (`device.device.destroy_device`) once everything made with it is
/// destroyed, and before `instance`.
pub unsafe fn create_shared_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extensions: &mut Vec<&'static CStr>,
    info: &vk::DeviceCreateInfo<'_>,
) -> VkResult<VulkanDevice> {
    let adapter = Adapter::new(instance, physical_device)?;
    if info.queue_create_info_count != 1 {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let queue_info = unsafe { &*info.p_queue_create_infos };
    let queue_family = adapter
        .queue_families()
        .iter()
        .find(|family| family.index() == queue_info.queue_family_index)
        .copied()
        .filter(|family| suits_device(family) && queue_info.queue_count >= 1)
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    unsafe { create_merged(instance, adapter, queue_family, extensions, info, false) }
}

/// Makes the device `info` describes on `adapter`, its one queue of `queue_family`, with every
/// extension and feature a [`VulkanDevice`] needs merged into `extensions` and `info`'s chain
/// (see [`create_shared_device`]): both creation paths enable the same set. With `diagnostics`, the
/// lost-device diagnostics too, where the adapter has them
/// ([`create_device_with_diagnostics`](super::create_device_with_diagnostics)).
///
/// # Safety
/// As [`create_shared_device`].
pub(super) unsafe fn create_merged(
    instance: &ash::Instance,
    adapter: Adapter,
    queue_family: QueueFamily,
    extensions: &mut Vec<&'static CStr>,
    info: &vk::DeviceCreateInfo<'_>,
    diagnostics: bool,
) -> VkResult<VulkanDevice> {
    let support = query_support(instance, &adapter);
    if !support.missing.is_empty() {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }

    let mut needed = vec![ext::mesh_shader::NAME];
    if support.pipeline_executables {
        needed.push(khr::pipeline_executable_properties::NAME);
    }
    if support.shader_clock {
        needed.push(khr::shader_clock::NAME);
    }
    let checkpoints = diagnostics && adapter.has_extension(nv::device_diagnostic_checkpoints::NAME);
    if checkpoints {
        needed.push(nv::device_diagnostic_checkpoints::NAME);
    }
    let device_fault = diagnostics && adapter.has_extension(ext::device_fault::NAME) && {
        let mut fault_features = vk::PhysicalDeviceFaultFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault_features);
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
        fault_features.device_fault == vk::TRUE
    };
    if device_fault {
        needed.push(ext::device_fault::NAME);
    }
    for name in needed {
        if !extensions.contains(&name) {
            extensions.push(name);
        }
    }
    let extension_names: Vec<*const c_char> = extensions.iter().map(|name| name.as_ptr()).collect();

    let chain = unsafe { Chain::of(info) };
    let mut merged = *info;
    merged.pp_enabled_extension_names = extension_names.as_ptr();
    merged.enabled_extension_count = extension_names.len() as u32;

    // Vulkan 1.0's features: in a `VkPhysicalDeviceFeatures2` if the chain has one, else in the
    // enabled features, copied so the other user's are left as they were.
    let mut core = if info.p_enabled_features.is_null() {
        vk::PhysicalDeviceFeatures::default()
    } else {
        unsafe { *info.p_enabled_features }
    };
    let enable_core = |features: &mut vk::PhysicalDeviceFeatures| {
        features.shader_int16 = vk::TRUE;
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
    let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut subgroup_extended_types =
        vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default();
    let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut subgroup_size_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
    let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default();
    let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let mut executables = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    let mut clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
    let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default();
    unsafe {
        match vulkan_12 {
            Some(features) => {
                let features = &mut *features;
                features.buffer_device_address = vk::TRUE;
                features.timeline_semaphore = vk::TRUE;
                features.shader_buffer_int64_atomics = vk::TRUE;
                features.shader_float16 = vk::TRUE;
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
                links.enable(&chain, &mut float16, |features| {
                    features.shader_float16 = vk::TRUE;
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
                features.shader_demote_to_helper_invocation = vk::TRUE;
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
                links.enable(&chain, &mut demote, |features| {
                    features.shader_demote_to_helper_invocation = vk::TRUE;
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
        if support.pipeline_executables {
            links.enable(&chain, &mut executables, |features| {
                features.pipeline_executable_info = vk::TRUE;
            });
        }
        if support.shader_clock {
            links.enable(&chain, &mut clock, |features| {
                features.shader_subgroup_clock = vk::TRUE;
                features.shader_device_clock = vk::TRUE;
            });
        }
        if device_fault {
            links.enable(&chain, &mut fault, |features| {
                features.device_fault = vk::TRUE;
            });
        }
    }
    merged.p_next = links.head;

    let device = unsafe { instance.create_device(adapter.handle(), &merged, None)? };
    Ok(VulkanDevice {
        mesh_shader: ext::mesh_shader::Device::new(instance, &device),
        pipeline_executables: support
            .pipeline_executables
            .then(|| khr::pipeline_executable_properties::Device::new(instance, &device)),
        queue: Queue::new(&queue_family, &device, 0),
        budget: adapter
            .has_extension(ext::memory_budget::NAME)
            .then(|| instance.clone()),
        shader_clock: support.shader_clock,
        checkpoints: checkpoints
            .then(|| nv::device_diagnostic_checkpoints::Device::new(instance, &device)),
        device_fault: device_fault.then(|| ext::device_fault::Device::new(instance, &device)),
        device,
        adapter,
        queue_family,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `Chain::of` reads every struct in a create info's chain, and `find` names each one there
    /// and nothing else.
    #[test]
    fn a_chain_finds_its_own_structs_and_no_others() {
        let mut vulkan_12 = vk::PhysicalDeviceVulkan12Features::default().shader_int8(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
        let vulkan_12_at = &mut vulkan_12 as *mut vk::PhysicalDeviceVulkan12Features;
        let maintenance4_at = &mut maintenance4 as *mut vk::PhysicalDeviceMaintenance4Features;
        let info = vk::DeviceCreateInfo::default()
            .push_next(&mut vulkan_12)
            .push_next(&mut maintenance4);
        let chain = unsafe { Chain::of(&info) };
        assert_eq!(chain.0.len(), 2);
        let found = unsafe { chain.find::<vk::PhysicalDeviceVulkan12Features>() };
        assert_eq!(found, Some(vulkan_12_at));
        assert_eq!(unsafe { (*found.unwrap()).shader_int8 }, vk::TRUE);
        let found = unsafe { chain.find::<vk::PhysicalDeviceMaintenance4Features>() };
        assert_eq!(found, Some(maintenance4_at));
        assert!(unsafe { chain.find::<vk::PhysicalDeviceVulkan13Features>() }.is_none());
        assert!(
            unsafe { Chain::of(&vk::DeviceCreateInfo::default()) }
                .0
                .is_empty()
        );
    }

    /// When the chain has the struct, `enable` turns the feature on in it, keeps the caller's own
    /// features on, and links nothing of its own.
    #[test]
    fn enabling_a_feature_the_chain_has_a_struct_for_turns_it_on_there() {
        let mut subgroup_size_control =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);
        let info = vk::DeviceCreateInfo::default().push_next(&mut subgroup_size_control);
        let chain = unsafe { Chain::of(&info) };
        let mut links = Links::new(info.p_next);
        let mut own = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        unsafe {
            links.enable(&chain, &mut own, |features| {
                features.compute_full_subgroups = vk::TRUE;
            });
        }
        assert_eq!(links.head, info.p_next, "nothing is linked in front");
        assert_eq!(own.compute_full_subgroups, vk::FALSE);
        let found = unsafe { chain.find::<vk::PhysicalDeviceSubgroupSizeControlFeatures>() }
            .expect("the chain's struct");
        let found = unsafe { &*found };
        assert_eq!(found.compute_full_subgroups, vk::TRUE);
        assert_eq!(
            found.subgroup_size_control,
            vk::TRUE,
            "the caller's feature stays on"
        );
    }

    /// When the chain lacks the struct, `enable` turns the feature on in its own struct and links
    /// it in front of the chain, which stays whole behind it; a second one goes in front of that.
    #[test]
    fn enabling_a_feature_the_chain_lacks_links_its_own_struct_in_front() {
        let mut vulkan_12 = vk::PhysicalDeviceVulkan12Features::default().shader_int8(true);
        let info = vk::DeviceCreateInfo::default().push_next(&mut vulkan_12);
        let chain = unsafe { Chain::of(&info) };
        let mut links = Links::new(info.p_next);
        let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
        unsafe {
            links.enable(&chain, &mut mesh_shader, |features| {
                features.mesh_shader = vk::TRUE;
            });
            links.enable(&chain, &mut maintenance4, |features| {
                features.maintenance4 = vk::TRUE;
            });
        }
        assert_eq!(mesh_shader.mesh_shader, vk::TRUE);
        assert_eq!(maintenance4.maintenance4, vk::TRUE);
        assert_eq!(links.head, &maintenance4 as *const _ as *const c_void);
        assert_eq!(
            maintenance4.p_next as *const c_void,
            &mesh_shader as *const _ as *const c_void
        );
        assert_eq!(
            mesh_shader.p_next as *const c_void, info.p_next,
            "the chain stays behind"
        );
        let first = unsafe { chain.find::<vk::PhysicalDeviceVulkan12Features>() }
            .expect("the chain's struct");
        assert_eq!(unsafe { (*first).shader_int8 }, vk::TRUE);
    }
}
