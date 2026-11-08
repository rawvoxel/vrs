//! vrs on a real GPU: making devices, buffers read and written, and what the recording functions
//! record, and refuse in debug builds. Ignored where there's no GPU, such as CI: run them with
//! `cargo test -p vrs -- --ignored`. With `VRS_VALIDATION=1` each test also fails on any
//! validation error besides the known ones.

use std::{
    ffi::{CStr, c_void},
    panic::{self, AssertUnwindSafe},
    ptr,
};
use vrs::ash::{self, vk};
use vrs::{
    App,
    vulkan::{
        commands,
        compute::{AddressedBuffer, HostAccess, HostBuffer},
        gpu::{self, RenderImage, VulkanDevice},
        image, sync, validation,
    },
};

/// Prints what vrs logs, such as validation messages, to the test's output.
struct PrintLogger;

impl log::Log for PrintLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        eprintln!("{}: {}", record.level(), record.args());
    }

    fn flush(&self) {}
}

fn headless_app() -> App {
    // Only the first test to get here installs it.
    if log::set_logger(&PrintLogger).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    App::headless(vk::ApplicationInfo::default()).expect("a Vulkan instance")
}

/// A device on the most preferred adapter that can make one.
fn preferred_device(app: &mut App) -> VulkanDevice {
    let adapter = app
        .adapters_by_preference()
        .into_iter()
        .find(|adapter| gpu::missing_features(app, adapter).is_empty())
        .cloned()
        .expect("an adapter that can make a device");
    unsafe { gpu::create_device(app, &adapter) }.expect("the device")
}

/// A command pool, its one command buffer and a fence, for one-shots.
struct OneShot {
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl OneShot {
    fn new(device: &VulkanDevice) -> Self {
        let pool = commands::create_command_pool(&device.device, &device.queue_family)
            .expect("a command pool");
        let command_buffer =
            commands::create_command_buffer(&device.device, pool, vk::CommandBufferLevel::PRIMARY)
                .expect("a command buffer");
        let fence = sync::create_fence(&device.device).expect("a fence");
        Self {
            pool,
            command_buffer,
            fence,
        }
    }

    fn destroy(&self, device: &VulkanDevice) {
        unsafe {
            device.device.destroy_fence(self.fence, None);
            device.device.destroy_command_pool(self.pool, None);
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn devices_are_made_on_every_adapter_that_can_and_destroyed_by_their_app() {
    let mut app = headless_app();
    println!("validation on: {}", app.validation_enabled());
    let adapters: Vec<_> = app.adapters().to_vec();
    let mut devices = Vec::new();
    for adapter in &adapters {
        let missing = gpu::missing_features(&app, adapter);
        if !missing.is_empty() {
            println!("{} can't make a device: {missing:?}", adapter.name());
            continue;
        }
        let device = unsafe { gpu::create_device(&mut app, adapter) }.expect("the device");
        assert!(!device.adapter.name().is_empty());
        assert_eq!(device.adapter.handle(), adapter.handle());
        let family = device.queue_family.properties();
        assert!(
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        );
        assert!(family.timestamp_valid_bits > 0);
        let subgroup = device.adapter.subgroup();
        assert!(subgroup.min_size.is_power_of_two() && subgroup.max_size.is_power_of_two());
        assert!(subgroup.min_size <= subgroup.size && subgroup.size <= subgroup.max_size);
        if let Some(heaps) = gpu::memory_budget(&device) {
            assert_eq!(
                heaps.len(),
                device.adapter.memory_properties().memory_heap_count as usize
            );
        }
        println!(
            "{}: subgroups {subgroup:?}, shader clock {}, pipeline executables {}",
            device.adapter.name(),
            device.shader_clock,
            device.pipeline_executables.is_some()
        );
        devices.push(device);
    }
    assert!(!devices.is_empty(), "no adapter can make a device");
    // The first destroyed early; the `App` destroys the rest.
    unsafe { app.destroy_device(&devices[0]) }.expect("the device finished its work");
    drop(app);
    validation::assert_no_new_errors();
}

#[test]
#[ignore = "needs a GPU"]
fn a_shared_device_has_its_features_merged_into_the_callers_chain() {
    let app = headless_app();
    let adapter = app
        .adapters_by_preference()
        .into_iter()
        .find(|adapter| gpu::missing_features(&app, adapter).is_empty())
        .cloned()
        .expect("an adapter that can make a device");
    let family = adapter
        .queue_families()
        .iter()
        .find(|family| {
            family
                .properties()
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .copied()
        .expect("a graphics and compute queue family");

    // What another user asks for: its own Vulkan 1.2 struct, with nothing on.
    let queue_priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family.index())
        .queue_priorities(&queue_priorities);
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let callers_struct = ptr::addr_of!(vulkan_12_features).cast::<c_void>();
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .push_next(&mut vulkan_12_features);
    let mut extensions: Vec<&'static CStr> = Vec::new();
    let device = unsafe {
        gpu::create_shared_device(app.instance(), adapter.handle(), &mut extensions, &info)
    }
    .expect("the shared device");
    let chain_head = info.p_next;

    assert!(extensions.contains(&ash::ext::mesh_shader::NAME));
    // Turned on in the caller's own struct, which still heads its chain.
    assert!(ptr::eq(chain_head, callers_struct));
    assert_eq!(vulkan_12_features.buffer_device_address, vk::TRUE);
    assert_eq!(vulkan_12_features.shader_float16, vk::TRUE);
    // The caller owns the device.
    unsafe {
        device
            .device
            .device_wait_idle()
            .expect("the device is idle");
        device.device.destroy_device(None);
    }
    drop(app);
    validation::assert_no_new_errors();
}

#[test]
#[ignore = "needs a GPU"]
fn host_buffers_write_and_read_on_the_host_and_through_the_device() {
    let mut app = headless_app();
    let device = preferred_device(&mut app);
    let one_shot = OneShot::new(&device);
    // More than one update holds.
    let words = 40_000usize;
    let size = (words * 4) as vk::DeviceSize;
    let values: Vec<u32> = (0..words as u32).collect();

    let mut upload = HostBuffer::new(
        &device.adapter,
        &device.device,
        size,
        HostAccess::Upload,
        vk::BufferUsageFlags::empty(),
    )
    .expect("an upload buffer");
    upload
        .write(&device.device, 0, &values)
        .expect("the values written");
    let mut read = vec![0u32; words];
    upload
        .read(&device.device, 0, &mut read)
        .expect("the values read");
    assert_eq!(read, values);
    // A write past the end, or so far its end overflows, is refused.
    for offset in [size - 4, u64::MAX - 2] {
        let refused = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = upload.write(&device.device, offset, &[1u32, 2]);
        }));
        assert!(refused.is_err(), "a write at {offset} doesn't fit");
    }

    let buffer = AddressedBuffer::new(
        &device.adapter,
        &device.device,
        size,
        vk::BufferUsageFlags::empty(),
    )
    .expect("a device buffer");
    let reversed: Vec<u32> = values.iter().rev().copied().collect();
    unsafe {
        commands::submit_and_wait(
            &device,
            one_shot.command_buffer,
            one_shot.fence,
            |command_buffer| {
                commands::cmd_copy_buffer(
                    &device.device,
                    command_buffer,
                    upload.handle(),
                    0,
                    buffer.handle(),
                    0,
                    size,
                );
                sync::cmd_pipeline_barrier(&device.device, command_buffer);
                // The first 16 words filled, then words from the 32nd on updated, across 64 KiB.
                commands::cmd_fill_buffer(
                    &device.device,
                    command_buffer,
                    buffer.handle(),
                    0,
                    64,
                    7,
                );
                commands::cmd_update_buffer(
                    &device.device,
                    command_buffer,
                    buffer.handle(),
                    128,
                    bytemuck::cast_slice(&reversed[32..]),
                );
            },
        )
    }
    .expect("the copies ran");

    let mut read_back = vec![0u32; words];
    unsafe {
        commands::read_back(
            &device,
            one_shot.command_buffer,
            one_shot.fence,
            buffer.handle(),
            0,
            &mut read_back,
        )
    }
    .expect("the buffer read back");
    assert!(read_back[..16].iter().all(|&word| word == 7));
    assert_eq!(read_back[16..32], values[16..32]);
    assert_eq!(read_back[32..], reversed[32..]);

    unsafe {
        buffer.destroy(&device.device);
        upload.destroy(&device.device);
    }
    one_shot.destroy(&device);
    drop(app);
    validation::assert_no_new_errors();
}

#[test]
#[ignore = "needs a GPU"]
fn rendering_clears_a_color_attachment_that_copies_back() {
    let mut app = headless_app();
    let device = preferred_device(&mut app);
    let one_shot = OneShot::new(&device);
    let extent = vk::Extent2D {
        width: 16,
        height: 8,
    };
    let color = RenderImage::new(
        &device.adapter,
        &device.device,
        extent,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
    )
    .expect("a color image");
    let depth = RenderImage::depth(
        &device.adapter,
        &device.device,
        extent,
        vk::Format::D32_SFLOAT,
    )
    .expect("a depth image");
    let pixels = (extent.width * extent.height) as usize;
    let copy = AddressedBuffer::new(
        &device.adapter,
        &device.device,
        (pixels * 4) as vk::DeviceSize,
        vk::BufferUsageFlags::empty(),
    )
    .expect("a buffer for the copy");

    unsafe {
        commands::submit_and_wait(
            &device,
            one_shot.command_buffer,
            one_shot.fence,
            |command_buffer| {
                image::transition_layout(
                    &device.device,
                    color.image(),
                    command_buffer,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
                    (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    vk::ImageAspectFlags::COLOR,
                );
                image::transition_layout(
                    &device.device,
                    depth.image(),
                    command_buffer,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
                    (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    vk::ImageAspectFlags::DEPTH,
                );
                commands::cmd_begin_rendering(
                    &device.device,
                    command_buffer,
                    extent,
                    Some((depth.view(), Some(0.0))),
                    Some((
                        color.view(),
                        vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    )),
                );
                commands::cmd_set_viewport(&device.device, command_buffer, extent);
                commands::cmd_set_scissor(&device.device, command_buffer, extent);
                commands::cmd_end_rendering(&device.device, command_buffer);
                image::transition_layout(
                    &device.device,
                    color.image(),
                    command_buffer,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    (
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                    vk::ImageAspectFlags::COLOR,
                );
                commands::cmd_copy_image_to_buffer(
                    &device.device,
                    command_buffer,
                    color.image(),
                    extent,
                    copy.handle(),
                );
            },
        )
    }
    .expect("the frame ran");

    let mut texels = vec![0u32; pixels];
    unsafe {
        commands::read_back(
            &device,
            one_shot.command_buffer,
            one_shot.fence,
            copy.handle(),
            0,
            &mut texels,
        )
    }
    .expect("the copy read back");
    let magenta = u32::from_le_bytes([255, 0, 255, 255]);
    assert!(texels.iter().all(|&texel| texel == magenta));

    unsafe {
        copy.destroy(&device.device);
        depth.destroy(&device.device);
        color.destroy(&device.device);
    }
    one_shot.destroy(&device);
    drop(app);
    validation::assert_no_new_errors();
}

/// Each recording function refuses what its command's rules forbid, before recording anything.
#[cfg(debug_assertions)]
#[test]
#[ignore = "needs a GPU"]
fn recording_functions_refuse_what_their_commands_forbid() {
    let mut app = headless_app();
    let device = preferred_device(&mut app);
    let one_shot = OneShot::new(&device);
    let buffer = AddressedBuffer::new(
        &device.adapter,
        &device.device,
        1024,
        vk::BufferUsageFlags::empty(),
    )
    .expect("a buffer");
    let (handle, command_buffer, raw) = (buffer.handle(), one_shot.command_buffer, &device.device);
    let refusals: [(&str, &dyn Fn()); 10] = [
        ("a fill at an odd offset", &|| unsafe {
            commands::cmd_fill_buffer(raw, command_buffer, handle, 2, 4, 0)
        }),
        ("a fill of part of a word", &|| unsafe {
            commands::cmd_fill_buffer(raw, command_buffer, handle, 0, 6, 0)
        }),
        ("an empty fill", &|| unsafe {
            commands::cmd_fill_buffer(raw, command_buffer, handle, 0, 0, 0)
        }),
        ("an update at an odd offset", &|| unsafe {
            commands::cmd_update_buffer(raw, command_buffer, handle, 2, &[0; 4])
        }),
        ("an update of part of a word", &|| unsafe {
            commands::cmd_update_buffer(raw, command_buffer, handle, 0, &[0; 6])
        }),
        ("an empty copy", &|| unsafe {
            commands::cmd_copy_buffer(raw, command_buffer, handle, 0, handle, 512, 0)
        }),
        ("a copy overlapping itself", &|| unsafe {
            commands::cmd_copy_buffer(raw, command_buffer, handle, 0, handle, 64, 128)
        }),
        ("push constants of part of a word", &|| unsafe {
            commands::cmd_push_constants(raw, command_buffer, vk::PipelineLayout::null(), &[0; 6])
        }),
        ("an indirect dispatch at an odd offset", &|| unsafe {
            commands::cmd_dispatch_indirect(
                raw,
                command_buffer,
                vk::Pipeline::null(),
                vk::PipelineLayout::null(),
                &[],
                handle,
                2,
            )
        }),
        ("a viewport without area", &|| unsafe {
            commands::cmd_set_viewport(
                raw,
                command_buffer,
                vk::Extent2D {
                    width: 0,
                    height: 8,
                },
            )
        }),
    ];
    unsafe { commands::begin_command_buffer(raw, command_buffer) }.expect("recording");
    for (what, record) in refusals {
        let refused = panic::catch_unwind(AssertUnwindSafe(record));
        assert!(refused.is_err(), "{what} was recorded");
    }
    // Nothing was recorded, so the buffer submits and runs.
    unsafe {
        commands::submit(&device, command_buffer, one_shot.fence, &[], &[])
            .expect("the empty submission");
        sync::wait_for_fence(raw, one_shot.fence).expect("it ran");
        assert!(sync::fence_signaled(raw, one_shot.fence).expect("the fence's status"));
        buffer.destroy(raw);
    }
    one_shot.destroy(&device);
    drop(app);
    validation::assert_no_new_errors();
}
