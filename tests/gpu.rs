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
        compute::{
            AddressedBuffer, ComputePipelines, HostAccess, HostBuffer, PipelineCache, TimestampPool,
        },
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

/// A compute kernel that never ends: one workgroup of 64 threads counting a 64-bit number up from
/// 1 until it equals the push constant `never` (0, so 2^64 steps), then writing it through the
/// address `words` (so the count can't be dropped). Its push constants: the address, then `never`.
/// Compiled from this GLSL with `glslangValidator -V --target-env vulkan1.3` (SDK 1.4.313.2):
///
/// ```glsl
/// #version 460
/// #extension GL_EXT_buffer_reference : require
/// #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
/// layout(local_size_x = 64) in;
/// layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint value; };
/// layout(push_constant) uniform Push { Words words; uint64_t never; } push;
/// void main()
/// {
///     uint64_t count = 0ul;
///     do
///     {
///         count += 1ul;
///     } while (count != push.never);
///     push.words.value = uint(count);
/// }
/// ```
const NEVER_ENDS_SPIRV: [u32; 299] = [
    0x07230203, 0x00010600, 0x0008000b, 0x0000002b, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x0000000b, 0x00020011, 0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x000014e4, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00000017, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001cc, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72,
    0x65726566, 0x0065636e, 0x000d0004, 0x455f4c47, 0x735f5458, 0x65646168, 0x78655f72, 0x63696c70,
    0x615f7469, 0x68746972, 0x6974656d, 0x79745f63, 0x5f736570, 0x36746e69, 0x00000034, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000008, 0x6e756f63, 0x00000074, 0x00040005,
    0x00000013, 0x68737550, 0x00000000, 0x00050006, 0x00000013, 0x00000000, 0x64726f77, 0x00000073,
    0x00050006, 0x00000013, 0x00000001, 0x6576656e, 0x00000072, 0x00040005, 0x00000015, 0x64726f57,
    0x00000073, 0x00050006, 0x00000015, 0x00000000, 0x756c6176, 0x00000065, 0x00040005, 0x00000017,
    0x68737570, 0x00000000, 0x00030047, 0x00000013, 0x00000002, 0x00050048, 0x00000013, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x00000013, 0x00000001, 0x00000023, 0x00000008, 0x00030047,
    0x00000015, 0x00000002, 0x00050048, 0x00000015, 0x00000000, 0x00000023, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000040, 0x00000000,
    0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x0005002b, 0x00000006, 0x00000009, 0x00000000,
    0x00000000, 0x0005002b, 0x00000006, 0x0000000e, 0x00000001, 0x00000000, 0x00030027, 0x00000012,
    0x000014e5, 0x0004001e, 0x00000013, 0x00000012, 0x00000006, 0x00040015, 0x00000014, 0x00000020,
    0x00000000, 0x0003001e, 0x00000015, 0x00000014, 0x00040020, 0x00000012, 0x000014e5, 0x00000015,
    0x00040020, 0x00000016, 0x00000009, 0x00000013, 0x0004003b, 0x00000016, 0x00000017, 0x00000009,
    0x00040015, 0x00000018, 0x00000020, 0x00000001, 0x0004002b, 0x00000018, 0x00000019, 0x00000001,
    0x00040020, 0x0000001a, 0x00000009, 0x00000006, 0x00020014, 0x0000001d, 0x0004002b, 0x00000018,
    0x0000001f, 0x00000000, 0x00040020, 0x00000020, 0x00000009, 0x00000012, 0x00040020, 0x00000025,
    0x000014e5, 0x00000014, 0x00040017, 0x00000027, 0x00000014, 0x00000003, 0x0004002b, 0x00000014,
    0x00000028, 0x00000040, 0x0004002b, 0x00000014, 0x00000029, 0x00000001, 0x0006002c, 0x00000027,
    0x0000002a, 0x00000028, 0x00000029, 0x00000029, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0003003e,
    0x00000008, 0x00000009, 0x000200f9, 0x0000000a, 0x000200f8, 0x0000000a, 0x000400f6, 0x0000000c,
    0x0000000d, 0x00000000, 0x000200f9, 0x0000000b, 0x000200f8, 0x0000000b, 0x0004003d, 0x00000006,
    0x0000000f, 0x00000008, 0x00050080, 0x00000006, 0x00000010, 0x0000000f, 0x0000000e, 0x0003003e,
    0x00000008, 0x00000010, 0x000200f9, 0x0000000d, 0x000200f8, 0x0000000d, 0x0004003d, 0x00000006,
    0x00000011, 0x00000008, 0x00050041, 0x0000001a, 0x0000001b, 0x00000017, 0x00000019, 0x0004003d,
    0x00000006, 0x0000001c, 0x0000001b, 0x000500ab, 0x0000001d, 0x0000001e, 0x00000011, 0x0000001c,
    0x000400fa, 0x0000001e, 0x0000000a, 0x0000000c, 0x000200f8, 0x0000000c, 0x00050041, 0x00000020,
    0x00000021, 0x00000017, 0x0000001f, 0x0004003d, 0x00000012, 0x00000022, 0x00000021, 0x0004003d,
    0x00000006, 0x00000023, 0x00000008, 0x00040071, 0x00000014, 0x00000024, 0x00000023, 0x00050041,
    0x00000025, 0x00000026, 0x00000022, 0x0000001f, 0x0005003e, 0x00000026, 0x00000024, 0x00000002,
    0x00000004, 0x000100fd, 0x00010038,
];

/// A compute kernel that faults: one workgroup of 64 threads reading a word through the address
/// `freed` (a buffer's, destroyed before the dispatch) and writing it, plus one, through `words`.
/// Compiled as [`NEVER_ENDS_SPIRV`] is:
///
/// ```glsl
/// #version 460
/// #extension GL_EXT_buffer_reference : require
/// layout(local_size_x = 64) in;
/// layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint value; };
/// layout(push_constant) uniform Push { Words words; Words freed; } push;
/// void main()
/// {
///     push.words.value = push.freed.value + 1u;
/// }
/// ```
const FAULTS_SPIRV: [u32; 214] = [
    0x07230203, 0x00010600, 0x0008000b, 0x0000001d, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001cc, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72, 0x65726566, 0x0065636e,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000007, 0x68737550, 0x00000000,
    0x00050006, 0x00000007, 0x00000000, 0x64726f77, 0x00000073, 0x00050006, 0x00000007, 0x00000001,
    0x65657266, 0x00000064, 0x00040005, 0x00000009, 0x64726f57, 0x00000073, 0x00050006, 0x00000009,
    0x00000000, 0x756c6176, 0x00000065, 0x00040005, 0x0000000b, 0x68737570, 0x00000000, 0x00030047,
    0x00000007, 0x00000002, 0x00050048, 0x00000007, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
    0x00000007, 0x00000001, 0x00000023, 0x00000008, 0x00030047, 0x00000009, 0x00000002, 0x00050048,
    0x00000009, 0x00000000, 0x00000023, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030027, 0x00000006, 0x000014e5, 0x0004001e, 0x00000007, 0x00000006, 0x00000006,
    0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0003001e, 0x00000009, 0x00000008, 0x00040020,
    0x00000006, 0x000014e5, 0x00000009, 0x00040020, 0x0000000a, 0x00000009, 0x00000007, 0x0004003b,
    0x0000000a, 0x0000000b, 0x00000009, 0x00040015, 0x0000000c, 0x00000020, 0x00000001, 0x0004002b,
    0x0000000c, 0x0000000d, 0x00000000, 0x00040020, 0x0000000e, 0x00000009, 0x00000006, 0x0004002b,
    0x0000000c, 0x00000011, 0x00000001, 0x00040020, 0x00000014, 0x000014e5, 0x00000008, 0x0004002b,
    0x00000008, 0x00000017, 0x00000001, 0x00040017, 0x0000001a, 0x00000008, 0x00000003, 0x0004002b,
    0x00000008, 0x0000001b, 0x00000040, 0x0006002c, 0x0000001a, 0x0000001c, 0x0000001b, 0x00000017,
    0x00000017, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x00050041, 0x0000000e, 0x0000000f, 0x0000000b, 0x0000000d, 0x0004003d, 0x00000006, 0x00000010,
    0x0000000f, 0x00050041, 0x0000000e, 0x00000012, 0x0000000b, 0x00000011, 0x0004003d, 0x00000006,
    0x00000013, 0x00000012, 0x00050041, 0x00000014, 0x00000015, 0x00000013, 0x0000000d, 0x0006003d,
    0x00000008, 0x00000016, 0x00000015, 0x00000002, 0x00000004, 0x00050080, 0x00000008, 0x00000018,
    0x00000016, 0x00000017, 0x00050041, 0x00000014, 0x00000019, 0x00000010, 0x0000000d, 0x0005003e,
    0x00000019, 0x00000018, 0x00000002, 0x00000004, 0x000100fd, 0x00010038,
];

/// A mesh shader that draws nothing (`SetMeshOutputsEXT(0, 0)`, 32 threads, triangles), compiled as
/// [`NEVER_ENDS_SPIRV`] is with `-S mesh`.
const DRAWS_NOTHING_SPIRV: [u32; 100] = [
    0x07230203, 0x00010600, 0x0008000b, 0x0000000c, 0x00000000, 0x00020011, 0x000014a3, 0x0006000a,
    0x5f565053, 0x5f545845, 0x6873656d, 0x6168735f, 0x00726564, 0x0006000b, 0x00000001, 0x4c534c47,
    0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x000014f5,
    0x00000004, 0x6e69616d, 0x00000000, 0x0006014b, 0x00000004, 0x00000026, 0x00000007, 0x00000008,
    0x00000008, 0x00040010, 0x00000004, 0x0000001a, 0x00000003, 0x00040010, 0x00000004, 0x00001496,
    0x00000001, 0x00030010, 0x00000004, 0x000014b2, 0x00030003, 0x00000002, 0x000001cc, 0x00060004,
    0x455f4c47, 0x6d5f5458, 0x5f687365, 0x64616873, 0x00007265, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x0004002b, 0x00000006, 0x00000007, 0x00000020, 0x0004002b, 0x00000006,
    0x00000008, 0x00000001, 0x0004002b, 0x00000006, 0x00000009, 0x00000000, 0x00040017, 0x0000000a,
    0x00000006, 0x00000003, 0x0006002c, 0x0000000a, 0x0000000b, 0x00000007, 0x00000008, 0x00000008,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000314af,
    0x00000009, 0x00000009, 0x000100fd, 0x00010038,
];

/// A lost device's objects destroyed in the spec's order, each step timed (NEW-010's close: the
/// RTX 4070's driver spun in `vkDestroyCommandPool` after a lost device, past 70 s): a kernel that
/// never ends ([`NEVER_ENDS_SPIRV`]) dispatched on the preferred adapter, so Windows' watchdog
/// resets the device and the fence's wait returns `ERROR_DEVICE_LOST`; then the fences waited for
/// again, the device waited for idle, and everything destroyed, the device last. A driver report's
/// repro: it loses the device, so run it alone, by hand, with a kill past 90 s:
/// `cargo test --release -p vrs --test gpu -- --exact a_lost_device_s_objects_are_destroyed_in_order
/// --include-ignored --nocapture`.
///
/// `LOST_DEVICE_REPRO_STEPS` (0 without it) adds what our engine does, the first that many, in
/// order:
/// 1. two submissions in flight from two command pools, each with its fence: the kernel that never
///    ends, then a short one, the loss reported waiting for the first;
/// 2. a timestamp query pool for each, reset and written around its commands;
/// 3. fills, updates and copies of buffers in each, between barriers;
/// 4. a mesh pipeline's draw of nothing in attachment-less dynamic rendering in each;
/// 5. several GB of buffers held;
/// 6. the pipeline cache's data read after the loss;
/// 7. a fault in place of the kernel that never ends: a read through a destroyed buffer's address
///    ([`FAULTS_SPIRV`]).
#[test]
#[ignore = "loses the device: a driver report's repro, run by hand"]
fn a_lost_device_s_objects_are_destroyed_in_order() {
    use std::io::Write;
    use std::time::Instant;
    let steps: u32 = std::env::var("LOST_DEVICE_REPRO_STEPS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(0);
    let started = Instant::now();
    // Each step said as it ends, with its time and the seconds since the test started.
    let said = |step: &str, began: Instant| {
        println!(
            "{step}: {:.3} s (at {:.3} s)",
            began.elapsed().as_secs_f64(),
            started.elapsed().as_secs_f64()
        );
        let _ = std::io::stdout().flush();
    };
    let mut app = headless_app();
    let device = preferred_device(&mut app);
    let properties = device.adapter.properties();
    println!(
        "{}: driver version {:#x}, Vulkan {}.{}.{}; steps {steps}",
        device.adapter.name(),
        properties.driver_version,
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
        vk::api_version_patch(properties.api_version)
    );
    let raw = &device.device;
    let first = OneShot::new(&device);
    let second = (steps >= 1).then(|| OneShot::new(&device));
    let words = AddressedBuffer::new(&device.adapter, raw, 4, vk::BufferUsageFlags::empty())
        .expect("a buffer");
    let cache = PipelineCache::new(raw, &[]).expect("a pipeline cache");
    let never_ends = ComputePipelines::new(
        &device.adapter,
        raw,
        &NEVER_ENDS_SPIRV,
        &[c"main"],
        16,
        &cache,
        None,
    )
    .expect("the kernel's pipeline");
    let faults = (steps >= 7).then(|| {
        ComputePipelines::new(
            &device.adapter,
            raw,
            &FAULTS_SPIRV,
            &[c"main"],
            16,
            &cache,
            None,
        )
        .expect("the faulting kernel's pipeline")
    });
    let timestamps = (steps >= 2).then(|| {
        [
            TimestampPool::new(&device, 512).expect("a timestamp pool"),
            TimestampPool::new(&device, 512).expect("a timestamp pool"),
        ]
    });
    let scratch = (steps >= 3).then(|| {
        [
            AddressedBuffer::new(&device.adapter, raw, 1 << 20, vk::BufferUsageFlags::empty())
                .expect("a scratch buffer"),
            AddressedBuffer::new(&device.adapter, raw, 1 << 20, vk::BufferUsageFlags::empty())
                .expect("a scratch buffer"),
        ]
    });
    let mesh = (steps >= 4).then(|| {
        gpu::MeshPipeline::new(
            raw,
            &gpu::MeshPipelineDesc {
                task: None,
                mesh: gpu::ShaderStage {
                    words: &DRAWS_NOTHING_SPIRV,
                    entry: c"main",
                },
                fragment: None,
                push_constant_size: 16,
                color_formats: &[],
                depth_format: None,
                depth: gpu::DepthState {
                    test: false,
                    write: false,
                    compare: vk::CompareOp::ALWAYS,
                },
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            },
            &cache,
        )
        .expect("the mesh pipeline")
    });
    let held: Vec<AddressedBuffer> = if steps >= 5 {
        (0..6)
            .map(|_| {
                AddressedBuffer::new(
                    &device.adapter,
                    raw,
                    512 << 20,
                    vk::BufferUsageFlags::empty(),
                )
                .expect("512 MiB held")
            })
            .collect()
    } else {
        Vec::new()
    };
    // The faulting kernel's address: a buffer's, destroyed before the dispatch.
    let freed = (steps >= 7).then(|| {
        let buffer =
            AddressedBuffer::new(&device.adapter, raw, 1 << 20, vk::BufferUsageFlags::empty())
                .expect("a buffer to free");
        let address = buffer.address();
        unsafe { buffer.destroy(raw) };
        address
    });
    // Each submission's commands: its timestamps reset, the copies, the dispatch, the draw, a
    // timestamp written after each.
    let record = |cmd: vk::CommandBuffer,
                  which: usize,
                  pipelines: &ComputePipelines,
                  push: [u64; 2]| unsafe {
        let mut stamp = 0u32;
        let mut stamped = |cmd: vk::CommandBuffer| {
            if let Some(pools) = &timestamps {
                pools[which].cmd_write(raw, cmd, stamp);
                stamp += 1;
            }
        };
        if let Some(pools) = &timestamps {
            pools[which].cmd_reset(raw, cmd);
        }
        stamped(cmd);
        if let Some([from, to]) = &scratch {
            commands::cmd_fill_buffer(raw, cmd, from.handle(), 0, vk::WHOLE_SIZE, which as u32);
            sync::cmd_pipeline_barrier(raw, cmd);
            commands::cmd_update_buffer(
                raw,
                cmd,
                from.handle(),
                0,
                bytemuck::cast_slice(&[7u32; 1024]),
            );
            sync::cmd_pipeline_barrier(raw, cmd);
            commands::cmd_copy_buffer(raw, cmd, from.handle(), 0, to.handle(), 0, 1 << 20);
            sync::cmd_pipeline_barrier(raw, cmd);
            stamped(cmd);
        }
        commands::cmd_dispatch(
            raw,
            cmd,
            pipelines.pipeline(0),
            pipelines.layout(),
            bytemuck::bytes_of(&push),
            1,
            1,
        );
        sync::cmd_pipeline_barrier(raw, cmd);
        stamped(cmd);
        if let Some(mesh) = &mesh {
            let extent = vk::Extent2D {
                width: 64,
                height: 64,
            };
            commands::cmd_begin_rendering(raw, cmd, extent, None, None);
            commands::cmd_set_viewport(raw, cmd, extent);
            commands::cmd_set_scissor(raw, cmd, extent);
            commands::cmd_bind_pipeline(raw, cmd, mesh.pipeline());
            commands::cmd_push_constants(raw, cmd, mesh.layout(), bytemuck::bytes_of(&push));
            commands::cmd_draw_mesh_tasks(&device.mesh_shader, cmd, 1);
            commands::cmd_end_rendering(raw, cmd);
            stamped(cmd);
        }
    };
    // The first submission loses the device (the kernel that never ends, or the fault); the second,
    // behind it, is short.
    let (lost_pipelines, lost_push) = match (&faults, freed) {
        (Some(faults), Some(freed)) => (faults, [words.address(), freed]),
        _ => (&never_ends, [words.address(), 0u64]),
    };
    let began = Instant::now();
    unsafe {
        commands::begin_command_buffer(raw, first.command_buffer).expect("recording");
        record(first.command_buffer, 0, lost_pipelines, lost_push);
        commands::submit(&device, first.command_buffer, first.fence, &[], &[])
            .expect("the submission");
        if let Some(second) = &second {
            commands::begin_command_buffer(raw, second.command_buffer).expect("recording");
            record(
                second.command_buffer,
                1,
                &never_ends,
                [words.address(), 1u64],
            );
            commands::submit(&device, second.command_buffer, second.fence, &[], &[])
                .expect("the second submission");
        }
    }
    let ran = unsafe { sync::wait_for_fence(raw, first.fence) };
    said(&format!("the first submission's wait ({ran:?})"), began);
    if let Some(second) = &second {
        let began = Instant::now();
        let waited = unsafe { sync::wait_for_fence(raw, second.fence) };
        said(&format!("the second submission's wait ({waited:?})"), began);
    }
    let began = Instant::now();
    let waited = unsafe { sync::wait_for_fence(raw, first.fence) };
    said(
        &format!("the first fence waited for again ({waited:?})"),
        began,
    );
    let began = Instant::now();
    let idle = unsafe { sync::wait_for_device_idle(raw) };
    said(&format!("the device waited for idle ({idle:?})"), began);
    if steps >= 6 {
        let began = Instant::now();
        let data = cache.data(raw);
        said(
            &format!(
                "the pipeline cache's data read ({:?} bytes)",
                data.map(|bytes| bytes.len())
            ),
            began,
        );
    }
    let began = Instant::now();
    unsafe {
        never_ends.destroy(raw);
        if let Some(faults) = &faults {
            faults.destroy(raw);
        }
        if let Some(mesh) = &mesh {
            mesh.destroy(raw);
        }
    }
    said("the pipelines destroyed", began);
    let began = Instant::now();
    unsafe { cache.destroy(raw) };
    said("the pipeline cache destroyed", began);
    let began = Instant::now();
    unsafe {
        words.destroy(raw);
        for buffer in scratch.iter().flatten().chain(&held) {
            buffer.destroy(raw);
        }
    }
    said("the buffers destroyed", began);
    if let Some(pools) = &timestamps {
        let began = Instant::now();
        unsafe {
            for pool in pools {
                pool.destroy(raw);
            }
        }
        said("the timestamp pools destroyed", began);
    }
    for (name, one_shot) in [("first", Some(&first)), ("second", second.as_ref())] {
        let Some(one_shot) = one_shot else {
            continue;
        };
        let began = Instant::now();
        unsafe { raw.destroy_fence(one_shot.fence, None) };
        said(&format!("the {name} fence destroyed"), began);
        let began = Instant::now();
        unsafe { raw.destroy_command_pool(one_shot.pool, None) };
        said(&format!("the {name} command pool destroyed"), began);
    }
    let began = Instant::now();
    let destroyed = unsafe { app.destroy_device(&device) };
    said(&format!("the device destroyed ({destroyed:?})"), began);
    let began = Instant::now();
    drop(app);
    said("the instance destroyed", began);
    assert_eq!(
        ran,
        Err(vk::Result::ERROR_DEVICE_LOST),
        "the first submission lost the device"
    );
}
