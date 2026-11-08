//! Command pools and buffers; beginning, submitting and waiting for them; and recording each
//! command the engine uses, with that command's rules checked in debug builds.

use crate::vulkan::{
    compute::{HostAccess, HostBuffer},
    gpu::{MeshPipeline, VulkanDevice},
    queue::QueueFamily,
    sync,
};
use ash::{ext, prelude::VkResult, vk};
use bytemuck::AnyBitPattern;
use std::{mem, slice};

#[inline]
pub fn create_command_pool(
    device: &ash::Device,
    queue_family: &QueueFamily,
) -> VkResult<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family.index);

    unsafe { device.create_command_pool(&info, None) }
}

/// Destroys `pool` and the command buffers allocated from it.
///
/// # Safety
///
/// `pool` was made on `device`, and none of its command buffers is still pending.
#[inline]
pub unsafe fn destroy_command_pool(device: &ash::Device, pool: vk::CommandPool) {
    unsafe { device.destroy_command_pool(pool, None) };
}

#[inline]
pub fn create_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    level: vk::CommandBufferLevel,
) -> VkResult<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .command_buffer_count(1)
        .level(level);

    unsafe { Ok(device.allocate_command_buffers(&allocate_info)?[0]) }
}

/// Records a barrier that makes every device write recorded before it visible to the host once the
/// submission's fence has signaled: a fence makes device writes available, not visible to the host.
/// A submission the host waits on to read what it wrote ends with it.
#[inline]
pub fn record_host_read_barrier(device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
    unsafe {
        sync::cmd_memory_barrier(
            device,
            cmd_buffer,
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
        )
    };
}

/// Begins recording `command_buffer` for one submission, after resetting it: whatever it held,
/// even a recording left unfinished, is gone. Its pool must let single buffers be reset, as
/// [`create_command_pool`]'s do.
///
/// # Safety
/// The device is done with `command_buffer`: it isn't pending.
#[inline]
pub unsafe fn begin_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
) -> VkResult<()> {
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(command_buffer, &begin_info)
    }
}

/// Ends recording `command_buffer` and submits it to `device`'s queue, waiting first on each of
/// `wait`'s semaphores at its stage and signaling `signal`'s semaphores and `fence` once the
/// device has run it. `fence` is reset first, signaled or not. On an error nothing is submitted,
/// and `fence` won't be signaled.
///
/// # Safety
/// `command_buffer` is recording ([`begin_command_buffer`]), and no submission still to finish
/// uses `fence`. Vulkan requires submits to a queue to be externally synchronized: no one else
/// submits to `device`'s queue meanwhile.
pub unsafe fn submit(
    device: &VulkanDevice,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
    signal: &[vk::Semaphore],
) -> VkResult<()> {
    let (wait_semaphores, wait_stages): (Vec<_>, Vec<_>) = wait.iter().copied().unzip();
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(slice::from_ref(&command_buffer))
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .signal_semaphores(signal);
    unsafe {
        device.device.end_command_buffer(command_buffer)?;
        device.device.reset_fences(slice::from_ref(&fence))?;
        device
            .device
            .queue_submit(device.queue.handle(), slice::from_ref(&submit_info), fence)
    }
}

/// Records commands with `record` into `command_buffer`, submits them and waits until the device
/// has run them, their writes then visible to the host ([`record_host_read_barrier`]): a blocking
/// one-shot, on a command buffer and fence its owner keeps for them.
///
/// # Safety
/// The device is done with `command_buffer` and `fence`, and no one else submits to `device`'s
/// queue meanwhile ([`submit`]).
pub unsafe fn submit_and_wait(
    device: &VulkanDevice,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    record: impl FnOnce(vk::CommandBuffer),
) -> VkResult<()> {
    unsafe {
        begin_command_buffer(&device.device, command_buffer)?;
        record(command_buffer);
        record_host_read_barrier(&device.device, command_buffer);
        submit(device, command_buffer, fence, &[], &[])?;
        sync::wait_for_fence(&device.device, fence)
    }
}

/// Copies `data`'s size from byte `offset` of `source` back into `data`, blocking: staged
/// through host memory of its own, after everything submitted to the queue before it, copied,
/// waited for and read ([`submit_and_wait`] on `command_buffer` and `fence`).
///
/// # Safety
/// As [`submit_and_wait`]; the bytes lie inside `source`, made with `TRANSFER_SRC`.
pub unsafe fn read_back<T: AnyBitPattern>(
    device: &VulkanDevice,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    source: vk::Buffer,
    offset: vk::DeviceSize,
    data: &mut [T],
) -> VkResult<()> {
    let size = mem::size_of_val(data) as vk::DeviceSize;
    if size == 0 {
        return Ok(());
    }
    let staging = HostBuffer::new(
        &device.adapter,
        &device.device,
        size,
        HostAccess::Readback,
        vk::BufferUsageFlags::empty(),
    )?;
    let read = unsafe {
        submit_and_wait(device, command_buffer, fence, |command_buffer| {
            sync::cmd_pipeline_barrier(&device.device, command_buffer);
            cmd_copy_buffer(
                &device.device,
                command_buffer,
                source,
                offset,
                staging.handle(),
                0,
                size,
            );
        })
    }
    .and_then(|()| staging.read(&device.device, 0, data));
    unsafe { staging.destroy(&device.device) };
    read
}

/// Records filling `size` bytes of `buffer` from byte `offset` with the word `data`, repeated:
/// `offset` a multiple of 4, and `size` a multiple of 4 or [`vk::WHOLE_SIZE`] for the rest of
/// the buffer (its whole words).
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; the bytes lie inside `buffer`, made with
/// `TRANSFER_DST`.
#[inline]
pub unsafe fn cmd_fill_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    data: u32,
) {
    debug_assert!(
        offset.is_multiple_of(4),
        "a fill starts at a multiple of 4 bytes"
    );
    debug_assert!(
        size == vk::WHOLE_SIZE || (size > 0 && size.is_multiple_of(4)),
        "a fill's size is a multiple of 4 bytes, or the whole size"
    );
    unsafe { device.cmd_fill_buffer(command_buffer, buffer, offset, size, data) };
}

/// The most bytes one `vkCmdUpdateBuffer` writes.
const UPDATE_LIMIT: usize = 65536;

/// Records writing `data` into `buffer` from byte `offset`, in updates of 64 KiB, the most one
/// writes: `offset` and `data`'s size multiples of 4. Empty `data` records nothing.
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; the bytes lie inside `buffer`, made with
/// `TRANSFER_DST`.
#[inline]
pub unsafe fn cmd_update_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    data: &[u8],
) {
    debug_assert!(
        offset.is_multiple_of(4),
        "an update starts at a multiple of 4 bytes"
    );
    debug_assert!(
        data.len().is_multiple_of(4),
        "an update's size is a multiple of 4 bytes"
    );
    for (number, piece) in data.chunks(UPDATE_LIMIT).enumerate() {
        let at = offset + (number * UPDATE_LIMIT) as vk::DeviceSize;
        unsafe { device.cmd_update_buffer(command_buffer, buffer, at, piece) };
    }
}

/// Records copying `size` bytes from byte `source_offset` of `source` to byte
/// `destination_offset` of `destination`: `size` not zero, and within one buffer the two ranges
/// apart.
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; the bytes lie inside the buffers,
/// `source` made with `TRANSFER_SRC` and `destination` with `TRANSFER_DST`.
#[inline]
pub unsafe fn cmd_copy_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    source: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination: vk::Buffer,
    destination_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) {
    debug_assert!(size > 0, "a copy's size isn't zero");
    debug_assert!(
        source != destination
            || source_offset.saturating_add(size) <= destination_offset
            || destination_offset.saturating_add(size) <= source_offset,
        "a copy within one buffer doesn't overlap itself"
    );
    let region = vk::BufferCopy::default()
        .src_offset(source_offset)
        .dst_offset(destination_offset)
        .size(size);
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            source,
            destination,
            slice::from_ref(&region),
        )
    };
}

/// Binds the compute `pipeline` and pushes `push_constants` (none when empty, else a multiple of
/// 4 bytes, from offset 0) through its `layout`.
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; `layout`'s push range for compute
/// shaders holds `push_constants`.
#[inline]
unsafe fn bind_compute(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_constants: &[u8],
) {
    debug_assert!(
        push_constants.len().is_multiple_of(4),
        "push constants are a multiple of 4 bytes"
    );
    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        if !push_constants.is_empty() {
            device.cmd_push_constants(
                command_buffer,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
    }
}

/// Records the compute `pipeline` run over `group_count_x` × `group_count_y` workgroups, after
/// binding it and pushing `push_constants` through its `layout` (none when empty, else a multiple
/// of 4 bytes, from offset 0).
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; `layout`'s push range for compute
/// shaders holds `push_constants`; the counts are within the device's
/// `maxComputeWorkGroupCount`.
#[inline]
pub unsafe fn cmd_dispatch(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_constants: &[u8],
    group_count_x: u32,
    group_count_y: u32,
) {
    unsafe {
        bind_compute(device, command_buffer, pipeline, layout, push_constants);
        device.cmd_dispatch(command_buffer, group_count_x, group_count_y, 1);
    }
}

/// Records the compute `pipeline` run over the workgroups at byte `offset` of `buffer` (a
/// `VkDispatchIndirectCommand`, `offset` a multiple of 4), after binding it and pushing
/// `push_constants` through its `layout` as [`cmd_dispatch`] does.
///
/// # Safety
/// As [`cmd_dispatch`]; the command lies inside `buffer`, made with `INDIRECT_BUFFER`.
#[inline]
pub unsafe fn cmd_dispatch_indirect(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_constants: &[u8],
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
) {
    debug_assert!(
        offset.is_multiple_of(4),
        "an indirect dispatch's command starts at a multiple of 4 bytes"
    );
    unsafe {
        bind_compute(device, command_buffer, pipeline, layout, push_constants);
        device.cmd_dispatch_indirect(command_buffer, buffer, offset);
    }
}

/// Binds the graphics `pipeline`, such as a [`MeshPipeline`]'s.
///
/// # Safety
/// `command_buffer` is recording.
#[inline]
pub unsafe fn cmd_bind_pipeline(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
) {
    unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline) };
}

/// Pushes `push_constants` to a [`MeshPipeline`]'s stages ([`MeshPipeline::PUSH_STAGES`])
/// through its `layout`, from offset 0: a multiple of 4 bytes, not none.
///
/// # Safety
/// `command_buffer` is recording; `layout`'s push range holds `push_constants`.
#[inline]
pub unsafe fn cmd_push_constants(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    push_constants: &[u8],
) {
    debug_assert!(
        !push_constants.is_empty() && push_constants.len().is_multiple_of(4),
        "push constants are a multiple of 4 bytes, not none"
    );
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            layout,
            MeshPipeline::PUSH_STAGES,
            0,
            push_constants,
        )
    };
}

/// Records drawing `group_count_x` task workgroups (mesh workgroups without a task shader) with
/// the bound mesh pipeline.
///
/// # Safety
/// `command_buffer` is recording, inside rendering, with a mesh pipeline bound; the count is
/// within the device's task (or mesh) workgroup count limits.
#[inline]
pub unsafe fn cmd_draw_mesh_tasks(
    mesh_shader: &ext::mesh_shader::Device,
    command_buffer: vk::CommandBuffer,
    group_count_x: u32,
) {
    unsafe { mesh_shader.cmd_draw_mesh_tasks(command_buffer, group_count_x, 1, 1) };
}

/// Records one draw of the workgroups at byte `offset` of `buffer` (a
/// `VkDrawMeshTasksIndirectCommandEXT`, `offset` a multiple of 4) with the bound mesh pipeline.
///
/// # Safety
/// As [`cmd_draw_mesh_tasks`]; the command lies inside `buffer`, made with `INDIRECT_BUFFER`.
#[inline]
pub unsafe fn cmd_draw_mesh_tasks_indirect(
    mesh_shader: &ext::mesh_shader::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
) {
    debug_assert!(
        offset.is_multiple_of(4),
        "an indirect draw's command starts at a multiple of 4 bytes"
    );
    let stride = mem::size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as u32;
    unsafe { mesh_shader.cmd_draw_mesh_tasks_indirect(command_buffer, buffer, offset, 1, stride) };
}

/// Begins rendering over `extent` (not empty) into the attachments given, or none at all, when
/// only what the fragments do matters:
/// - `depth`: a depth view in `DEPTH_ATTACHMENT_OPTIMAL`, cleared to the depth given or loaded
///   when there's none;
/// - `color`: a color view in `COLOR_ATTACHMENT_OPTIMAL`, cleared to the color given.
///
/// Both are stored.
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; the views hold `extent`, and the
/// pipelines drawn have the attachments' formats.
#[inline]
pub unsafe fn cmd_begin_rendering(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    depth: Option<(vk::ImageView, Option<f32>)>,
    color: Option<(vk::ImageView, vk::ClearColorValue)>,
) {
    debug_assert!(
        extent.width > 0 && extent.height > 0,
        "rendering covers some area"
    );
    let depth_attachment = depth.map(|(view, clear)| {
        vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(match clear {
                Some(_) => vk::AttachmentLoadOp::CLEAR,
                None => vk::AttachmentLoadOp::LOAD,
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear.unwrap_or(0.0),
                    stencil: 0,
                },
            })
    });
    let color_attachment = color.map(|(view, clear)| {
        vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: clear })
    });
    let mut rendering = vk::RenderingInfo::default()
        .render_area(extent.into())
        .layer_count(1)
        .color_attachments(color_attachment.as_slice());
    if let Some(depth_attachment) = &depth_attachment {
        rendering = rendering.depth_attachment(depth_attachment);
    }
    unsafe { device.cmd_begin_rendering(command_buffer, &rendering) };
}

/// Ends the rendering [`cmd_begin_rendering`] began.
///
/// # Safety
/// `command_buffer` is recording, inside rendering it began.
#[inline]
pub unsafe fn cmd_end_rendering(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    unsafe { device.cmd_end_rendering(command_buffer) };
}

/// Sets the viewport to all of `extent` (not empty), depths 0 to 1.
///
/// # Safety
/// `command_buffer` is recording.
#[inline]
pub unsafe fn cmd_set_viewport(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
) {
    debug_assert!(
        extent.width > 0 && extent.height > 0,
        "a viewport covers some area"
    );
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    unsafe { device.cmd_set_viewport(command_buffer, 0, slice::from_ref(&viewport)) };
}

/// Sets the scissor to all of `extent`.
///
/// # Safety
/// `command_buffer` is recording.
#[inline]
pub unsafe fn cmd_set_scissor(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
) {
    let scissor = vk::Rect2D::from(extent);
    unsafe { device.cmd_set_scissor(command_buffer, 0, slice::from_ref(&scissor)) };
}

/// Records copying `extent` (not empty) of the color `image`, in `TRANSFER_SRC_OPTIMAL`, into
/// `buffer` from its start, rows top first and tightly packed.
///
/// # Safety
/// `command_buffer` is recording, outside any rendering; `image` holds `extent` and was made with
/// `TRANSFER_SRC`, and `buffer` holds its texels and was made with `TRANSFER_DST`.
#[inline]
pub unsafe fn cmd_copy_image_to_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    extent: vk::Extent2D,
    buffer: vk::Buffer,
) {
    debug_assert!(
        extent.width > 0 && extent.height > 0,
        "a copy covers some area"
    );
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(extent.into());
    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            slice::from_ref(&region),
        )
    };
}
