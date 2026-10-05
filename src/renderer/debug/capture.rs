//! Copying a frame back from the device, for tests that check what was drawn.

use crate::{renderer::Renderer, vulkan::image};
use ash::{prelude::VkResult, vk};

/// A frame copied back from the device, see `Renderer::debug_capture_next_frame`.
#[derive(Clone, Debug)]
pub struct Capture {
    pub width: u32,
    pub height: u32,
    /// The color format, every format the renderer draws to has 4 bytes per pixel.
    pub format: vk::Format,
    /// Tightly packed rows, top row first.
    pub pixels: Vec<u8>,
}

impl Capture {
    /// The pixel's red, green, blue and alpha bytes, `None` outside the frame or
    /// for formats other than 8 bit RGBA and BGRA.
    pub fn rgba8(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let start = 4 * (y as usize * self.width as usize + x as usize);
        let [a, b, c, d] = self.pixels[start..start + 4].try_into().unwrap();
        match self.format {
            vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM => Some([a, b, c, d]),
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM => Some([c, b, a, d]),
            _ => None,
        }
    }

    /// Whether the bytes are sRGB encoded, so linear colors must be encoded to compare with them.
    #[inline]
    pub fn is_srgb(&self) -> bool {
        matches!(
            self.format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        )
    }
}

/// A frame whose copy was recorded, for `Renderer::debug_read_capture`.
pub(in crate::renderer) struct RecordedCapture {
    frame: usize,
    extent: vk::Extent2D,
}

impl Renderer {
    /// Copies the next submitted frame back from the device, to read with `debug_read_capture`.
    /// `false` when the window's images can't be copied.
    pub fn debug_capture_next_frame(&mut self) -> bool {
        self.capture_requested = self.surface.can_capture();
        self.capture_requested
    }

    /// The frame `debug_capture_next_frame` asked for, once it's been submitted.
    ///
    /// For tests and tools: waits for the device to go idle.
    pub fn debug_read_capture(&mut self) -> VkResult<Option<Capture>> {
        let Some(RecordedCapture { frame, extent }) = self.recorded_capture.take() else {
            return Ok(None);
        };
        unsafe { self.render_device.device_wait_idle()? };

        let mut pixels = vec![0u8; 4 * extent.width as usize * extent.height as usize];
        // SAFETY: Any byte is a valid `u8`.
        unsafe {
            self.capture_buffers[frame]
                .buffer
                .mem_read_at(0, &mut pixels)
        };
        Ok(Some(Capture {
            width: extent.width,
            height: extent.height,
            format: self.surface.format,
            pixels,
        }))
    }

    /// Records copying the rendered `image` into `frame`'s capture buffer, leaving it
    /// in `TRANSFER_SRC_OPTIMAL`.
    pub(in crate::renderer) fn cmd_capture(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        frame: usize,
        image: vk::Image,
    ) -> VkResult<()> {
        let extent = self.swapchain.extent;
        let byte_count = 4 * extent.width as vk::DeviceSize * extent.height as vk::DeviceSize;
        // NOTE: The fence wait at the start of the frame freed up this frame's buffer.
        unsafe {
            self.capture_buffers[frame].reserve(&self.adapter, &self.render_device, byte_count)?;
        }

        image::transition_layout(
            &self.render_device,
            image,
            cmd_buffer,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COPY,
            vk::ImageAspectFlags::COLOR,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(extent.into());
        unsafe {
            self.render_device.cmd_copy_image_to_buffer(
                cmd_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.capture_buffers[frame].buffer.handle,
                &[region],
            );
        }

        self.capture_requested = false;
        self.recorded_capture = Some(RecordedCapture { frame, extent });
        Ok(())
    }
}
