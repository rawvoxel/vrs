use super::GpuContext;
use crate::{App, vulkan::sync};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

/// A swapchain on the app's window that frames are blitted into: an image is acquired, a
/// command buffer blits a rendered image into it, and it is presented.
///
/// Acquiring waits on nothing on the host: each frame in flight has its own acquire semaphore,
/// which its submission waits on, and each swapchain image its own render-done semaphore, which
/// the submission signals and presenting waits on.
pub struct Presenter {
    device: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    acquired: Vec<vk::Semaphore>,
    rendered: Vec<vk::Semaphore>,
    vsync: bool,
}

/// A swapchain image acquired for a frame.
#[derive(Clone, Copy, Debug)]
pub struct Acquired {
    pub index: u32,
    pub image: vk::Image,
    /// The submission that renders into it waits on this (at the transfer stage)...
    pub wait: vk::Semaphore,
    /// ...and signals this, which presenting waits on.
    pub signal: vk::Semaphore,
}

impl Presenter {
    /// A swapchain of `extent` (or the window's, when the surface decides) for `context`, which
    /// must have been made with `GpuOptions { present: true }` on `app`'s window, with an
    /// acquire semaphore for each of `frames_in_flight` frames.
    pub fn new(
        app: &App,
        context: &GpuContext,
        extent: vk::Extent2D,
        vsync: bool,
        frames_in_flight: usize,
    ) -> VkResult<Self> {
        let device = khr::swapchain::Device::new(app.instance(), context.device());
        let mut acquired = Vec::new();
        for _ in 0..frames_in_flight {
            acquired.push(sync::create_semaphore(context.device())?);
        }
        let mut presenter = Self {
            device,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent,
            acquired,
            rendered: Vec::new(),
            vsync,
        };
        presenter.create(app, context, extent)?;
        Ok(presenter)
    }

    fn create(&mut self, app: &App, context: &GpuContext, extent: vk::Extent2D) -> VkResult<()> {
        let surface = app.surface().ok_or(vk::Result::ERROR_SURFACE_LOST_KHR)?;
        let adapter = context.adapter().handle();
        let (capabilities, formats, present_modes) = unsafe {
            (
                app.surface_instance
                    .get_physical_device_surface_capabilities(adapter, surface)?,
                app.surface_instance
                    .get_physical_device_surface_formats(adapter, surface)?,
                app.surface_instance
                    .get_physical_device_surface_present_modes(adapter, surface)?,
            )
        };
        let format = formats
            .iter()
            .find(|format| {
                matches!(
                    format.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                ) && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
            .to_owned();
        let present_mode = if self.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
                .into_iter()
                .find(|mode| present_modes.contains(mode))
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };
        let extent = match capabilities.current_extent.width {
            u32::MAX => extent,
            _ => capabilities.current_extent,
        };
        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let old = self.swapchain;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);
        self.swapchain = unsafe { self.device.create_swapchain(&info, None)? };
        unsafe {
            if old != vk::SwapchainKHR::null() {
                self.device.destroy_swapchain(old, None);
            }
            for &semaphore in &self.rendered {
                context.device().destroy_semaphore(semaphore, None);
            }
        }
        self.rendered.clear();
        self.images = unsafe { self.device.get_swapchain_images(self.swapchain)? };
        for _ in &self.images {
            self.rendered
                .push(sync::create_semaphore(context.device())?);
        }
        self.format = format.format;
        self.extent = extent;
        Ok(())
    }

    /// Makes the swapchain again at `extent`, once the device is done with the old one (after
    /// a resize, or when presenting said it's out of date).
    pub fn recreate(
        &mut self,
        app: &App,
        context: &GpuContext,
        extent: vk::Extent2D,
    ) -> VkResult<()> {
        unsafe { context.device().device_wait_idle()? };
        self.create(app, context, extent)
    }

    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Acquires the next image for frame slot `frame`; `None` when the swapchain is out of date
    /// and has to be made again first.
    pub fn acquire(&mut self, frame: usize) -> VkResult<Option<Acquired>> {
        let wait = self.acquired[frame % self.acquired.len()];
        match unsafe {
            self.device
                .acquire_next_image(self.swapchain, u64::MAX, wait, vk::Fence::null())
        } {
            Ok((index, _suboptimal)) => Ok(Some(Acquired {
                index,
                image: self.images[index as usize],
                wait,
                signal: self.rendered[index as usize],
            })),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Records `source` (in `TRANSFER_SRC_OPTIMAL`, `source_extent` texels) blitted over the
    /// whole acquired image with `filter`, leaving it ready to present.
    ///
    /// # Safety
    /// In a command buffer that is recording, outside any rendering.
    pub unsafe fn cmd_blit(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        acquired: &Acquired,
        source: vk::Image,
        source_extent: vk::Extent2D,
        filter: vk::Filter,
    ) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let to_destination = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(acquired.image)
            .subresource_range(range);
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let corner = |extent: vk::Extent2D| vk::Offset3D {
            x: extent.width as i32,
            y: extent.height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit2::default()
            .src_subresource(layers)
            .src_offsets([vk::Offset3D::default(), corner(source_extent)])
            .dst_subresource(layers)
            .dst_offsets([vk::Offset3D::default(), corner(self.extent)]);
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(acquired.image)
            .subresource_range(range);
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(slice::from_ref(&to_destination)),
            );
            device.cmd_blit_image2(
                cmd,
                &vk::BlitImageInfo2::default()
                    .src_image(source)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .dst_image(acquired.image)
                    .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .regions(slice::from_ref(&blit))
                    .filter(filter),
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&to_present)),
            );
        }
    }

    /// Presents the acquired image once its submission has signaled; `false` when the swapchain
    /// is out of date or suboptimal and should be made again.
    pub fn present(&self, context: &GpuContext, acquired: &Acquired) -> VkResult<bool> {
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&acquired.signal))
            .swapchains(slice::from_ref(&self.swapchain))
            .image_indices(slice::from_ref(&acquired.index));
        match context.with_queue(|queue| unsafe { self.device.queue_present(queue, &info) }) {
            Ok(false) => Ok(true),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for &semaphore in self.acquired.iter().chain(&self.rendered) {
                device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_swapchain(self.swapchain, None);
        }
        self.acquired.clear();
        self.rendered.clear();
    }
}
