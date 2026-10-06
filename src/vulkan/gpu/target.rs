use crate::vulkan::{
    adapter::Adapter,
    image::{Image, create_image_view},
};
use ash::{prelude::VkResult, vk};

/// A 2D image to render into, or to copy and blit between, with its view, in device memory.
pub struct RenderImage {
    image: Image,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl RenderImage {
    /// Owns nothing; destroying it does nothing.
    pub fn null() -> Self {
        Self {
            image: Image::null(),
            view: vk::ImageView::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
        }
    }

    /// `extent` texels of `format`, usable for `usage`, viewed through `aspect`.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> VkResult<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = Image::new(adapter, device, info, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        match create_image_view(device, image.handle(), format, Default::default(), aspect) {
            Ok(view) => Ok(Self {
                image,
                view,
                format,
                extent,
            }),
            Err(error) => {
                unsafe { image.destroy(device) };
                Err(error)
            }
        }
    }

    /// A depth attachment of `format` (such as `D32_SFLOAT`), which can also be sampled.
    pub fn depth(
        adapter: &Adapter,
        device: &ash::Device,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> VkResult<Self> {
        Self::new(
            adapter,
            device,
            extent,
            format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
        )
    }

    #[inline]
    pub fn image(&self) -> vk::Image {
        self.image.handle()
    }

    #[inline]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            self.image.destroy(device);
        }
    }
}
