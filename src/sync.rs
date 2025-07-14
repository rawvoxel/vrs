use ash::{Device, prelude::VkResult, vk};

#[inline]
pub fn create_semaphores(device: &Device, count: u32) -> VkResult<Vec<vk::Semaphore>> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let semaphores: Vec<vk::Semaphore> = (0..count)
        .into_iter()
        .map(|_| unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .expect("Could not create vk::Semaphore.")
        })
        .collect();
    Ok(semaphores)
}

#[inline]
pub fn create_fences(device: &Device, count: u32) -> VkResult<Vec<vk::Fence>> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let fences = (0..count)
        .into_iter()
        .map(|_| unsafe {
            device
                .create_fence(&fence_info, None)
                .expect("Could not create vk::Fence.")
        })
        .collect();
    Ok(fences)
}
