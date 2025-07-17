use ash::vk;

#[inline]
pub fn create_semaphores(logical_device: &ash::Device, count: u32) -> Vec<vk::Semaphore> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    (0..count)
        .map(|_| unsafe {
            logical_device
                .create_semaphore(&semaphore_info, None)
                .expect("Could not create vk::Semaphore.")
        })
        .collect()
}

#[inline]
pub fn create_fences(logical_device: &ash::Device, count: u32) -> Vec<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    (0..count)
        .map(|_| unsafe {
            logical_device
                .create_fence(&fence_info, None)
                .expect("Could not create vk::Fence.")
        })
        .collect()
}
