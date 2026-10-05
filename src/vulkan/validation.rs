use ash::{ext::debug_utils, prelude::VkResult, vk};
use std::{
    borrow::Cow,
    ffi,
    sync::atomic::{AtomicU32, Ordering},
};

static ERROR_COUNT: AtomicU32 = AtomicU32::new(0);
static WARNING_COUNT: AtomicU32 = AtomicU32::new(0);

/// Validation messages reported since the program started, across every `App`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessageCounts {
    pub errors: u32,
    pub warnings: u32,
}

/// For tests and tools, which can fail on validation messages instead of a person reading them.
/// Always zero without validation, see `App::validation_enabled`.
pub fn message_counts() -> MessageCounts {
    MessageCounts {
        errors: ERROR_COUNT.load(Ordering::Relaxed),
        warnings: WARNING_COUNT.load(Ordering::Relaxed),
    }
}

#[derive(Clone, Copy)]
pub struct Messenger {
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Messenger {
    pub fn new(debug_utils_instance: &debug_utils::Instance) -> VkResult<Self> {
        let messenger = create_messenger(debug_utils_instance)?;
        Ok(Self { messenger })
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, debug_utils_instance: &debug_utils::Instance) {
        unsafe {
            debug_utils_instance.destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

#[inline]
pub fn create_messenger(
    debug_utils_instance: &debug_utils::Instance,
) -> VkResult<vk::DebugUtilsMessengerEXT> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    unsafe { debug_utils_instance.create_debug_utils_messenger(&debug_info, None) }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        ERROR_COUNT.fetch_add(1, Ordering::Relaxed);
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        WARNING_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    unsafe {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;
        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };
        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }
}
