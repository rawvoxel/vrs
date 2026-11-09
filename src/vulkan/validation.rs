//! The Khronos validation layer, turned on by an environment variable ([`VARIABLE`]): its errors
//! and warnings go through the `log` facade, and are counted for tests to fail on.

use ash::{ext::debug_utils, prelude::VkResult, vk};
use std::{
    borrow::Cow,
    ffi,
    sync::atomic::{AtomicU32, Ordering},
};

/// The environment variable that turns validation on for every `App` made after it's set, to any
/// value but `0`, when the layer is installed.
pub const VARIABLE: &str = "VRS_VALIDATION";

/// The validation errors known and not yet fixed, which [`assert_no_new_errors`] lets pass, each
/// by its message ID: none.
pub const KNOWN_ERRORS: [&str; 0] = [];

static ERROR_COUNT: AtomicU32 = AtomicU32::new(0);
static WARNING_COUNT: AtomicU32 = AtomicU32::new(0);
/// The errors not among [`KNOWN_ERRORS`].
static NEW_ERROR_COUNT: AtomicU32 = AtomicU32::new(0);

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

/// For tests: fails if the validation layer reported any error since the program started, across
/// every `App`, other than the [`KNOWN_ERRORS`]. Passes without validation, see
/// `App::validation_enabled`.
///
/// # Panics
/// When it reported one; each was logged as it came.
pub fn assert_no_new_errors() {
    let new_errors = NEW_ERROR_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        new_errors, 0,
        "the validation layer reported {new_errors} errors besides the known ones"
    );
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
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
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
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;
    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };
    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        ERROR_COUNT.fetch_add(1, Ordering::Relaxed);
        if !KNOWN_ERRORS.contains(&message_id_name.as_ref()) {
            NEW_ERROR_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        log::error!("{message_type:?} [{message_id_name} ({message_id_number})]: {message}");
    } else {
        WARNING_COUNT.fetch_add(1, Ordering::Relaxed);
        log::warn!("{message_type:?} [{message_id_name} ({message_id_number})]: {message}");
    }
    vk::FALSE
}
