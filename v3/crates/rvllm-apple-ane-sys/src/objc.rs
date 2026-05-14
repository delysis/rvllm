use core::ffi::{c_char, c_void};
use std::ffi::CStr;
use std::ptr::NonNull;

use rvllm_core::Result;

use crate::{private_api_unavailable, ObjcClass, ObjcSelector, PrivateAneRuntime};

#[link(name = "objc")]
extern "C" {
    #[link_name = "objc_getClass"]
    fn raw_objc_get_class(name: *const c_char) -> *mut c_void;

    #[link_name = "sel_registerName"]
    fn raw_sel_register_name(name: *const c_char) -> *mut c_void;

    #[allow(dead_code)]
    pub fn objc_msgSend();
}

pub(crate) fn open_runtime() -> Result<PrivateAneRuntime> {
    Ok(PrivateAneRuntime::new())
}

pub(crate) fn get_class(name: &CStr) -> Result<ObjcClass> {
    let ptr = unsafe { raw_objc_get_class(name.as_ptr()) };
    NonNull::new(ptr)
        .map(ObjcClass::new)
        .ok_or_else(|| private_api_unavailable("objc_getClass", "objc_getClass"))
}

pub(crate) fn register_selector(name: &CStr) -> Result<ObjcSelector> {
    let ptr = unsafe { raw_sel_register_name(name.as_ptr()) };
    NonNull::new(ptr)
        .map(ObjcSelector::new)
        .ok_or_else(|| private_api_unavailable("sel_registerName", "sel_registerName"))
}
