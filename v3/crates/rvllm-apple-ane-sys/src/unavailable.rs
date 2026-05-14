use std::ffi::CStr;

use rvllm_core::Result;

use crate::{feature_not_available, ObjcClass, ObjcSelector, PrivateAneRuntime};

pub(crate) fn open_runtime() -> Result<PrivateAneRuntime> {
    Err(feature_not_available("open_private_ane_runtime"))
}

pub(crate) fn get_class(_name: &CStr) -> Result<ObjcClass> {
    Err(feature_not_available("objc_getClass"))
}

pub(crate) fn register_selector(_name: &CStr) -> Result<ObjcSelector> {
    Err(feature_not_available("sel_registerName"))
}
