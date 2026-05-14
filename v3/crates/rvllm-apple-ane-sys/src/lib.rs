//! Unsafe Objective-C runtime boundary for private ANE experiments.
//!
//! This crate is intentionally separate from `rvllm-apple`. The safe Apple
//! crate owns planning and handoff contracts; this sys crate owns raw runtime
//! probing and returns typed `rvllm-core` errors instead of falling back.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::c_void;
use std::ffi::CStr;
use std::ptr::NonNull;

use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

const BACKEND: &str = "private-ane";
#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane")))]
const SYS_BACKEND: &str = "rvllm-apple-ane-sys";
const DEVICE: &str = "apple-silicon";

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
mod objc;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane")))]
mod unavailable;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
use objc as imp;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane")))]
use unavailable as imp;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct ObjcClass(NonNull<c_void>);

impl ObjcClass {
    #[must_use]
    pub fn as_ptr(self) -> *mut c_void {
        self.0.as_ptr()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
    fn new(ptr: NonNull<c_void>) -> Self {
        Self(ptr)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct ObjcSelector(NonNull<c_void>);

impl ObjcSelector {
    #[must_use]
    pub fn as_ptr(self) -> *mut c_void {
        self.0.as_ptr()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
    fn new(ptr: NonNull<c_void>) -> Self {
        Self(ptr)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PrivateAneRuntime {
    _private: (),
}

impl PrivateAneRuntime {
    pub fn open() -> Result<Self> {
        imp::open_runtime()
    }

    pub fn get_class(self, name: &CStr) -> Result<ObjcClass> {
        imp::get_class(name)
    }

    pub fn register_selector(self, name: &CStr) -> Result<ObjcSelector> {
        imp::register_selector(name)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
    fn new() -> Self {
        Self { _private: () }
    }
}

#[must_use]
pub const fn objective_c_ffi_enabled() -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "private-ane"
    ))
}

pub fn open_private_ane_runtime() -> Result<PrivateAneRuntime> {
    PrivateAneRuntime::open()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane")))]
fn feature_not_available(op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: SYS_BACKEND,
            op,
        },
        apple_ctx(op),
    )
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
fn private_api_unavailable(symbol: &'static str, op: &'static str) -> RvllmError {
    RvllmError::apple(AppleError::PrivateApiUnavailable { symbol }, apple_ctx(op))
}

fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: BACKEND,
        op,
        device: DEVICE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{AppleError, RvllmError};

    #[test]
    fn unavailable_gate_returns_typed_apple_error() {
        if objective_c_ffi_enabled() {
            return;
        }

        let Err(err) = PrivateAneRuntime::open() else {
            panic!("private ANE runtime must be unavailable without the explicit cfg gate");
        };

        match err {
            RvllmError::Apple {
                err:
                    AppleError::FeatureNotAvailable {
                        backend: "rvllm-apple-ane-sys",
                        op: "open_private_ane_runtime",
                    },
                ctx,
                ..
            } => {
                assert_eq!(ctx.backend, "private-ane");
                assert_eq!(ctx.op, "open_private_ane_runtime");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
