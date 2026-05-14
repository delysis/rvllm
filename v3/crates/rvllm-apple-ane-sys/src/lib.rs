#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(apple_silicon)]
pub mod ffi;

#[cfg(apple_silicon)]
pub use ffi::*;
