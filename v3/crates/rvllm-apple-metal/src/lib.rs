//! rvllm-apple-metal: Metal GPU compute backend for Apple Silicon inference.
//!
//! This crate implements the Metal compute pipeline for rvLLM on Apple
//! Silicon. It provides:
//! - Device detection and command queue management
//! - Persistent buffer arena with zero hot-path allocation
//! - Compute shader pipeline compilation and caching
//! - Layer-forward execution for transformer decoder blocks
//!
//! All Metal FFI is isolated here. The parent `rvllm-apple` crate
//! remains safe and host-testable.

#[cfg(target_os = "macos")]
pub mod arena;
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod gemma4_model;
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod layer_forward;
#[cfg(target_os = "macos")]
pub mod pipeline;
#[cfg(not(target_os = "macos"))]
mod unavailable;
#[cfg(target_os = "macos")]
pub mod weight_loader;

#[cfg(target_os = "macos")]
pub use arena::MetalBufferArena;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use pipeline::PipelineCache;
#[cfg(not(target_os = "macos"))]
pub use unavailable::MetalBufferArena;
#[cfg(not(target_os = "macos"))]
pub use unavailable::MetalContext;
#[cfg(not(target_os = "macos"))]
pub use unavailable::PipelineCache;

#[cfg(not(target_os = "macos"))]
pub use unavailable::MetalRegion;
