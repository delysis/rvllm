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

#[cfg(not(target_os = "macos"))]
compile_error!("rvllm-apple-metal requires macOS (Apple Silicon)");

pub mod context;
pub mod arena;
pub mod pipeline;
pub mod kernels;
pub mod layer_forward;
pub mod weight_loader;

pub use context::MetalContext;
pub use arena::MetalBufferArena;
pub use pipeline::PipelineCache;
