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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalFloatType {
    F16,
    Bf16,
}

impl MetalFloatType {
    #[must_use]
    pub const fn report_name(self) -> &'static str {
        match self {
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
        }
    }
}

/// Compile-time platform surface for the Metal crate.
///
/// Metal primitives and model-package loading are portable across Apple OSes.
/// Environment-driven development loading remains a macOS-only policy of the
/// runtime crate; iOS callers must provide explicit bundle/model paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalPlatformCapabilities {
    pub metal_primitives: bool,
    pub layer_encoding: bool,
    pub model_package_loading: bool,
    pub development_environment_loading: bool,
}

impl MetalPlatformCapabilities {
    #[must_use]
    pub const fn current() -> Self {
        let apple_os = cfg!(any(target_os = "macos", target_os = "ios"));
        Self {
            metal_primitives: apple_os,
            layer_encoding: apple_os,
            model_package_loading: apple_os,
            development_environment_loading: cfg!(target_os = "macos"),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod arena;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod context;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod gemma4_model;
pub mod kernels;
pub mod options;
pub use options::{MetalKernelOptions, MetalModelLimits};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod layer_forward;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod low_bit_metal;
pub mod memory_budget;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod pipeline;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod unavailable;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod weight_loader;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use arena::MetalBufferArena;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use context::{MetalContext, MetalDeviceCapabilities};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use low_bit_metal::{
    LowBitMetalError, LowBitMetalResult, MetalLowBitProjection, MetalLowBitProjectionOffsets,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use pipeline::PipelineCache;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use unavailable::MetalBufferArena;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use unavailable::MetalContext;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use unavailable::PipelineCache;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use unavailable::MetalRegion;

pub use rvllm_apple::{
    dequantize_apple_low_bit_reference, project_apple_low_bit_reference,
    quantize_apple_low_bit_reference, AppleLowBitWeightFormat, LowBitWeightError,
    PackedAppleLowBitWeights, APPLE_LOW_BIT_GROUP_SIZE, APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
};

#[cfg(test)]
mod platform_tests {
    use super::MetalPlatformCapabilities;

    #[test]
    fn platform_capability_contract_is_fail_closed() {
        let capabilities = MetalPlatformCapabilities::current();
        assert_eq!(
            capabilities.metal_primitives,
            cfg!(any(target_os = "macos", target_os = "ios"))
        );
        assert_eq!(capabilities.layer_encoding, capabilities.metal_primitives);
        assert_eq!(
            capabilities.development_environment_loading,
            cfg!(target_os = "macos")
        );
        assert!(
            !capabilities.model_package_loading || capabilities.metal_primitives,
            "model loading must never be advertised without real Metal primitives"
        );
        assert_eq!(
            capabilities.model_package_loading,
            capabilities.metal_primitives
        );
    }
}
