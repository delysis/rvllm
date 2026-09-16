//! Compute pipeline state object cache.
//!
//! Compiles Metal functions to PSOs once at init. PSOs are keyed by
//! function name. No compilation happens during inference.

use crate::context::MetalContext;
use crate::{MetalFloatType, MetalKernelOptions};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
use rvllm_apple::device::AppleGpuFamily;
use rvllm_core::Result;
use std::collections::HashMap;

/// Cached compute pipeline state objects, keyed by function name.
pub struct PipelineCache {
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    gpu_family: AppleGpuFamily,
    float_type: Option<MetalFloatType>,
    kernel_options: MetalKernelOptions,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::with_kernel_options(MetalKernelOptions::from_development_environment())
    }

    #[must_use]
    pub fn with_kernel_options(kernel_options: MetalKernelOptions) -> Self {
        Self {
            pipelines: HashMap::new(),
            gpu_family: AppleGpuFamily::Unknown,
            float_type: None,
            kernel_options,
        }
    }

    #[must_use]
    pub const fn kernel_options(&self) -> MetalKernelOptions {
        self.kernel_options
    }

    /// Compile a named function from the context's library into a PSO.
    pub fn compile(&mut self, ctx: &MetalContext, function_name: &str) -> Result<()> {
        self.float_type = None;
        let pso = ctx.make_pipeline(function_name)?;
        self.pipelines.insert(function_name.to_owned(), pso);
        tracing::debug!(function = function_name, "Compiled Metal PSO");
        Ok(())
    }

    /// Compile one typed library. Piecemeal replacement clears its dtype so
    /// dtype-specific optimizations cannot use an unclassified pipeline cache.
    pub fn compile_all_for_type(
        &mut self,
        ctx: &MetalContext,
        dtype: MetalFloatType,
    ) -> Result<()> {
        self.compile_all(ctx)?;
        self.float_type = Some(dtype);
        Ok(())
    }

    #[must_use]
    pub const fn float_type(&self) -> Option<MetalFloatType> {
        self.float_type
    }

    /// Compile all required kernel functions for inference.
    pub fn compile_all(&mut self, ctx: &MetalContext) -> Result<()> {
        self.gpu_family = ctx.capabilities().gpu_family;
        let required = crate::kernels::KERNEL_NAMES;
        for name in required {
            self.compile(ctx, name)?;
        }
        tracing::info!(count = required.len(), "All Metal PSOs compiled");
        Ok(())
    }

    /// Get a cached PSO by name.
    pub fn get(
        &self,
        name: &str,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.pipelines.get(name).ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::PipelineMissing { name: "unknown" },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "get_pso",
                    device: "apple-silicon",
                },
            )
        })
    }

    pub fn len(&self) -> usize {
        self.pipelines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Live GPU family captured when the inference PSOs were compiled.
    ///
    /// Shape-specialized dispatch must fail back to portable kernels if the
    /// cache was assembled piecemeal and therefore has no device identity.
    #[must_use]
    pub const fn gpu_family(&self) -> AppleGpuFamily {
        self.gpu_family
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("count", &self.pipelines.len())
            .field("functions", &self.pipelines.keys().collect::<Vec<_>>())
            .finish()
    }
}
