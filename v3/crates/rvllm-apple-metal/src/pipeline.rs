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
    max_threadgroup_memory: usize,
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
            max_threadgroup_memory: 0,
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
        self.max_threadgroup_memory = ctx.capabilities().max_threadgroup_memory_length;
        let required = crate::kernels::KERNEL_NAMES;
        for name in required {
            self.compile(ctx, name)?;
        }
        // Optional source may be absent from an explicitly supplied metallib.
        // Clear any old PSO before attempting replacement: a caught failure must
        // not leave a previous library's dtype or executable active.
        for name in self.kernel_options.research.pipeline_names() {
            self.pipelines.remove(*name);
            match self.compile(ctx, name) {
                Ok(()) => tracing::info!(candidate = self.kernel_options.research.name(),
                    function = *name, "Research PSO compiled; not hardware-qualified"),
                Err(error) => tracing::warn!(candidate = self.kernel_options.research.name(),
                    function = *name, %error, "Research PSO unavailable; known-good fallback retained"),
            }
        }
        tracing::info!(count = required.len(), "All required Metal PSOs compiled");
        Ok(())
    }

    /// No platform assumptions are inferred from a product name. A missing,
    /// piecemeal, untyped, wrong-family, or resource-incompatible PSO is refused.
    pub(crate) fn research_pso(
        &self,
        name: &str,
        threads: usize,
        planned_bytes: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if self.gpu_family != AppleGpuFamily::Apple9
            || self.float_type.is_none()
            || self.kernel_options.quantized_bf16_accumulation
            || !self.kernel_options.research.pipeline_names().contains(&name)
        {
            return None;
        }
        let pso = self.pipelines.get(name)?;
        crate::research::launch_fits(
            pso.threadExecutionWidth(), pso.maxTotalThreadsPerThreadgroup(),
            pso.staticThreadgroupMemoryLength(), self.max_threadgroup_memory,
            threads, planned_bytes,
        ).then_some(pso)
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
