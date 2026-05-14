//! Compute pipeline state object cache.
//!
//! Compiles Metal functions to PSOs once at init. PSOs are keyed by
//! function name. No compilation happens during inference.

use std::collections::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
use crate::context::MetalContext;
use rvllm_core::Result;

/// Cached compute pipeline state objects, keyed by function name.
pub struct PipelineCache {
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self { pipelines: HashMap::new() }
    }

    /// Compile a named function from the context's library into a PSO.
    pub fn compile(&mut self, ctx: &MetalContext, function_name: &str) -> Result<()> {
        let pso = ctx.make_pipeline(function_name)?;
        self.pipelines.insert(function_name.to_owned(), pso);
        tracing::debug!(function = function_name, "Compiled Metal PSO");
        Ok(())
    }

    /// Compile all required kernel functions for inference.
    pub fn compile_all(&mut self, ctx: &MetalContext) -> Result<()> {
        let required = crate::kernels::KERNEL_NAMES;
        for name in required {
            self.compile(ctx, name)?;
        }
        tracing::info!(count = required.len(), "All Metal PSOs compiled");
        Ok(())
    }

    /// Get a cached PSO by name.
    pub fn get(&self, name: &str) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.pipelines.get(name).ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::PipelineMissing { name: "unknown" },
                rvllm_core::AppleCtx { backend: "metal", op: "get_pso", device: "apple-silicon" },
            )
        })
    }

    pub fn len(&self) -> usize { self.pipelines.len() }
    pub fn is_empty(&self) -> bool { self.pipelines.is_empty() }
}

impl Default for PipelineCache {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("count", &self.pipelines.len())
            .field("functions", &self.pipelines.keys().collect::<Vec<_>>())
            .finish()
    }
}
