//! Metal device context: device discovery, command queue, and library management.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};
use rvllm_apple::device::{AppleAcceleratorTarget, AppleGpuFamily};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

/// Metal device context. Owns the device, command queue, and compiled
/// shader library. Created once at engine init; shared (immutably) by
/// all inference operations.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    target: AppleAcceleratorTarget,
}

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "metal",
        op,
        device: "apple-silicon",
    }
}

impl MetalContext {
    /// Create a Metal context using the system default device.
    ///
    /// Fails with `MetalUnavailable` if no Metal device is found, or if
    /// the GPU family is below Apple7 (pre-M1 hardware).
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RvllmError::apple(AppleError::MetalUnavailable, ctx("init")))?;

        let name = device.name().to_string();
        let target = AppleAcceleratorTarget::from_device_name(&name, 1);

        if target.gpu_family == AppleGpuFamily::Unknown {
            return Err(RvllmError::apple(
                AppleError::UnsupportedDevice { name: "unknown" },
                ctx("init"),
            ));
        }

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RvllmError::apple(AppleError::MetalUnavailable, ctx("create_queue")))?;

        tracing::info!(
            device = %name,
            gpu_family = ?target.gpu_family,
            tier = ?target.tier,
            ane_cores = target.ane_cores,
            "Metal context initialized"
        );

        Ok(Self {
            device,
            queue,
            library: None,
            target,
        })
    }

    /// Compile Metal Shading Language source into a library.
    pub fn compile_library(&mut self, source: &str) -> Result<()> {
        let ns_source = NSString::from_str(source);
        let lib = self
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| {
                tracing::error!(error = %e, "Metal shader compilation failed");
                RvllmError::apple(
                    AppleError::MilCompileFailed {
                        procedure: "metallib",
                    },
                    ctx("compile_library"),
                )
            })?;
        self.library = Some(lib);
        Ok(())
    }

    /// Load a pre-compiled .metallib file.
    pub fn load_metallib(&mut self, path: &std::path::Path) -> Result<()> {
        let ns_path = NSString::from_str(&path.to_string_lossy());
        let url = unsafe { objc2_foundation::NSURL::fileURLWithPath(&ns_path) };
        let lib = self.device.newLibraryWithURL_error(&url).map_err(|_| {
            RvllmError::apple(
                AppleError::MetallibMissing {
                    path: path.to_path_buf(),
                },
                ctx("load_metallib"),
            )
        })?;
        self.library = Some(lib);
        Ok(())
    }

    /// Create a compute pipeline state object from a named function.
    pub fn make_pipeline(
        &self,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let lib = self.library.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::PipelineMissing { name: "no_library" },
                ctx("make_pipeline"),
            )
        })?;
        let ns_name = NSString::from_str(function_name);
        let func = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::PipelineMissing { name: "unknown" },
                ctx("get_function"),
            )
        })?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|_| {
                RvllmError::apple(
                    AppleError::PipelineMissing { name: "unknown" },
                    ctx("compile_pso"),
                )
            })?;
        Ok(pso)
    }

    #[inline]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &*self.device
    }

    #[inline]
    pub fn device_retained(&self) -> &Retained<ProtocolObject<dyn MTLDevice>> {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &*self.queue
    }

    #[inline]
    pub fn queue_retained(&self) -> &Retained<ProtocolObject<dyn MTLCommandQueue>> {
        &self.queue
    }

    #[inline]
    pub fn library(&self) -> Option<&ProtocolObject<dyn MTLLibrary>> {
        self.library.as_deref()
    }

    #[inline]
    pub fn target(&self) -> &AppleAcceleratorTarget {
        &self.target
    }

    /// Maximum threadgroup memory in bytes (Apple9: 32KB, Apple10+: 64KB).
    pub fn max_threadgroup_memory(&self) -> usize {
        if self.target.has_nax {
            65536
        } else {
            32768
        }
    }
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.target.device_name)
            .field("gpu_family", &self.target.gpu_family)
            .field("tier", &self.target.tier)
            .finish()
    }
}
