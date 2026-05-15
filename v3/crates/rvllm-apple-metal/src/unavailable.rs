use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "metal",
        op,
        device: "apple-silicon",
    }
}

static EMPTY_UNIT: () = ();

#[derive(Clone, Debug)]
pub struct MetalRegion {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Stub Metal context unavailable on non-macOS targets.
#[derive(Debug)]
pub struct MetalContext;

impl MetalContext {
    pub fn new() -> Result<Self> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "init",
            },
            ctx("init"),
        ))
    }

    pub fn compile_library(&mut self, _source: &str) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "compile_library",
            },
            ctx("compile_library"),
        ))
    }

    pub fn load_metallib(&mut self, _path: &std::path::Path) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "load_metallib",
            },
            ctx("load_metallib"),
        ))
    }

    pub fn make_pipeline(&self, _function_name: &str) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "make_pipeline",
            },
            ctx("make_pipeline"),
        ))
    }
    pub fn device(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn device_retained(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn queue(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn queue_retained(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn library(&self) -> Option<&()> {
        None
    }
    pub fn target(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn max_threadgroup_memory(&self) -> usize {
        0
    }
}

/// Stub metal arena unavailable on non-macOS targets.
#[derive(Debug)]
pub struct MetalBufferArena;

impl MetalBufferArena {
    pub fn new(_device: &(), _capacity_bytes: usize) -> Result<Self> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "arena_alloc",
            },
            ctx("arena_new"),
        ))
    }

    pub fn region(&mut self, _name: &str, _bytes: usize, _align: usize) -> Result<MetalRegion> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "arena_region",
            },
            ctx("arena_region"),
        ))
    }

    pub fn host_ptr(&self, _region: &MetalRegion) -> *mut u8 {
        std::ptr::null_mut()
    }
    pub unsafe fn write_region(&self, _region: &MetalRegion, _data: &[u8]) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "arena_write",
            },
            ctx("arena_write"),
        ))
    }
    pub fn buffer(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn buffer_retained(&self) -> &() {
        &EMPTY_UNIT
    }
    pub fn allocated(&self) -> usize {
        0
    }
    pub fn capacity(&self) -> usize {
        0
    }
    pub fn remaining(&self) -> usize {
        0
    }
    pub fn regions(&self) -> &[MetalRegion] {
        static EMPTY: [MetalRegion; 0] = [];
        &EMPTY
    }
    pub fn reset(&mut self) {}
}

/// Stub pipeline cache unavailable on non-macOS targets.
#[derive(Debug, Default)]
pub struct PipelineCache;

impl PipelineCache {
    pub fn new() -> Self {
        Self
    }

    pub fn compile(&mut self, _ctx: &MetalContext, _function_name: &str) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "compile_pipeline",
            },
            ctx("compile_pipeline"),
        ))
    }

    pub fn compile_all(&mut self, _ctx: &MetalContext) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "compile_pipelines",
            },
            ctx("compile_pipelines"),
        ))
    }

    pub fn get(&self, _name: &str) -> Result<&()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "pipeline_get",
            },
            ctx("pipeline_get"),
        ))
    }

    pub fn len(&self) -> usize {
        0
    }
    pub fn is_empty(&self) -> bool {
        true
    }
}
