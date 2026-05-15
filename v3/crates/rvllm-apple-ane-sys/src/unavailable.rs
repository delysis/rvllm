#[derive(Clone)]
pub struct AneRequest;

impl AneRequest {
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(
        _inputs: &[AneSurface],
        _input_indices: &[u64],
        _outputs: &[AneSurface],
        _output_indices: &[u64],
        _procedure_index: u64,
    ) -> Option<Self> {
        None
    }
}

#[derive(Clone)]
pub struct AneSurface;

impl AneSurface {
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(_width: usize, _height: usize, _pixel_size: usize) -> Option<Self> {
        None
    }

    #[allow(clippy::unnecessary_wraps)]
    pub fn from_id(_id: u32) -> Option<Self> {
        None
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    pub fn read_u32(&self, _offset_bytes: usize) -> u32 {
        0
    }
}

#[derive(Clone)]
pub struct AneModelHandle;

impl AneModelHandle {
    #[allow(clippy::unnecessary_wraps)]
    pub fn load(_path: &str) -> Option<Self> {
        None
    }

    pub fn evaluate(&self, _request: &AneRequest) -> Result<(), String> {
        Err("private-ane unavailable on this platform".to_string())
    }
}

pub fn load_frameworks() -> Result<(), String> {
    Err("private-ane unavailable on non-macOS/aarch64 target".to_string())
}

