#[cfg(apple_silicon)]
pub mod ffi;

#[cfg(apple_silicon)]
pub use ffi::*;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;

#[derive(Clone)]
pub struct AneRequest {
    pub inner: Retained<AnyObject>,
}

impl AneRequest {
    pub fn new(
        inputs: &[AneSurface],
        input_indices: &[u64],
        outputs: &[AneSurface],
        output_indices: &[u64],
        procedure_index: u64,
    ) -> Option<Self> {
        let ns_inputs = create_ns_array(&inputs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>());
        let ns_input_indices = create_ns_array(&input_indices.iter().map(|&i| create_ns_number_u64(i)).collect::<Vec<_>>());
        let ns_outputs = create_ns_array(&outputs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>());
        let ns_output_indices = create_ns_array(&output_indices.iter().map(|&i| create_ns_number_u64(i)).collect::<Vec<_>>());
        
        create_ane_request(&ns_inputs, &ns_input_indices, &ns_outputs, &ns_output_indices, procedure_index).map(|inner| Self { inner })
    }
}

#[derive(Clone)]
pub struct AneSurface {
    pub inner: Retained<AnyObject>,
}

impl AneSurface {
    pub fn new(width: usize, height: usize, pixel_size: usize) -> Option<Self> {
        create_ane_iosurface(width, height, pixel_size).map(|inner| Self { inner })
    }

    pub fn from_id(id: u32) -> Option<Self> {
        get_ane_surface_from_id(id).map(|inner| Self { inner })
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        get_iosurface_from_object(&self.inner)
    }

    pub fn read_u32(&self, offset_bytes: usize) -> u32 {
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::ptr::read_unaligned(ptr.add(offset_bytes) as *const u32)
        }
    }
}

#[derive(Clone)]
pub struct AneModelHandle {
    pub client: Retained<AnyObject>,
    pub model: Retained<AnyObject>,
}

impl AneModelHandle {
    pub fn load(path: &str) -> Option<Self> {
        let client = get_ane_client()?;
        let model = compile_and_load_ane_model(path, &client)?;
        Some(Self { client, model })
    }

    pub fn evaluate(&self, request: &AneRequest) -> Result<(), String> {
        evaluate_ane_request(&self.client, &self.model, &request.inner)
    }
}
