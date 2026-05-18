#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use ffi::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    use super::*;
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
            let ns_inputs =
                create_ns_array(&inputs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>());
            let ns_input_indices = create_ns_array(
                &input_indices
                    .iter()
                    .map(|&i| create_ns_number_u64(i))
                    .collect::<Vec<_>>(),
            );
            let ns_outputs =
                create_ns_array(&outputs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>());
            let ns_output_indices = create_ns_array(
                &output_indices
                    .iter()
                    .map(|&i| create_ns_number_u64(i))
                    .collect::<Vec<_>>(),
            );

            create_ane_request(
                &ns_inputs,
                &ns_input_indices,
                &ns_outputs,
                &ns_output_indices,
                procedure_index,
            )
            .map(|inner| Self { inner })
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
            self.try_read_u32(offset_bytes).unwrap_or_default()
        }

        pub fn try_read_u32(&self, offset_bytes: usize) -> Result<u32, String> {
            read_iosurface_u32(self.as_ptr(), offset_bytes)
        }

        pub fn try_read_f32(&self, offset_elements: usize) -> Result<f32, String> {
            read_iosurface_f32(self.as_ptr(), offset_elements)
        }

        pub fn write_f32(&self, offset_elements: usize, value: f32) -> Result<(), String> {
            write_iosurface_f32(self.as_ptr(), offset_elements, value)
        }
    }

    #[derive(Clone)]
    pub struct AneModelHandle {
        pub client: Retained<AnyObject>,
        pub model: Retained<AnyObject>,
    }

    impl AneModelHandle {
        pub fn load(path: &str) -> Option<Self> {
            Self::load_with_error(path).ok()
        }

        pub fn load_with_error(path: &str) -> Result<Self, String> {
            let client = get_ane_client().ok_or_else(|| "_ANEClient unavailable".to_string())?;
            let model = compile_and_load_ane_model(path, &client)?;
            Ok(Self { client, model })
        }

        pub fn evaluate(&self, request: &AneRequest) -> Result<(), String> {
            evaluate_ane_request(&self.client, &self.model, &request.inner)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use platform::*;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod unavailable;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub use unavailable::*;
