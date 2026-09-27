//! Explicit, bounded GPU capture for diagnostic binaries. Never in normal inference.

use std::path::Path;

use objc2::rc::Retained;
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use crate::MetalContext;

/// Stops capture even when the inference operation returns early.
pub struct MetalGpuCapture {
    manager: Retained<MTLCaptureManager>,
    active: bool,
}

fn validate_output(path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.extension() != Some(std::ffi::OsStr::new("gputrace")) {
        return Err("GPU capture path must be absolute and end in .gputrace".into());
    }
    if path.exists() {
        return Err(format!(
            "GPU capture output already exists: {}",
            path.display()
        ));
    }
    if !path.parent().map_or(false, Path::is_dir) {
        return Err("GPU capture parent directory does not exist".into());
    }
    Ok(())
}

impl MetalGpuCapture {
    pub fn start(context: &MetalContext, path: &Path) -> Result<Self, String> {
        validate_output(path)?;
        // Apple's shared manager is process-owned; the retained wrapper keeps it alive
        // for the complete capture scope. No raw object pointer escapes this call.
        let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err("GPU trace document capture is unsupported on this device".into());
        }
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(context.device());
        let output_url = NSURL::from_file_path(path)
            .ok_or_else(|| "GPU capture output path is not a valid file URL".to_owned())?;
        descriptor.setOutputURL(Some(&output_url));
        manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|error| format!("start GPU capture: {error}"))?;
        Ok(Self {
            manager,
            active: true,
        })
    }

    /// Close the capture before inspecting it. A successful Metal API call alone
    /// does not prove that a replayable trace was written.
    pub fn finish(mut self, path: &Path) -> Result<(), String> {
        self.manager.stopCapture();
        self.active = false;
        let metadata = std::fs::metadata(path).map_err(|error| {
            format!(
                "GPU capture produced no trace at {}: {error}",
                path.display()
            )
        })?;
        let nonempty = if metadata.is_file() {
            metadata.len() > 0
        } else if metadata.is_dir() {
            std::fs::read_dir(path)
                .map_err(|error| format!("read GPU trace: {error}"))?
                .next()
                .is_some()
        } else {
            false
        };
        nonempty
            .then_some(())
            .ok_or_else(|| "GPU capture output is empty or invalid".into())
    }
}

impl Drop for MetalGpuCapture {
    fn drop(&mut self) {
        if self.active {
            self.manager.stopCapture();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_output;
    use std::path::Path;

    #[test]
    fn rejects_relative_and_wrong_extension() {
        assert!(validate_output(Path::new("relative.gputrace")).is_err());
        assert!(validate_output(Path::new("/tmp/trace.txt")).is_err());
    }
}
