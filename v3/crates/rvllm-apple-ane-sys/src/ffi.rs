use std::ffi::CString;
use objc2::{class, msg_send};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

// Load the private framework into the process.
pub fn load_ane_framework() -> Result<(), String> {
    let path = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine").unwrap();
    let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        return Err("Failed to dlopen AppleNeuralEngine.framework".to_string());
    }
    Ok(())
}

pub unsafe fn get_ane_client() -> Option<Retained<AnyObject>> {
    let cls = class!(_ANEClient);
    unsafe { msg_send![cls, sharedConnection] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_ane() {
        load_ane_framework().expect("ANE framework should load on Apple Silicon");
        let client = unsafe { get_ane_client() };
        println!("Got ANEClient: {:?}", client);
        assert!(client.is_some());
    }
}
