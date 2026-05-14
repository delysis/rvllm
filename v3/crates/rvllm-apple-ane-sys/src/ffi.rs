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
        
        let dump_class = |cls_name: &str| {
            let cstr = std::ffi::CString::new(cls_name).unwrap();
            let cls = unsafe { objc2::runtime::AnyClass::get(&cstr) };
            if let Some(cls) = cls {
                println!("\n--- Methods for {} ---", cls_name);
                let mut count = 0;
                let methods = unsafe { objc2::ffi::class_copyMethodList(cls as *const _ as *const _, &mut count) };
                if !methods.is_null() {
                    for i in 0..count {
                        let m = unsafe { *methods.add(i as usize) };
                        let sel = unsafe { objc2::ffi::method_getName(m) };
                        if let Some(sel) = sel {
                            let sel_name = unsafe { std::ffi::CStr::from_ptr(objc2::ffi::sel_getName(sel)) };
                            println!("Method: {}", sel_name.to_string_lossy());
                        }
                    }
                    unsafe { libc::free(methods as *mut _) };
                }
            } else {
                println!("Class {} not found", cls_name);
            }
        };

        dump_class("_ANEModel");
        dump_class("_ANECompiler");
        dump_class("_ANEProgramForEvaluation");
    }
}
