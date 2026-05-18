use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use std::ffi::c_void;
use std::ffi::CString;

// Load the frameworks into the process.
pub fn load_frameworks() -> Result<(), String> {
    let ane_path = CString::new(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
    )
    .unwrap();
    let ane_handle = unsafe { libc::dlopen(ane_path.as_ptr(), libc::RTLD_LAZY) };
    if ane_handle.is_null() {
        return Err("Failed to dlopen AppleNeuralEngine.framework".to_string());
    }
    let coreml_path = CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap();
    let coreml_handle = unsafe { libc::dlopen(coreml_path.as_ptr(), libc::RTLD_LAZY) };
    if coreml_handle.is_null() {
        return Err("Failed to dlopen CoreML.framework".to_string());
    }
    Ok(())
}

pub fn get_ane_client() -> Option<Retained<AnyObject>> {
    let cls = class!(_ANEClient);
    unsafe { msg_send![cls, sharedConnection] }
}

pub fn coreml_compile_model(model_url_path: &str) -> Result<String, String> {
    let cls_model = class!(MLModel);
    let cls_url = class!(NSURL);
    let cls_nsstring = class!(NSString);

    let path_str = std::ffi::CString::new(model_url_path).unwrap();
    let ns_path: *mut AnyObject =
        unsafe { msg_send![cls_nsstring, stringWithUTF8String: path_str.as_ptr()] };
    let url: *mut AnyObject = unsafe { msg_send![cls_url, fileURLWithPath: ns_path] };

    let mut error: *mut AnyObject = std::ptr::null_mut();
    let compiled_url: *mut AnyObject =
        unsafe { msg_send![cls_model, compileModelAtURL: url, error: &mut error] };

    if compiled_url.is_null() {
        if !error.is_null() {
            let desc: *mut AnyObject = unsafe { msg_send![error, localizedDescription] };
            if !desc.is_null() {
                let utf8: *const std::ffi::c_char = unsafe { msg_send![desc, UTF8String] };
                if !utf8.is_null() {
                    let s = unsafe { std::ffi::CStr::from_ptr(utf8) }.to_string_lossy();
                    return Err(format!("MLModel compileModelAtURL failed: {}", s));
                }
            }
        }
        return Err("MLModel compileModelAtURL failed with unknown error".to_string());
    }

    let path_ns: *mut AnyObject = unsafe { msg_send![compiled_url, path] };
    let path_utf8: *const std::ffi::c_char = unsafe { msg_send![path_ns, UTF8String] };
    let path = unsafe { std::ffi::CStr::from_ptr(path_utf8) }
        .to_string_lossy()
        .into_owned();

    Ok(path)
}

pub fn compile_model_with_ane_client(
    model_url_path: &str,
    client: &Retained<AnyObject>,
) -> Result<(), String> {
    let cls_url = class!(NSURL);
    let cls_nsstring = class!(NSString);

    let path_str = std::ffi::CString::new(model_url_path).unwrap();
    let ns_path: *mut AnyObject =
        unsafe { msg_send![cls_nsstring, stringWithUTF8String: path_str.as_ptr()] };
    let url: *mut AnyObject = unsafe { msg_send![cls_url, fileURLWithPath: ns_path] };

    let mut error: *mut AnyObject = std::ptr::null_mut();
    let compiled: bool = unsafe {
        msg_send![client, compileModel: url, options: std::ptr::null_mut::<AnyObject>(), qos: 0_u32, error: &mut error]
    };

    if !compiled {
        if !error.is_null() {
            let desc: *mut AnyObject = unsafe { msg_send![error, localizedDescription] };
            if !desc.is_null() {
                let utf8: *const std::ffi::c_char = unsafe { msg_send![desc, UTF8String] };
                if !utf8.is_null() {
                    let s = unsafe { std::ffi::CStr::from_ptr(utf8) }.to_string_lossy();
                    return Err(format!("_ANEClient compileModel failed: {}", s));
                }
            }
        }
        return Err("_ANEClient compileModel failed with unknown error".to_string());
    }

    Ok(())
}

pub fn create_ane_options() -> Retained<AnyObject> {
    let cls_dict = class!(NSDictionary);
    let cls_string = class!(NSString);

    let key: *mut AnyObject = unsafe {
        msg_send![cls_string, stringWithUTF8String: "ForceEspresso\0".as_ptr() as *const i8]
    };
    let val: *mut AnyObject = unsafe { msg_send![class!(NSNumber), numberWithBool: true] };

    let dict: Retained<AnyObject> =
        unsafe { msg_send![cls_dict, dictionaryWithObject: val, forKey: key] };
    dict
}

pub fn compile_and_load_ane_model(
    compiled_url_path: &str,
    client: &Retained<AnyObject>,
) -> Result<Retained<AnyObject>, String> {
    let cls_model = class!(_ANEModel);
    let cls_url = class!(NSURL);
    let cls_nsstring = class!(NSString);

    let path_str = std::ffi::CString::new(compiled_url_path).unwrap();
    let ns_path: *mut AnyObject =
        unsafe { msg_send![cls_nsstring, stringWithUTF8String: path_str.as_ptr()] };
    let url: *mut AnyObject = unsafe { msg_send![cls_url, fileURLWithPath: ns_path] };
    let key_str = std::ffi::CString::new("rvllm_key").unwrap();
    let ns_key: *mut AnyObject =
        unsafe { msg_send![cls_nsstring, stringWithUTF8String: key_str.as_ptr()] };

    let model: *mut AnyObject = unsafe { msg_send![cls_model, alloc] };
    let model: *mut AnyObject = unsafe {
        msg_send![model, initWithModelAtURL: url, key: ns_key, identifierSource: 1_i64, cacheURLIdentifier: std::ptr::null_mut::<AnyObject>(), modelAttributes: std::ptr::null_mut::<AnyObject>(), standardizeURL: true]
    };

    if model.is_null() {
        return Err("_ANEModel initWithModelAtURL returned null".to_string());
    }

    let mut error: *mut AnyObject = std::ptr::null_mut();
    let load_res: bool = unsafe {
        msg_send![client, loadModel: model, options: std::ptr::null_mut::<AnyObject>(), qos: 0_u32, error: &mut error]
    };

    if !load_res {
        if !error.is_null() {
            let desc: *mut AnyObject = unsafe { msg_send![error, localizedDescription] };
            if !desc.is_null() {
                let utf8: *const std::ffi::c_char = unsafe { msg_send![desc, UTF8String] };
                if !utf8.is_null() {
                    let s = unsafe { std::ffi::CStr::from_ptr(utf8) }.to_string_lossy();
                    return Err(format!("_ANEClient loadModel failed: {s}"));
                }
            }
        }
        return Err("_ANEClient loadModel failed with unknown error".to_string());
    }

    Ok(unsafe { Retained::retain(model).unwrap() })
}

pub fn create_ane_iosurface(
    width: usize,
    height: usize,
    pixel_size: usize,
) -> Option<Retained<AnyObject>> {
    let cls = class!(_ANEIOSurfaceObject);
    let obj: *mut AnyObject = unsafe {
        msg_send![cls, createIOSurfaceWithWidth: width, pixel_size: pixel_size, height: height]
    };
    if obj.is_null() {
        None
    } else {
        Some(unsafe { Retained::retain(obj).unwrap() })
    }
}

pub fn get_iosurface_from_object(obj: &Retained<AnyObject>) -> *mut std::ffi::c_void {
    unsafe { msg_send![obj, ioSurface] }
}

pub fn create_ane_request(
    inputs: &Retained<AnyObject>,         // NSArray of _ANEIOSurfaceObject
    input_indices: &Retained<AnyObject>,  // NSArray of NSNumber
    outputs: &Retained<AnyObject>,        // NSArray of _ANEIOSurfaceObject
    output_indices: &Retained<AnyObject>, // NSArray of NSNumber
    procedure_index: u64,
) -> Option<Retained<AnyObject>> {
    let cls = class!(_ANERequest);
    let req: *mut AnyObject = unsafe {
        msg_send![cls, requestWithInputs: Retained::as_ptr(inputs), inputIndices: Retained::as_ptr(input_indices), outputs: Retained::as_ptr(outputs), outputIndices: Retained::as_ptr(output_indices), procedureIndex: procedure_index]
    };
    if req.is_null() {
        None
    } else {
        Some(unsafe { Retained::retain(req).unwrap() })
    }
}

pub fn evaluate_ane_request(
    client: &Retained<AnyObject>,
    model: &Retained<AnyObject>,
    request: &Retained<AnyObject>,
) -> Result<(), String> {
    let mut err_ptr: *mut AnyObject = std::ptr::null_mut();
    let res: bool = unsafe {
        msg_send![client, evaluateWithModel: Retained::as_ptr(model), options: std::ptr::null_mut::<AnyObject>(), request: Retained::as_ptr(request), qos: 0_i64, error: &mut err_ptr]
    };
    if res {
        Ok(())
    } else {
        Err("Evaluation failed".to_string())
    }
}

pub fn create_ns_number_u64(val: u64) -> Retained<AnyObject> {
    let cls = class!(NSNumber);
    unsafe {
        let obj: *mut AnyObject = msg_send![cls, numberWithUnsignedLongLong: val];
        Retained::retain(obj).unwrap()
    }
}

pub fn create_ns_array(objects: &[Retained<AnyObject>]) -> Retained<AnyObject> {
    let cls = class!(NSArray);
    let count = objects.len();
    let ptrs: Vec<*const AnyObject> = objects.iter().map(|o| &**o as *const AnyObject).collect();
    unsafe {
        let obj: *mut AnyObject = msg_send![cls, arrayWithObjects: ptrs.as_ptr(), count: count];
        Retained::retain(obj).unwrap()
    }
}

extern "C" {
    fn IOSurfaceLookup(id: u32) -> *mut std::ffi::c_void;
    fn IOSurfaceLock(buffer: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(buffer: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(buffer: *mut c_void) -> *mut c_void;
}

pub fn get_ane_surface_from_id(id: u32) -> Option<Retained<AnyObject>> {
    let surface = unsafe { IOSurfaceLookup(id) };
    if surface.is_null() {
        return None;
    }
    let cls = class!(_ANEIOSurfaceObject);
    let obj: *mut AnyObject = unsafe { msg_send![cls, objectWithIOSurface: surface] };
    if obj.is_null() {
        None
    } else {
        Some(unsafe { Retained::retain(obj).unwrap() })
    }
}

fn with_iosurface_base<R>(surface: *mut c_void, f: impl FnOnce(*mut u8) -> R) -> Result<R, String> {
    if surface.is_null() {
        return Err("IOSurface pointer is null".to_string());
    }

    let mut seed = 0u32;
    let lock = unsafe { IOSurfaceLock(surface, 0, &mut seed) };
    if lock != 0 {
        return Err(format!("IOSurfaceLock failed: {lock}"));
    }

    let base = unsafe { IOSurfaceGetBaseAddress(surface) };
    let result = if base.is_null() {
        Err("IOSurfaceGetBaseAddress returned null".to_string())
    } else {
        Ok(f(base.cast::<u8>()))
    };

    let unlock = unsafe { IOSurfaceUnlock(surface, 0, &mut seed) };
    if unlock != 0 {
        return Err(format!("IOSurfaceUnlock failed: {unlock}"));
    }

    result
}

pub fn read_iosurface_u32(surface: *mut c_void, offset_bytes: usize) -> Result<u32, String> {
    with_iosurface_base(surface, |base| unsafe {
        std::ptr::read_unaligned(base.add(offset_bytes).cast::<u32>())
    })
}

pub fn read_iosurface_f32(surface: *mut c_void, offset_elements: usize) -> Result<f32, String> {
    with_iosurface_base(surface, |base| unsafe {
        std::ptr::read_unaligned(base.add(offset_elements * 4).cast::<f32>())
    })
}

pub fn write_iosurface_f32(
    surface: *mut c_void,
    offset_elements: usize,
    value: f32,
) -> Result<(), String> {
    with_iosurface_base(surface, |base| unsafe {
        std::ptr::write_unaligned(base.add(offset_elements * 4).cast::<f32>(), value);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_ane() {
        load_frameworks().expect("ANE framework should load on Apple Silicon");
        let client = get_ane_client();
        println!("Got ANEClient: {:?}", client);
        assert!(client.is_some());

        let dump_class = |cls_name: &str| {
            let cstr = std::ffi::CString::new(cls_name).unwrap();
            let cls = objc2::runtime::AnyClass::get(&cstr);
            if let Some(cls) = cls {
                println!("\n--- Instance Methods for {} ---", cls_name);
                let mut count = 0;
                let methods = unsafe {
                    objc2::ffi::class_copyMethodList(cls as *const _ as *mut _, &mut count)
                };
                if !methods.is_null() {
                    for i in 0..count {
                        let m = unsafe { *methods.add(i as usize) };
                        let sel = unsafe { objc2::ffi::method_getName(m) };
                        if let Some(sel) = sel {
                            let sel_name =
                                unsafe { std::ffi::CStr::from_ptr(objc2::ffi::sel_getName(sel)) };
                            println!("Method: {}", sel_name.to_string_lossy());
                        }
                    }
                    unsafe { libc::free(methods as *mut _) };
                }

                println!("\n--- Class Methods for {} ---", cls_name);
                let meta_cls = unsafe { objc2::ffi::object_getClass(cls as *const _ as *mut _) };
                let mut count = 0;
                let methods = unsafe {
                    objc2::ffi::class_copyMethodList(meta_cls as *const _ as *mut _, &mut count)
                };
                if !methods.is_null() {
                    for i in 0..count {
                        let m = unsafe { *methods.add(i as usize) };
                        let sel = unsafe { objc2::ffi::method_getName(m) };
                        if let Some(sel) = sel {
                            let sel_name =
                                unsafe { std::ffi::CStr::from_ptr(objc2::ffi::sel_getName(sel)) };
                            println!("Class Method: {}", sel_name.to_string_lossy());
                        }
                    }
                    unsafe { libc::free(methods as *mut _) };
                }
            } else {
                println!("Class {} not found", cls_name);
            }
        };

        dump_class("_ANEClient");
        dump_class("_ANEModel");
        dump_class("_ANECompiler");
        dump_class("_ANEProgramForEvaluation");
        dump_class("_ANERequest");
        dump_class("_ANEIOSurfaceObject");
    }

    #[test]
    fn test_public_compile() {
        load_frameworks().unwrap();
        let mil_path = "/tmp/rvllm_debug_workspace/model.mlmodel";
        if !std::path::Path::new(mil_path).exists() {
            println!("MIL file not found, skipping test");
            return;
        }
        let compiled_path = coreml_compile_model(mil_path).unwrap();
        println!("Public compiled path: {}", compiled_path);

        // Try loading it with MLModel
        let cls_model = class!(MLModel);
        let cls_url = class!(NSURL);
        let ns_path = std::ffi::CString::new(compiled_path.clone()).unwrap();
        let ns_path_obj: *mut AnyObject =
            unsafe { msg_send![class!(NSString), stringWithUTF8String: ns_path.as_ptr()] };
        let url: *mut AnyObject = unsafe { msg_send![cls_url, fileURLWithPath: ns_path_obj] };

        let mut error: *mut AnyObject = std::ptr::null_mut();
        let model: *mut AnyObject =
            unsafe { msg_send![cls_model, modelWithContentsOfURL: url, error: &mut error] };

        if model.is_null() {
            let desc: *mut AnyObject = unsafe { msg_send![error, localizedDescription] };
            let utf8: *const std::ffi::c_char = unsafe { msg_send![desc, UTF8String] };
            println!(
                "MLModel load failed: {}",
                unsafe { std::ffi::CStr::from_ptr(utf8) }.to_string_lossy()
            );
        } else {
            println!("MLModel load SUCCESS!");
        }
    }
}
