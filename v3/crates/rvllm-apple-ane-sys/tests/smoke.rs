use std::ffi::CString;

use rvllm_apple_ane_sys::PrivateAneRuntime;

#[test]
#[ignore = "requires macOS/aarch64 with --features private-ane and Objective-C runtime access"]
fn objc_runtime_class_and_selector_smoke() -> rvllm_core::Result<()> {
    let runtime = PrivateAneRuntime::open()?;
    let class = runtime.get_class(c"NSObject")?;
    let selector = runtime.register_selector(c"alloc")?;

    assert!(!class.as_ptr().is_null());
    assert!(!selector.as_ptr().is_null());
    Ok(())
}

#[test]
#[ignore = "requires a known private ANE Objective-C class name in RVLLM_PRIVATE_ANE_OBJC_CLASS"]
fn objc_private_ane_class_probe_smoke() -> rvllm_core::Result<()> {
    let class_name = match std::env::var("RVLLM_PRIVATE_ANE_OBJC_CLASS") {
        Ok(value) => value,
        Err(_) => {
            panic!("set RVLLM_PRIVATE_ANE_OBJC_CLASS to the private ANE runtime class to probe")
        }
    };
    let class_name = match CString::new(class_name) {
        Ok(value) => value,
        Err(_) => panic!("RVLLM_PRIVATE_ANE_OBJC_CLASS must not contain interior NUL bytes"),
    };

    let runtime = PrivateAneRuntime::open()?;
    let class = runtime.get_class(class_name.as_c_str())?;

    assert!(!class.as_ptr().is_null());
    Ok(())
}
