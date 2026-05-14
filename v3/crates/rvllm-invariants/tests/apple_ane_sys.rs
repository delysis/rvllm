use std::fs;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn apple_private_ane_sys_boundary_is_explicit() {
    let crates = crates_dir();
    let sys_root = crates.join("rvllm-apple-ane-sys");
    let sys_manifest = sys_root.join("Cargo.toml");
    assert!(
        sys_manifest.exists(),
        "rvllm-apple-ane-sys crate must own private ANE Objective-C FFI"
    );

    let sys_lib = fs::read_to_string(sys_root.join("src/lib.rs")).expect("read sys lib.rs");
    assert!(
        sys_lib.contains("target_os = \"macos\"")
            && sys_lib.contains("target_arch = \"aarch64\"")
            && sys_lib.contains("feature = \"private-ane\""),
        "Objective-C runtime FFI must be gated on macOS + aarch64 + private-ane"
    );
    assert!(
        sys_lib.contains("unsafe_op_in_unsafe_fn"),
        "sys crate must make unsafe Objective-C calls explicit"
    );
    let objc_ffi = fs::read_to_string(sys_root.join("src/objc.rs")).expect("read objc ffi");
    assert!(
        objc_ffi.contains("extern \"C\"")
            && objc_ffi.contains("objc_getClass")
            && objc_ffi.contains("sel_registerName")
            && objc_ffi.contains("objc_msgSend"),
        "sys crate must scaffold the Objective-C runtime FFI boundary"
    );

    let apple_manifest =
        fs::read_to_string(crates.join("rvllm-apple/Cargo.toml")).expect("read apple Cargo.toml");
    let safe_crate_depends_on_sys = apple_manifest.lines().any(|line| {
        let line = line.trim();
        !line.starts_with('#')
            && (line.starts_with("rvllm-apple-ane-sys")
                || line == "[dependencies.rvllm-apple-ane-sys]"
                || line == "[dev-dependencies.rvllm-apple-ane-sys]"
                || line == "[build-dependencies.rvllm-apple-ane-sys]")
    });
    assert!(
        !safe_crate_depends_on_sys,
        "safe rvllm-apple must not depend directly on the unsafe sys crate yet"
    );

    let apple_lib =
        fs::read_to_string(crates.join("rvllm-apple/src/lib.rs")).expect("read apple lib.rs");
    assert!(
        apple_lib.contains("#![forbid(unsafe_code)]"),
        "safe rvllm-apple must continue forbidding unsafe code"
    );
}

#[test]
fn apple_private_ane_sys_has_ignored_smoke_tests() {
    let smoke = crates_dir()
        .join("rvllm-apple-ane-sys")
        .join("tests/smoke.rs");
    assert!(
        smoke.exists(),
        "rvllm-apple-ane-sys must include ignored hardware/runtime smoke tests"
    );
    let body = fs::read_to_string(smoke).expect("read smoke tests");
    assert!(
        body.contains("#[ignore"),
        "private ANE smoke tests must be ignored by default"
    );
    assert!(
        body.contains("objc"),
        "smoke tests should exercise the Objective-C runtime boundary"
    );
}
