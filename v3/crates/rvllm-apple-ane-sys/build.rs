fn main() {
    println!("cargo::rustc-check-cfg=cfg(apple_silicon)");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os == "macos" && target_arch == "aarch64" {
        println!("cargo:rustc-cfg=apple_silicon");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    }
}
