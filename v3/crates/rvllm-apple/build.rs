#[path = "build_support/metallib.rs"]
mod metallib;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use metallib::{
    discover_metal_sources, plan_metallib_build, CiBehavior, MetallibBuildEnv, MetallibBuildPlan,
    MetallibCompilePlan, MetallibSkipReason,
};

fn main() {
    println!("cargo:rerun-if-env-changed=CI");
    println!("cargo:rerun-if-env-changed=RVLLM_APPLE_BUILD_METALLIB_IN_CI");
    println!("cargo:rustc-check-cfg=cfg(rvllm_apple_metallib_built)");
    println!("cargo:rerun-if-changed=build_support/metallib.rs");
    println!("cargo:rerun-if-changed=metal");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        panic!("CARGO_MANIFEST_DIR is required for rvllm-apple metallib build")
    }));
    let out_dir = PathBuf::from(
        env::var("OUT_DIR").unwrap_or_else(|_| panic!("OUT_DIR is required for metallib build")),
    );
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .unwrap_or_else(|_| panic!("CARGO_CFG_TARGET_OS is required for metallib build"));
    let ci = env_flag("CI");
    let ci_behavior = if env::var("RVLLM_APPLE_BUILD_METALLIB_IN_CI").ok().as_deref() == Some("1") {
        CiBehavior::Build
    } else {
        CiBehavior::SkipUnlessOptedIn
    };

    let should_discover_sources =
        target_os == "macos" && !(ci && ci_behavior == CiBehavior::SkipUnlessOptedIn);
    let metal_sources = if should_discover_sources {
        discover_metal_sources(&manifest_dir.join("metal"))
            .unwrap_or_else(|e| panic!("failed to discover Metal sources: {e}"))
    } else {
        Vec::new()
    };
    for source in &metal_sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    let plan = plan_metallib_build(MetallibBuildEnv {
        manifest_dir: &manifest_dir,
        out_dir: &out_dir,
        target_os: &target_os,
        ci,
        ci_behavior,
        metal_sources,
    })
    .unwrap_or_else(|e| panic!("invalid rvllm-apple metallib build plan: {e}"));

    match plan {
        MetallibBuildPlan::Compile(plan) => compile_metallib(plan),
        MetallibBuildPlan::Skip(MetallibSkipReason::NonMacosTarget) => {}
        MetallibBuildPlan::Skip(reason) => {
            println!("cargo:warning=rvllm-apple metallib build skipped: {reason:?}");
        }
    }
}

fn compile_metallib(plan: MetallibCompilePlan) {
    for air in &plan.air_outputs {
        let parent = air
            .parent()
            .unwrap_or_else(|| panic!("AIR output has no parent directory: {}", air.display()));
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create AIR output directory {parent:?}: {e}"));
    }

    for (source, air) in plan.metal_sources.iter().zip(plan.air_outputs.iter()) {
        let args = vec![
            "-sdk".to_owned(),
            "macosx".to_owned(),
            "metal".to_owned(),
            "-c".to_owned(),
            source.display().to_string(),
            "-o".to_owned(),
            air.display().to_string(),
        ];
        run_xcrun("metal", &args);
    }

    let mut args = vec![
        "-sdk".to_string(),
        "macosx".to_string(),
        "metallib".to_string(),
    ];
    args.extend(plan.air_outputs.iter().map(|air| air.display().to_string()));
    args.push("-o".to_string());
    args.push(plan.metallib_path.display().to_string());

    run_xcrun("metallib", &args);

    println!(
        "cargo:rustc-env=RVLLM_APPLE_METALLIB_PATH={}",
        plan.metallib_path.display()
    );
    println!("cargo:rustc-cfg=rvllm_apple_metallib_built");
}

fn run_xcrun(step: &'static str, args: &[String]) {
    let status = Command::new("xcrun")
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch xcrun for {step}: {e}"));
    if !status.success() {
        panic!("xcrun {step} failed with status {status}");
    }
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}
