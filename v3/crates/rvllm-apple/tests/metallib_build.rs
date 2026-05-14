#[allow(dead_code)]
#[path = "../build_support/metallib.rs"]
mod metallib;

use std::fs;
use std::path::{Path, PathBuf};

use metallib::{
    discover_metal_sources, plan_metallib_build, CiBehavior, MetallibBuildEnv, MetallibBuildPlan,
    MetallibBuildPlanError, MetallibSkipReason,
};
use rvllm_apple::DirectMetalPipelineName;

fn p(path: &str) -> PathBuf {
    Path::new(path).to_path_buf()
}

#[test]
fn discovers_first_party_prefill_source_and_plans_air_output() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metal_root = manifest_dir.join("metal");
    let prefill_source = metal_root.join("prefill.metal");
    let out_dir = manifest_dir.join("target/metallib-build-test/out");

    let sources = match discover_metal_sources(&metal_root) {
        Ok(sources) => sources,
        Err(err) => panic!("metal source discovery failed: {err}"),
    };
    assert!(
        sources.contains(&prefill_source),
        "expected discovered Metal sources to include {prefill_source:?}; got {sources:?}"
    );

    let source_text = match fs::read_to_string(&prefill_source) {
        Ok(source_text) => source_text,
        Err(err) => panic!("failed to read {prefill_source:?}: {err}"),
    };
    for pipeline in [
        DirectMetalPipelineName::RmsNorm,
        DirectMetalPipelineName::Matmul,
        DirectMetalPipelineName::Rope,
        DirectMetalPipelineName::Attention,
    ] {
        let needle = format!("kernel void {}(", pipeline.symbol());
        assert!(
            source_text.contains(&needle),
            "expected {prefill_source:?} to define {needle}"
        );
    }

    let plan = match plan_metallib_build(MetallibBuildEnv {
        manifest_dir: &manifest_dir,
        out_dir: &out_dir,
        target_os: "macos",
        ci: false,
        ci_behavior: CiBehavior::Build,
        metal_sources: sources,
    }) {
        Ok(plan) => plan,
        Err(err) => panic!("valid first-party Metal build plan failed: {err}"),
    };
    let MetallibBuildPlan::Compile(plan) = plan else {
        panic!("expected compile plan");
    };
    let source_index = match plan
        .metal_sources
        .iter()
        .position(|source| source == &prefill_source)
    {
        Some(source_index) => source_index,
        None => panic!("compile plan omitted {prefill_source:?}: {plan:?}"),
    };

    assert_eq!(
        plan.air_outputs[source_index],
        out_dir.join("metal-air").join("prefill.air")
    );
}

#[test]
fn resolves_metal_sources_under_manifest_and_outputs_under_out_dir() {
    let env = MetallibBuildEnv {
        manifest_dir: Path::new("/repo/v3/crates/rvllm-apple"),
        out_dir: Path::new("/repo/v3/target/debug/build/rvllm-apple/out"),
        target_os: "macos",
        ci: false,
        ci_behavior: CiBehavior::SkipUnlessOptedIn,
        metal_sources: vec![
            p("/repo/v3/crates/rvllm-apple/metal/prefill.metal"),
            p("/repo/v3/crates/rvllm-apple/metal/kernels/decode.metal"),
        ],
    };

    let plan = match plan_metallib_build(env) {
        Ok(plan) => plan,
        Err(err) => panic!("valid build plan failed: {err}"),
    };
    let MetallibBuildPlan::Compile(plan) = plan else {
        panic!("expected compile plan");
    };

    assert_eq!(
        plan.metallib_path,
        p("/repo/v3/target/debug/build/rvllm-apple/out/rvllm_apple.metallib")
    );
    assert_eq!(
        plan.air_outputs,
        vec![
            p("/repo/v3/target/debug/build/rvllm-apple/out/metal-air/prefill.air"),
            p("/repo/v3/target/debug/build/rvllm-apple/out/metal-air/kernels/decode.air"),
        ]
    );
}

#[test]
fn macos_ci_skips_metallib_compilation_unless_explicitly_enabled() {
    let env = MetallibBuildEnv {
        manifest_dir: Path::new("/repo/v3/crates/rvllm-apple"),
        out_dir: Path::new("/repo/v3/target/debug/build/rvllm-apple/out"),
        target_os: "macos",
        ci: true,
        ci_behavior: CiBehavior::SkipUnlessOptedIn,
        metal_sources: vec![p("/repo/v3/crates/rvllm-apple/metal/prefill.metal")],
    };

    assert_eq!(
        match plan_metallib_build(env) {
            Ok(plan) => plan,
            Err(err) => panic!("ci skip plan failed: {err}"),
        },
        MetallibBuildPlan::Skip(MetallibSkipReason::CiDefault)
    );
}

#[test]
fn non_macos_targets_skip_without_resolving_macos_toolchain_paths() {
    let env = MetallibBuildEnv {
        manifest_dir: Path::new("/repo/v3/crates/rvllm-apple"),
        out_dir: Path::new("/repo/v3/target/debug/build/rvllm-apple/out"),
        target_os: "linux",
        ci: false,
        ci_behavior: CiBehavior::Build,
        metal_sources: vec![p("/repo/v3/crates/rvllm-apple/metal/prefill.metal")],
    };

    assert_eq!(
        match plan_metallib_build(env) {
            Ok(plan) => plan,
            Err(err) => panic!("linux skip plan failed: {err}"),
        },
        MetallibBuildPlan::Skip(MetallibSkipReason::NonMacosTarget)
    );
}

#[test]
fn rejects_metal_sources_outside_the_crate_metal_tree() {
    let env = MetallibBuildEnv {
        manifest_dir: Path::new("/repo/v3/crates/rvllm-apple"),
        out_dir: Path::new("/repo/v3/target/debug/build/rvllm-apple/out"),
        target_os: "macos",
        ci: false,
        ci_behavior: CiBehavior::Build,
        metal_sources: vec![p("/repo/v3/other/prefill.metal")],
    };

    assert_eq!(
        plan_metallib_build(env),
        Err(MetallibBuildPlanError::SourceOutsideMetalRoot {
            source: p("/repo/v3/other/prefill.metal"),
            metal_root: p("/repo/v3/crates/rvllm-apple/metal"),
        })
    );
}

#[test]
fn rejects_parent_dir_escape_after_metal_root_prefix() {
    let env = MetallibBuildEnv {
        manifest_dir: Path::new("/repo/v3/crates/rvllm-apple"),
        out_dir: Path::new("/repo/v3/target/debug/build/rvllm-apple/out"),
        target_os: "macos",
        ci: false,
        ci_behavior: CiBehavior::Build,
        metal_sources: vec![p("/repo/v3/crates/rvllm-apple/metal/../escape.metal")],
    };

    assert_eq!(
        plan_metallib_build(env),
        Err(MetallibBuildPlanError::SourceOutsideMetalRoot {
            source: p("/repo/v3/crates/rvllm-apple/metal/../escape.metal"),
            metal_root: p("/repo/v3/crates/rvllm-apple/metal"),
        })
    );
}
