use std::path::PathBuf;

use rvllm_apple::{
    AppleAcceleratorTarget, AppleBackendMode, AppleGpuFamily, AppleNpuGeneration, DeviceTier,
    RolloutBucket,
};
use rvllm_deploy::apple_offgrid::{
    AppleOffgridBundleManifest, AppleOffgridMilCache, AppleOffgridMilProgram,
    AppleOffgridModelFormat, AppleOffgridModelRef, AppleOffgridPolicyMode,
    AppleOffgridRuntimePolicy, BundleArtifact,
};

fn artifact(path: &str) -> BundleArtifact {
    BundleArtifact {
        path: PathBuf::from(path),
        sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        bytes: 4096,
    }
}

fn manifest() -> AppleOffgridBundleManifest {
    AppleOffgridBundleManifest {
        schema_version: 1,
        bundle_id: "rvllm-apple-offgrid-gemma4-e2b-m4max".into(),
        revision: "0123456789abcdef0123456789abcdef01234567".into(),
        model: AppleOffgridModelRef {
            format: AppleOffgridModelFormat::Safetensors,
            config: artifact("model/config.json"),
            tokenizer: artifact("model/tokenizer.json"),
            weights: vec![artifact("model/model-00001-of-00002.safetensors")],
        },
        metallib: artifact("metal/rvllm_apple.metallib"),
        mil_cache: AppleOffgridMilCache {
            cache_key: "gemma4-e2b:m4max:rollout-v1".into(),
            root: PathBuf::from("mil-cache"),
            programs: vec![AppleOffgridMilProgram {
                name: "layer0_ffn_b8t4".into(),
                bucket: RolloutBucket { seqs: 8, tokens: 4 },
                mil: artifact("mil-cache/layer0_ffn_b8t4.mil"),
                compiled: artifact("mil-cache/layer0_ffn_b8t4.milc"),
            }],
        },
        hardware_profile: AppleAcceleratorTarget {
            device_name: "Apple M4 Max".into(),
            gpu_family: AppleGpuFamily::Apple9,
            tier: DeviceTier::Max,
            npu_generation: AppleNpuGeneration::M4,
            architecture_gen: 16,
            has_nax: false,
            ane_cores: 16,
            die_count: 1,
        },
        backend_mode: AppleBackendMode::MetalPrefillAneFfnRollout,
        energy_policy: AppleOffgridRuntimePolicy {
            mode: AppleOffgridPolicyMode::Sustained,
            max_package_watts: 42,
            max_gpu_watts: 28,
            max_ane_watts: 10,
            low_power_mode: true,
            thermal_pressure_limit: "NominalOrFair".into(),
        },
    }
}

#[test]
fn apple_offgrid_manifest_roundtrips_required_schema() {
    let manifest = manifest();

    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let decoded: AppleOffgridBundleManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.schema_version, 1);
    assert_eq!(decoded.model.weights.len(), 1);
    assert_eq!(
        decoded.metallib.path,
        PathBuf::from("metal/rvllm_apple.metallib")
    );
    assert_eq!(
        decoded.mil_cache.programs[0].bucket,
        RolloutBucket { seqs: 8, tokens: 4 }
    );
    assert_eq!(decoded.hardware_profile.device_name, "Apple M4 Max");
    assert_eq!(
        decoded.backend_mode,
        AppleBackendMode::MetalPrefillAneFfnRollout
    );
    assert!(decoded.energy_policy.low_power_mode);
    assert!(decoded.validate_schema().is_ok());
}

#[test]
fn apple_offgrid_manifest_rejects_missing_artifact_paths() {
    let mut manifest = manifest();
    manifest.metallib.path = PathBuf::new();

    let err = manifest.validate_schema().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("metallib.path"));
}

#[test]
fn apple_offgrid_manifest_rejects_unknown_fields() {
    let mut json = serde_json::to_value(manifest()).unwrap();
    json.as_object_mut()
        .unwrap()
        .insert("fallback_model_dir".into(), serde_json::json!("model-alt"));

    let err = serde_json::from_value::<AppleOffgridBundleManifest>(json).unwrap_err();
    assert!(err.to_string().contains("unknown field"));
}
