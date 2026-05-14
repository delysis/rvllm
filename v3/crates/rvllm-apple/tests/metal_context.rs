use std::path::PathBuf;

use rvllm_apple::{
    DirectMetalContextConfig, DirectMetalPipelineName, MetalPrefillBackend, MetalPrefillConfig,
};
use rvllm_core::{AppleError, ConfigError, RvllmError};

fn direct_prefill_config(path: Option<PathBuf>) -> MetalPrefillConfig {
    MetalPrefillConfig {
        backend: MetalPrefillBackend::DirectMetal,
        metallib_path: path,
        max_prompt_tokens: 4096,
        max_batch: 8,
    }
}

#[test]
fn direct_metal_context_config_preserves_no_fallback_contract() {
    let config = match DirectMetalContextConfig::new(direct_prefill_config(Some(PathBuf::from(
        "rvllm_apple.metallib",
    )))) {
        Ok(v) => v,
        Err(e) => panic!("unexpected config error: {e}"),
    };

    assert_eq!(config.prefill().backend, MetalPrefillBackend::DirectMetal);
    assert_eq!(
        config.pipeline_names(),
        &[
            DirectMetalPipelineName::RmsNorm,
            DirectMetalPipelineName::Matmul,
            DirectMetalPipelineName::Rope,
            DirectMetalPipelineName::Attention,
        ]
    );
    assert_eq!(config.contract().command_buffers_per_layer_group, 1);
    assert!(!config.contract().allocates_in_hot_path);
    assert!(config.contract().owns_kv_cache_write);
    assert!(config.contract().uses_persistent_parameter_buffers);
}

#[test]
fn direct_metal_context_config_requires_direct_backend_and_explicit_metallib() {
    let mlx = MetalPrefillConfig {
        backend: MetalPrefillBackend::MlxPrototype,
        metallib_path: Some(PathBuf::from("rvllm_apple.metallib")),
        max_prompt_tokens: 4096,
        max_batch: 8,
    };
    match DirectMetalContextConfig::new(mlx) {
        Err(RvllmError::Apple {
            err: AppleError::FeatureNotAvailable { backend, op },
            ..
        }) => {
            assert_eq!(backend, "mlx-prototype");
            assert_eq!(op, "direct_metal_context");
        }
        other => panic!("expected typed Apple FeatureNotAvailable error, got {other:?}"),
    }

    match DirectMetalContextConfig::new(direct_prefill_config(None)) {
        Err(RvllmError::Config {
            err: ConfigError::MissingField { name },
            field,
        }) => {
            assert_eq!(name, "metallib_path");
            assert_eq!(field, "metallib_path");
        }
        other => panic!("expected typed Config MissingField error, got {other:?}"),
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn direct_metal_context_rejects_missing_metallib_without_fallback() {
    use rvllm_apple::DirectMetalContext;

    let missing = PathBuf::from("does-not-exist/rvllm_apple.metallib");
    let config = match DirectMetalContextConfig::new(direct_prefill_config(Some(missing.clone()))) {
        Ok(v) => v,
        Err(e) => panic!("unexpected config error: {e}"),
    };

    match DirectMetalContext::new(config) {
        Err(RvllmError::Apple {
            err: AppleError::MetallibMissing { path },
            ..
        }) => assert_eq!(path, missing),
        Ok(_) => panic!("expected missing metallib error"),
        Err(e) => panic!("expected typed MetallibMissing error, got {e}"),
    }
}

#[test]
#[ignore = "requires macOS, --features metal, and RVLLM_APPLE_METALLIB pointing at a compiled metallib"]
fn direct_metal_context_creates_system_device_and_queue() {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use rvllm_apple::DirectMetalContext;

        let metallib = match std::env::var_os("RVLLM_APPLE_METALLIB") {
            Some(v) => PathBuf::from(v),
            None => panic!("RVLLM_APPLE_METALLIB must be set explicitly"),
        };
        let config = match DirectMetalContextConfig::new(direct_prefill_config(Some(metallib))) {
            Ok(v) => v,
            Err(e) => panic!("unexpected config error: {e}"),
        };
        let context = match DirectMetalContext::new(config) {
            Ok(v) => v,
            Err(e) => panic!("unexpected Metal context error: {e}"),
        };

        assert!(!context.device_name().is_empty());
        assert_eq!(
            context.loaded_pipeline_count(),
            context.config().pipeline_names().len()
        );
    }
}
