use super::*;
#[cfg(target_os = "macos")]
use rvllm_apple_metal::weight_loader::scan_safetensor_tensors;
use serde_json::{Map, Value};
use std::fs::{self, File};
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FULL_NONZERO_ZERO_DIM: usize = 0;
const FULL_NONZERO_ORIGINAL_DIM: usize = 7;
const FULL_NONZERO_ATTENTION_DIM: usize = 9;
const FULL_NONZERO_VALUE_DIM: usize = 11;
const FULL_NONZERO_FFN_DIM: usize = 13;
const GQA_SOURCE_DIM: usize = 7;
const GQA_OUTPUT_DIM: usize = 9;
const GQA_VALUE_DIM: usize = 11;
const GQA_OUTPUT_HEAD: usize = 1;

#[cfg(all(feature = "apple", target_os = "macos"))]
static METAL_DEBUG_SYNC_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Clone)]
struct SharedModelMetalBackend {
    inner: std::rc::Rc<std::cell::RefCell<ModelMetalBackend>>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl SharedModelMetalBackend {
    fn new(model_dir: std::path::PathBuf) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(ModelMetalBackend::new(model_dir))),
        }
    }

    fn debug_read_decode_logits_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
        self.inner.borrow().debug_read_decode_logits_f32(num_tokens)
    }

    fn debug_read_residual_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
        self.inner.borrow().debug_read_residual_f32(num_tokens)
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl AppleBackend for SharedModelMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        self.inner.borrow_mut().prepare(plan)
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.inner.borrow_mut().launch_prefill(handoff)
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<rvllm_apple::RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.inner.borrow_mut().launch_rollout(handoff, bucket)
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        self.inner.borrow_mut().collect(ticket)
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
struct MetalDebugSyncEnvGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<std::ffi::OsString>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl MetalDebugSyncEnvGuard {
    fn new() -> Self {
        Self {
            _guard: METAL_DEBUG_SYNC_ENV_LOCK.lock().expect("lock env guard"),
            previous: std::env::var_os(RVLLM_METAL_DEBUG_SYNC_ENV),
        }
    }

    fn set_current(&self, value: Option<&str>) {
        if let Some(value) = value {
            std::env::set_var(RVLLM_METAL_DEBUG_SYNC_ENV, value);
        } else {
            std::env::remove_var(RVLLM_METAL_DEBUG_SYNC_ENV);
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl Drop for MetalDebugSyncEnvGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            std::env::set_var(RVLLM_METAL_DEBUG_SYNC_ENV, previous);
        } else {
            std::env::remove_var(RVLLM_METAL_DEBUG_SYNC_ENV);
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
struct MetalDebugEnvGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl MetalDebugEnvGuard {
    fn new(names: &[&'static str]) -> Self {
        Self {
            _guard: METAL_DEBUG_SYNC_ENV_LOCK.lock().expect("lock env guard"),
            previous: names
                .iter()
                .map(|&name| (name, std::env::var_os(name)))
                .collect(),
        }
    }

    fn set(&self, name: &'static str, value: impl AsRef<std::ffi::OsStr>) {
        std::env::set_var(name, value);
    }

    fn remove(&self, name: &'static str) {
        std::env::remove_var(name);
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl Drop for MetalDebugEnvGuard {
    fn drop(&mut self) {
        for (name, previous) in &self.previous {
            if let Some(previous) = previous {
                std::env::set_var(name, previous);
            } else {
                std::env::remove_var(name);
            }
        }
    }
}

fn temp_fixture_dir() -> std::path::PathBuf {
    static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    let serial = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rvllm-metal-zero-layer-test-{}-{}-{}",
        std::process::id(),
        now,
        serial
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn paged_contract_handoff(fingerprint: [u8; 32]) -> HandoffCapsule {
    HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(9)],
        vec![TokenId(7)],
        vec![0, 1],
        vec![0],
        vec![1],
    )
    .with_paged_kv(
        vec![rvllm_apple::HandoffKvChain {
            id: 2,
            generation: 4,
        }],
        1,
        vec![0],
        vec![0],
        fingerprint,
    )
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn ane_prefill_page_plan_binds_complete_prompt_to_logical_pages() {
    let full = |pages: Vec<u32>, start: u32, tokens: u32| {
        let slots = (start..start + tokens)
            .map(|position| (pages[position as usize / 32] * 32 + position % 32) as i32)
            .collect();
        HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(9)],
            vec![TokenId(7); tokens as usize],
            vec![0, tokens],
            vec![start + tokens - 1],
            vec![start + tokens],
        )
        .with_paged_kv(
            vec![rvllm_apple::HandoffKvChain {
                id: 2,
                generation: 4,
            }],
            pages.len() as u32,
            pages,
            slots,
            [0xA5; 32],
        )
    };
    let valid = full(vec![2, 0], 0, 33);
    assert_eq!(
        ane_full_prompt_pages(&valid).unwrap(),
        vec![BlockId(2), BlockId(0)]
    );
    assert!(ane_full_prompt_pages(&full(vec![2, 2], 0, 33)).is_err());
    assert!(ane_full_prompt_pages(&full(vec![2, 0], 32, 1)).is_err());

    let replace_slots = |slots| {
        valid.clone().with_paged_kv(
            valid.kv_chains.clone(),
            valid.max_blocks_per_seq,
            valid.block_tables.clone(),
            slots,
            valid.model_layout_fingerprint,
        )
    };
    let mut slots = valid.slot_mapping.clone();
    slots[32] = 32;
    let mismatched = replace_slots(slots);
    assert!(mismatched.validate().is_ok());
    assert!(ane_full_prompt_pages(&mismatched).is_err());
    assert!(ane_full_prompt_pages(&replace_slots(Vec::new())).is_err());
    let mut mismatched = valid;
    mismatched.positions[0] = u32::MAX;
    assert!(ane_full_prompt_pages(&mismatched).is_err());

    let legacy = HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(9)],
        vec![TokenId(7)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    assert!(ane_full_prompt_pages(&legacy).is_err());
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn paged_kv_contract_accepts_only_the_prepared_nonzero_fingerprint() {
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    backend.prepared_model_layout_fingerprint = Some([0xA5; 32]);
    let matching = paged_contract_handoff([0xA5; 32]);
    assert!(matching.is_well_formed());
    backend
        .validate_paged_kv_contract(&matching, "test")
        .expect("matching prepared layout must be accepted");

    let mismatch = paged_contract_handoff([0x5A; 32]);
    let error = backend
        .validate_paged_kv_contract(&mismatch, "test")
        .expect_err("stale layout fingerprint must fail closed");
    assert!(format!("{error}").contains("fingerprint mismatch"));

    backend.prepared_model_layout_fingerprint = Some([0; 32]);
    let error = backend
        .validate_paged_kv_contract(&paged_contract_handoff([0; 32]), "test")
        .expect_err("placeholder fingerprints must not authorize paged KV");
    assert!(format!("{error}").contains("prepared model layout fingerprint must be nonzero"));
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn paged_kv_contract_requires_a_prepared_fingerprint_but_legacy_handoffs_do_not() {
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    let paged = paged_contract_handoff([0x11; 32]);
    let error = backend
        .validate_paged_kv_contract(&paged, "test")
        .expect_err("unprepared layout must fail closed");
    assert!(format!("{error}").contains("no prepared model layout fingerprint"));

    let legacy = HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(9)],
        vec![TokenId(7)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    backend
        .validate_paged_kv_contract(&legacy, "test")
        .expect("legacy ordinal handoff must remain compatible");

    // Current direct CLI/server plans use the zero placeholder, but they do
    // not emit paged metadata. Keep that shipping legacy path compatible.
    backend.prepared_model_layout_fingerprint = Some([0; 32]);
    backend
        .validate_paged_kv_contract(&legacy, "test")
        .expect("a prepared zero-hash plan remains valid for ordinal KV only");
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn paged_kv_contract_cannot_be_bypassed_by_omitting_chain_handles() {
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    backend.prepared_model_layout_fingerprint = Some([0x44; 32]);
    let handoff = HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(9)],
        vec![TokenId(7)],
        vec![0, 1],
        vec![0],
        vec![1],
    )
    .with_paged_kv(Vec::new(), 1, vec![0], vec![0], [0x55; 32]);
    assert!(handoff.is_well_formed());

    let error = backend
        .validate_paged_kv_contract(&handoff, "test")
        .expect_err("physical page metadata must always authenticate its layout");
    assert!(format!("{error}").contains("fingerprint mismatch"));
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn paged_bridge_fingerprint_matches_the_prepared_runtime_plan_contract() {
    let fingerprint = [0x6D; 32];
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    backend.prepared_model_layout_fingerprint = Some(fingerprint);

    let mut pool =
        crate::paged_kv::PagedKvPool::new(crate::paged_kv::PagedKvConfig::apple_v1(4, 2))
            .expect("create Apple-v1 pool");
    let chain = pool
        .allocate_chain(rvllm_core::ReqId(41), 33)
        .expect("allocate request-owned pages");
    let plan = crate::BatchPlan::Prefill {
        req_ids: vec![rvllm_core::ReqId(41)],
        prompt_tokens_flat: vec![TokenId(7)],
        cu_seqlens_q: vec![0, 1],
        query_start_positions: vec![32],
        context_lens: vec![33],
        kv_chains: vec![Some(chain)],
    };
    let handoff = crate::apple_bridge::handoff_from_prefill_plan_with_paged_kv(
        &plan,
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        None,
        &pool,
        fingerprint,
    )
    .expect("build the production paged bridge capsule");

    assert!(handoff.is_well_formed());
    assert_ne!(handoff.model_layout_fingerprint, [0; 32]);
    backend
        .validate_paged_kv_contract(&handoff, "test")
        .expect("the bridge and prepared plan must share the exact fingerprint");
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn paged_kv_fingerprint_mismatch_is_rejected_before_ticket_or_encoding() {
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    backend.prepared = true;
    backend.prepared_model_layout_fingerprint = Some([0x22; 32]);
    let error = backend
        .launch_prefill(&paged_contract_handoff([0x33; 32]))
        .expect_err("mismatch must be rejected before model state is touched");
    assert!(format!("{error}").contains("fingerprint mismatch"));
    assert_eq!(backend.next_step_id, 0);
    assert_eq!(backend.in_flight.len(), 0);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn configured_metal_float_type_auto_uses_checkpoint_dtype() {
    let guard = MetalDebugEnvGuard::new(&[RVLLM_METAL_DTYPE_ENV]);
    guard.remove(RVLLM_METAL_DTYPE_ENV);
    let f16_dir = write_dtype_probe_fixture("F16", &f16_bytes(&[1.0]));
    let bf16_dir = write_dtype_probe_fixture("BF16", &bf16_bytes(&[1.0]));

    assert_eq!(
        configured_metal_float_type(&f16_dir).expect("default f16 dtype"),
        MetalFloatType::F16
    );
    assert_eq!(
        configured_metal_float_type(&bf16_dir).expect("default bf16 dtype"),
        MetalFloatType::Bf16
    );

    guard.set(RVLLM_METAL_DTYPE_ENV, "auto");
    assert_eq!(
        configured_metal_float_type(&bf16_dir).expect("auto bf16 dtype"),
        MetalFloatType::Bf16
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn configured_metal_float_type_keeps_explicit_overrides() {
    let guard = MetalDebugEnvGuard::new(&[RVLLM_METAL_DTYPE_ENV]);
    let missing_dir = temp_fixture_dir().join("does-not-need-to-exist");

    guard.set(RVLLM_METAL_DTYPE_ENV, "bfloat16");
    assert_eq!(
        configured_metal_float_type(&missing_dir).expect("explicit bf16 dtype"),
        MetalFloatType::Bf16
    );

    guard.set(RVLLM_METAL_DTYPE_ENV, "float16");
    assert_eq!(
        configured_metal_float_type(&missing_dir).expect("explicit f16 dtype"),
        MetalFloatType::F16
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn explicit_metal_identity_does_not_follow_process_environment() {
    use rvllm_apple_metal::{MetalKernelOptions, MetalModelLimits};
    let names = [
        "RVLLM_METAL_PREFILL_GEMM",
        "RVLLM_METAL_PREFILL_ATTENTION",
        "RVLLM_METAL_BF16_ACCUM",
    ];
    let guard = MetalDebugEnvGuard::new(&names);
    let options = ModelMetalOptions {
        float_type: MetalFloatType::Bf16,
        kernels: MetalKernelOptions {
            prefill_mma32: true,
            prefill_simd_attention: true,
            ..MetalKernelOptions::default()
        },
        limits: MetalModelLimits {
            max_context_tokens: 1024,
            max_batch_tokens: 1024,
            max_batch_sequences: 1,
        },
    };
    let backend =
        ModelMetalBackend::with_options("unused-model".into(), "unused.metallib".into(), options);
    let fingerprint = |kernels| {
        metal_numeric_abi_fingerprint_impl(
            MetalFloatType::Bf16,
            false,
            false,
            None,
            None,
            MetalLowBitResidencyPolicy::HybridFallback,
            kernels,
        )
    };
    let before = fingerprint(backend.kernel_options);
    guard.set(names[0], "off");
    guard.set(names[1], "off");
    guard.set(names[2], "quantized");
    assert_eq!(before, fingerprint(backend.kernel_options));
    let source = kernels::kernel_source_with_options(options.float_type, backend.kernel_options);
    assert!(!source.contains("acc = bf16_acc"));
    let scalar_options = MetalKernelOptions {
        prefill_simd_attention: false,
        ..options.kernels
    };
    assert_ne!(before, fingerprint(scalar_options));
    assert!(!backend.debug_sync && !backend.experimental_kv_int8);
    let mut invalid = ModelMetalBackend::with_options(
        "unused-model".into(),
        "unused.metallib".into(),
        ModelMetalOptions {
            kernels: MetalKernelOptions {
                quantized_bf16_accumulation: true,
                ..options.kernels
            },
            ..options
        },
    );
    let err = invalid
        .initialize_model_resources()
        .err()
        .expect("reject before opening model/library or creating a device");
    assert!(err
        .to_string()
        .contains("explicit native Metal libraries require FP32 accumulation"));
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn metal_numeric_abi_fingerprint_separates_dtype_and_kv_format() {
    const BF16_ACCUM_ENV: &str = "RVLLM_METAL_BF16_ACCUM";
    const QKV_PREFILL_ENV: &str = "RVLLM_METAL_QKV_PREFILL";
    const MMA_PREFILL_ENV: &str = "RVLLM_METAL_PREFILL_GEMM";
    let guard = MetalDebugEnvGuard::new(&[BF16_ACCUM_ENV, QKV_PREFILL_ENV, MMA_PREFILL_ENV]);
    guard.remove(BF16_ACCUM_ENV);
    guard.remove(QKV_PREFILL_ENV);
    guard.remove(MMA_PREFILL_ENV);
    let f16_native = metal_numeric_abi_fingerprint(MetalFloatType::F16, false, false);
    let f16_native_again = metal_numeric_abi_fingerprint(MetalFloatType::F16, false, false);
    let bf16_native = metal_numeric_abi_fingerprint(MetalFloatType::Bf16, false, false);
    guard.set(QKV_PREFILL_ENV, "batch8");
    let bf16_batch8 = metal_numeric_abi_fingerprint(MetalFloatType::Bf16, false, false);
    guard.remove(QKV_PREFILL_ENV);
    guard.set(MMA_PREFILL_ENV, "mma32");
    let bf16_mma32 = metal_numeric_abi_fingerprint(MetalFloatType::Bf16, false, false);
    guard.remove(MMA_PREFILL_ENV);
    guard.set(BF16_ACCUM_ENV, "quantized");
    let bf16_quantized_accum = metal_numeric_abi_fingerprint(MetalFloatType::Bf16, false, false);
    let f16_int8_opt_in = metal_numeric_abi_fingerprint(MetalFloatType::F16, true, false);
    let f16_int8_active = metal_numeric_abi_fingerprint(MetalFloatType::F16, true, true);
    let packaged_a = metal_numeric_abi_fingerprint_impl(
        MetalFloatType::F16,
        false,
        false,
        None,
        Some([0x11; 32]),
        MetalLowBitResidencyPolicy::HybridFallback,
        rvllm_apple_metal::MetalKernelOptions::from_development_environment(),
    );
    let packaged_b = metal_numeric_abi_fingerprint_impl(
        MetalFloatType::F16,
        false,
        false,
        None,
        Some([0x22; 32]),
        MetalLowBitResidencyPolicy::HybridFallback,
        rvllm_apple_metal::MetalKernelOptions::from_development_environment(),
    );
    let replace_native = metal_numeric_abi_fingerprint_impl(
        MetalFloatType::F16,
        false,
        false,
        None,
        Some([0x11; 32]),
        MetalLowBitResidencyPolicy::ReplaceNative,
        rvllm_apple_metal::MetalKernelOptions::from_development_environment(),
    );

    assert_eq!(f16_native, f16_native_again);
    assert_ne!(f16_native, [0; 32]);
    assert_ne!(f16_native, bf16_native);
    assert_ne!(bf16_native, bf16_batch8);
    assert_ne!(bf16_native, bf16_mma32);
    assert_ne!(bf16_batch8, bf16_mma32);
    assert_ne!(bf16_native, bf16_quantized_accum);
    assert_ne!(f16_native, f16_int8_opt_in);
    assert_ne!(f16_int8_opt_in, f16_int8_active);
    assert_ne!(f16_native, f16_int8_active);
    assert_ne!(f16_native, packaged_a);
    assert_ne!(packaged_a, packaged_b);
    assert_ne!(packaged_a, replace_native);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn runtime_low_bit_replacement(
    tensor_name: impl Into<String>,
    format: AppleLowBitWeightFormat,
    shape: [usize; 2],
) -> MetalLowBitWeightReplacement {
    let row_bytes = match format {
        AppleLowBitWeightFormat::W4A16 => shape[1].div_ceil(2),
        AppleLowBitWeightFormat::W8A16 => shape[1],
    };
    MetalLowBitWeightReplacement {
        tensor_name: tensor_name.into(),
        format,
        shape,
        packed_values_bytes: shape[0] * row_bytes,
        scales_bytes: shape[0]
            * shape[1].div_ceil(APPLE_LOW_BIT_GROUP_SIZE)
            * std::mem::size_of::<half::f16>(),
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn low_bit_residency_policy_defaults_to_hybrid_and_requires_explicit_replacement() {
    let model_dir = temp_fixture_dir();
    let hybrid = ModelMetalBackend::new(model_dir.clone());
    assert_eq!(
        hybrid.low_bit_residency_policy,
        MetalLowBitResidencyPolicy::HybridFallback
    );
    let replacement = ModelMetalBackend::new(model_dir.clone())
        .with_low_bit_residency_policy(MetalLowBitResidencyPolicy::ReplaceNative);
    assert_eq!(
        replacement.low_bit_residency_policy,
        MetalLowBitResidencyPolicy::ReplaceNative
    );
    let _ = fs::remove_dir_all(model_dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn two_layer_low_bit_preflight_is_sorted_complete_and_fail_closed() {
    let dir = write_tiny_two_layer_fixture(false);
    let shape = [128, 256];
    let layer_zero = runtime_low_bit_replacement(
        "model.layers.0.mlp.down_proj.weight",
        AppleLowBitWeightFormat::W4A16,
        shape,
    );
    let layer_one = runtime_low_bit_replacement(
        "model.layers.1.mlp.down_proj.weight",
        AppleLowBitWeightFormat::W8A16,
        shape,
    );
    let sorted = preflight_low_bit_replacement_descriptors(
        &dir,
        MetalFloatType::F16,
        &[layer_one.clone(), layer_zero.clone()],
    )
    .expect("preflight two sidecars");
    assert_eq!(
        sorted
            .iter()
            .map(|replacement| replacement.tensor_name.as_str())
            .collect::<Vec<_>>(),
        [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
    );
    assert_eq!(
        ModelMetalBackend::hybrid_low_bit_arena_budget_bytes(&sorted).expect("exact hybrid budget"),
        layer_zero.packed_values_bytes
            + layer_zero.scales_bytes
            + layer_one.packed_values_bytes
            + layer_one.scales_bytes
    );

    assert!(preflight_low_bit_replacement_descriptors(
        &dir,
        MetalFloatType::F16,
        &[layer_zero.clone(), layer_zero.clone()],
    )
    .is_err());
    assert!(preflight_low_bit_replacement_descriptors(
        &dir,
        MetalFloatType::F16,
        &[runtime_low_bit_replacement(
            "model.layers.9.mlp.down_proj.weight",
            AppleLowBitWeightFormat::W4A16,
            shape,
        )],
    )
    .is_err());
    let mut wrong_shape = layer_zero.clone();
    wrong_shape.shape[1] -= 1;
    wrong_shape.packed_values_bytes = wrong_shape.shape[0] * wrong_shape.shape[1].div_ceil(2);
    wrong_shape.scales_bytes = wrong_shape.shape[0]
        * wrong_shape.shape[1].div_ceil(APPLE_LOW_BIT_GROUP_SIZE)
        * std::mem::size_of::<half::f16>();
    assert!(
        preflight_low_bit_replacement_descriptors(&dir, MetalFloatType::F16, &[wrong_shape],)
            .is_err()
    );
    let mut incomplete = layer_one;
    incomplete.scales_bytes -= 1;
    assert!(
        preflight_low_bit_replacement_descriptors(&dir, MetalFloatType::F16, &[incomplete],)
            .is_err()
    );
    assert!(
        preflight_low_bit_replacement_descriptors(&dir, MetalFloatType::Bf16, &[layer_zero],)
            .is_err()
    );
    let _ = fs::remove_dir_all(dir);
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<half::f16>());
    for value in values {
        let bits = half::f16::from_f32(*value).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<half::bf16>());
    for value in values {
        let bits = half::bf16::from_f32(*value).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_dtype_probe_fixture(dtype: &str, data: &[u8]) -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let mut header = Map::<String, Value>::new();
    let mut meta = Map::new();
    meta.insert("dtype".to_owned(), Value::String(dtype.to_owned()));
    meta.insert(
        "shape".to_owned(),
        Value::Array(vec![Value::Number(1u64.into())]),
    );
    meta.insert(
        "data_offsets".to_owned(),
        Value::Array(vec![
            Value::Number(0u64.into()),
            Value::Number((data.len() as u64).into()),
        ]),
    );
    header.insert("probe.weight".to_owned(), Value::Object(meta));

    let header_json = serde_json::to_string(&header).expect("serialize dtype probe header");
    let mut out =
        File::create(dir.join("model.safetensors")).expect("create dtype probe safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(data).expect("write payload");
    dir
}

fn write_tiny_zero_layer_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let embedding = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let norm = [1.0, 1.0, 1.0, 1.0];
    let lm_head = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0,
    ];

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[4, 4],
        &mut payload,
        &mut header,
    );
    add_tensor("model.norm.weight", &norm, &[4], &mut payload, &mut header);
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[4, 4],
        &mut payload,
        &mut header,
    );

    let config = r#"{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {
    "num_hidden_layers": 0,
    "hidden_size": 4,
    "intermediate_size": 8,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "vocab_size": 4,
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }
}"#;

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

fn zero_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
    rvllm_apple::AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: 1,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: 4,
        ane_intermediate_size: 8,
        ane_num_layers: 1,
        model_layout_hash: [0u8; 32],
        weights_path: Some(model_dir),
    }
}

#[test]
fn tiny_zero_layer_fixture_has_expected_files() {
    let dir = write_tiny_zero_layer_fixture();
    assert!(dir.join("config.json").is_file());
    assert!(dir.join("model.safetensors").is_file());

    let config_raw = fs::read_to_string(dir.join("config.json")).expect("read config");
    let config: Value = serde_json::from_str(&config_raw).expect("parse config");
    assert_eq!(config["architectures"][0], "Gemma4ForCausalLM");
    assert_eq!(config["text_config"]["num_hidden_layers"], 0);
    assert_eq!(config["text_config"]["vocab_size"], 4);

    #[cfg(target_os = "macos")]
    {
        let tensors = scan_safetensor_tensors(&dir).expect("read fixture tensors");
        let embed = tensors
            .get("model.embed_tokens.weight")
            .expect("embed tensor");
        let norm = tensors.get("model.norm.weight").expect("norm tensor");
        let lm_head = tensors.get("lm_head.weight").expect("lm_head tensor");
        assert_eq!(embed.shape, vec![4, 4]);
        assert_eq!(norm.shape, vec![4]);
        assert_eq!(lm_head.shape, vec![4, 4]);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
#[test]
fn model_metal_backend_non_macos_fails_closed() {
    let mut backend = RuntimeMetalBackend::new();
    let plan = rvllm_apple::AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: 1,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: 4,
        ane_intermediate_size: 8,
        ane_num_layers: 1,
        model_layout_hash: [0u8; 32],
        weights_path: None,
    };
    assert!(backend.prepare(&plan).is_err());
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_zero_layer_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_zero_layer_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend.prepare(&plan).expect("prepare tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_kv_page_io_fails_closed_before_prepare() {
    let mut backend = ModelMetalBackend::new(std::path::PathBuf::new());
    assert_eq!(backend.page_bytes(), None);
    assert!(matches!(
        backend.capture_page(rvllm_core::BlockId(0)),
        Err(KvPageIoError::BackendUnavailable)
    ));
    assert!(matches!(
        backend.restore_page(rvllm_core::BlockId(0), &[0; 2]),
        Err(KvPageIoError::BackendUnavailable)
    ));
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn model_kv_page_io_roundtrips_multilayer_bits_and_rejects_gpu_ownership() {
    let dir = write_tiny_two_layer_fixture(false);
    let config_path = dir.join("config.json");
    let config = fs::read_to_string(&config_path)
        .expect("read fixture config")
        .replace(
            "\"max_position_embeddings\": 16",
            "\"max_position_embeddings\": 64",
        );
    fs::write(&config_path, config).expect("extend fixture to two physical KV pages");
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = two_layer_plan(dir.clone());
    backend.prepare(&plan).expect("prepare two-layer model");

    let layout = backend.kv_page_layout().expect("validated KV layout");
    assert_eq!(layout.layers.len(), 2);
    assert!(layout.total_pages >= 2);
    let page_bytes = backend.page_bytes().expect("prepared page byte size");
    assert_eq!(page_bytes, layout.serialized_page_bytes);
    let source_bits: Vec<u8> = (0..page_bytes)
        .map(|index| ((index * 131 + 17) & 0xff) as u8)
        .collect();
    let other_bits: Vec<u8> = (0..page_bytes)
        .map(|index| ((index * 29 + 203) & 0xff) as u8)
        .collect();

    backend
        .restore_page(rvllm_core::BlockId(0), &source_bits)
        .expect("restore exact source bits");
    backend
        .restore_page(rvllm_core::BlockId(1), &other_bits)
        .expect("restore distinct destination bits");
    assert_eq!(
        backend
            .capture_page(rvllm_core::BlockId(0))
            .expect("capture source")
            .as_ref(),
        source_bits
    );
    assert_eq!(
        backend
            .capture_page(rvllm_core::BlockId(1))
            .expect("capture destination")
            .as_ref(),
        other_bits
    );
    backend
        .copy_page(CowPageCopy {
            source: rvllm_core::BlockId(0),
            destination: rvllm_core::BlockId(1),
            valid_tokens: 17,
        })
        .expect("copy full physical page for partial-tail COW");
    assert_eq!(
        backend
            .capture_page(rvllm_core::BlockId(1))
            .expect("capture copied page")
            .as_ref(),
        source_bits,
        "COW must preserve every F16/BF16 payload bit in canonical layer/K/V order"
    );
    assert!(matches!(
        backend.capture_page(rvllm_core::BlockId(layout.total_pages)),
        Err(KvPageIoError::InvalidPageId { .. })
    ));
    assert!(matches!(
        backend.restore_page(rvllm_core::BlockId(0), &source_bits[..page_bytes - 1]),
        Err(KvPageIoError::InvalidPageBytes { .. })
    ));

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let ticket = backend
        .launch_rollout(&handoff, None)
        .expect("commit asynchronous rollout");
    assert!(backend.in_flight.is_submitted(ticket));
    assert!(matches!(
        backend.capture_page(rvllm_core::BlockId(0)),
        Err(KvPageIoError::BackendBusy)
    ));
    assert!(matches!(
        backend.restore_page(rvllm_core::BlockId(0), &source_bits),
        Err(KvPageIoError::BackendBusy)
    ));
    assert!(matches!(
        backend.copy_page(CowPageCopy {
            source: rvllm_core::BlockId(0),
            destination: rvllm_core::BlockId(1),
            valid_tokens: 17,
        }),
        Err(KvPageIoError::BackendBusy)
    ));
    backend.collect(ticket).expect("collect rollout");
    backend
        .capture_page(rvllm_core::BlockId(0))
        .expect("collection releases page I/O ownership");

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn model_metal_async_submission_uses_three_independent_execution_slots() {
    let dir = write_tiny_one_layer_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend.prepare(&plan).expect("prepare tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let ticket = backend
        .launch_rollout(&handoff, None)
        .expect("commit asynchronous rollout");
    assert!(
        backend.in_flight.is_submitted(ticket),
        "launch must retain a submitted command buffer instead of manufacturing a ready result"
    );
    assert_eq!(
        backend.probe_perf_stats().forced_waits,
        0,
        "normal launch must not wait for Metal completion"
    );

    let second_handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(2)],
        vec![rvllm_core::TokenId(7)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let second_ticket = backend
        .launch_rollout(&second_handoff, None)
        .expect("a second committed step must own another execution slot");
    assert_ne!(
        backend.in_flight.execution_slot(ticket).unwrap(),
        backend.in_flight.execution_slot(second_ticket).unwrap()
    );
    let first_state = &backend.execution_states[backend.in_flight.execution_slot(ticket).unwrap()];
    let second_state =
        &backend.execution_states[backend.in_flight.execution_slot(second_ticket).unwrap()];
    assert_ne!(first_state.token_ids.offset, second_state.token_ids.offset);
    let arena = backend.arena.as_ref().expect("prepared arena");
    let first_resident_token = unsafe { *(arena.host_ptr(&first_state.token_ids) as *const u32) };
    let second_resident_token = unsafe { *(arena.host_ptr(&second_state.token_ids) as *const u32) };
    assert_eq!(
        (first_resident_token, second_resident_token),
        (2, 7),
        "each live launch must retain private token metadata"
    );

    let third_handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(3)],
        vec![rvllm_core::TokenId(5)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let third_ticket = backend
        .launch_rollout(&third_handoff, None)
        .expect("third execution slot must be admissible");
    let full_error = backend
        .launch_rollout(&third_handoff, None)
        .expect_err("fourth uncollected step must fail closed");
    assert!(format!("{full_error}").contains("in_flight_ring_full"));

    let output = match backend.try_collect(ticket).expect("poll submission") {
        Some(output) => output,
        None => backend.collect(ticket).expect("wait for submitted rollout"),
    };
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].token_id, rvllm_core::TokenId(3));
    let second_output = backend
        .collect(second_ticket)
        .expect("collect second rollout");
    assert_eq!(second_output.len(), 1);
    assert_eq!(
        backend.collect(third_ticket).expect("collect third").len(),
        1
    );

    let next_ticket = backend
        .launch_rollout(&handoff, None)
        .expect("collection must release an execution slot");
    assert_eq!(backend.collect(next_ticket).expect("collect next").len(), 1);

    // Build an explicitly enqueued-but-uncommitted command buffer so polling
    // has a deterministic not-ready state independent of GPU speed.
    let queue = backend
        .ctx
        .as_ref()
        .expect("prepared context")
        .queue_retained();
    let command_buffer = queue.commandBuffer().expect("empty command buffer");
    command_buffer.enqueue();
    let poll_ticket = backend
        .in_flight
        .reserve(999, AppleLaunchKind::Prefill, None)
        .expect("reserve polling ticket");
    backend
        .in_flight
        .submit(
            poll_ticket,
            ModelGpuSubmission {
                command_buffer: command_buffer.clone(),
                output: ModelGpuOutput::Prefill,
                perf_before: backend.perf.snapshot(),
                wall_start: Instant::now(),
                num_tokens: 0,
                is_decode: false,
            },
        )
        .expect("bind enqueued command buffer");
    assert_eq!(
        backend.try_collect(poll_ticket).expect("nonblocking poll"),
        None,
        "poll must preserve a ticket whose command buffer is not complete"
    );
    assert!(backend.in_flight.is_submitted(poll_ticket));
    assert!(matches!(
        backend.prefill_for_ane(&paged_contract_handoff([0xA5; 32])),
        Err(crate::ane_prefill::AnePrefillError::Cache(
            KvPageIoError::BackendBusy
        ))
    ));
    command_buffer.commit();
    assert!(backend
        .collect(poll_ticket)
        .expect("blocking collect waits and reclaims")
        .is_empty());
    assert_eq!(backend.in_flight.len(), 0);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn metal_probe_microbench_counters_hook_reports_decode_work() {
    let dir = write_tiny_zero_layer_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend.prepare(&plan).expect("prepare tiny model");

    for req in [1_u64, 2_u64] {
        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(req)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );
        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
    }

    let stats = backend.probe_perf_stats();
    eprintln!("metal_probe_microbench_counters_hook stats: {stats:?}");
    assert_eq!(stats.decode_steps, 2);
    assert_eq!(stats.last_step_tokens, 1);
    assert!(stats.command_buffers > 0);
    assert!(stats.encoders > 0);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_zero_layer_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_zero_layer_fixture();
    let plan = zero_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty(), "zero-layer prefill returns no tokens");

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));

    assert!(
        !engine.has_pending_work(),
        "request should finish after one decoded token"
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_noop_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    // token 3 should be chosen if dim 7 is high
    for d in 0..hidden {
        lm_head[3 * hidden + d] = if d == 7 { 2.0 } else { 0.0 };
        lm_head[2 * hidden + d] = if d == 7 { 1.0 } else { 0.0 };
    }

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    // One layer tensors
    let ones = vec![1.0f32; hidden];
    let zeros_qkv = vec![0.0f32; 3 * hidden * hidden];
    let zeros_o = vec![0.0f32; hidden * hidden];
    let zeros_gate = vec![0.0f32; 2 * intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.qkv.weight",
        &zeros_qkv,
        &[3 * hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &zeros_o,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_up.weight",
        &zeros_gate,
        &[2 * intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &zeros_down,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

fn one_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
    rvllm_apple::AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: 1,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: 128,
        ane_intermediate_size: 256,
        ane_num_layers: 1,
        model_layout_hash: [0u8; 32],
        weights_path: Some(model_dir),
    }
}

fn two_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
    n_layer_plan(model_dir, 2)
}

fn n_layer_plan(model_dir: std::path::PathBuf, num_layers: usize) -> rvllm_apple::AppleRuntimePlan {
    rvllm_apple::AppleRuntimePlan {
        ane_num_layers: num_layers,
        ..one_layer_plan(model_dir)
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_two_layer_fixture(first_layer_ffn_nonzero: bool) -> std::path::PathBuf {
    write_tiny_n_layer_fixture(2, first_layer_ffn_nonzero)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_n_layer_fixture(
    num_layers: usize,
    first_layer_ffn_nonzero: bool,
) -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    if first_layer_ffn_nonzero {
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;
    } else {
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 7] = 2.0;
    }

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_qkv = vec![0.0f32; 3 * hidden * hidden];
    let zeros_o = vec![0.0f32; hidden * hidden];

    for layer_idx in 0..num_layers {
        let mut gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        if first_layer_ffn_nonzero && layer_idx == 0 {
            gate_up[7] = 0.5;
            gate_up[intermediate * hidden + 7] = 0.5;
            down_proj[9 * intermediate] = 4.0;
        }

        add_tensor(
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
            &zeros_qkv,
            &[3 * hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            &zeros_o,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp_norm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
            &gate_up,
            &[2 * intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );
    }

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": {},
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        num_layers, hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_two_layer_sliding_global_noop_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;
    let sliding_head_dim = 128;
    let global_head_dim = 256;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 7] = 2.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_gate_up = vec![0.0f32; 2 * intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];

    for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
        let zeros_qkv = vec![0.0f32; 3 * head_dim * hidden];
        let zeros_o = vec![0.0f32; hidden * head_dim];
        add_tensor(
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
            &zeros_qkv,
            &[3 * head_dim, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            &zeros_o,
            &[hidden, head_dim],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp_norm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
            &zeros_gate_up,
            &[2 * intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            &zeros_down,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );
    }

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "global_head_dim": {},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, sliding_head_dim, global_head_dim, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_two_layer_noop_fixture() -> std::path::PathBuf {
    write_tiny_two_layer_fixture(false)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_two_layer_first_ffn_nonzero_fixture() -> std::path::PathBuf {
    write_tiny_two_layer_fixture(true)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_n_layer_noop_fixture(num_layers: usize) -> std::path::PathBuf {
    write_tiny_n_layer_fixture(num_layers, false)
}

fn rmsnorm_f32(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let hidden = input.len();
    let sum_sq = input.iter().map(|v| v * v).sum::<f32>();
    let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * inv_rms * g)
        .collect()
}

fn gemm_f32(input: &[f32], weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0.0f32;
        for col in 0..in_dim {
            acc += input[col] * weights[row * in_dim + col];
        }
        out[row] = acc;
    }
    out
}

fn gelu_tanh_f32(x: f32) -> f32 {
    let c = 0.7978845608f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn cpu_reference_one_layer_ffn_nonzero_argmax() -> usize {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 0.000001f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    let norm = vec![1.0f32; hidden];

    let mut residual = vec![0.0f32; hidden];
    let embedding_scale = (hidden as f32).sqrt();
    for dim in 0..hidden {
        residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
    }

    let mlp_input = rmsnorm_f32(&residual, &norm, eps);
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    gate_proj[7] = 0.5;
    up_proj[7] = 0.5;
    down_proj[9 * intermediate] = 4.0;

    let gate = gemm_f32(&mlp_input, &gate_proj, intermediate, hidden);
    let up = gemm_f32(&mlp_input, &up_proj, intermediate, hidden);
    let mut activated = vec![0.0f32; intermediate];
    for dim in 0..intermediate {
        activated[dim] = gelu_tanh_f32(gate[dim]) * up[dim];
    }
    let mlp_out = gemm_f32(&activated, &down_proj, hidden, intermediate);
    for dim in 0..hidden {
        residual[dim] += mlp_out[dim];
    }

    let final_hidden = rmsnorm_f32(&residual, &norm, eps);
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;
    let logits = gemm_f32(&final_hidden, &lm_head, vocab, hidden);
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite logits"))
        .map(|(idx, _)| idx)
        .expect("nonempty logits")
}

fn cpu_reference_one_layer_attention_nonzero_argmax() -> usize {
    let hidden = 128usize;
    let vocab = 8usize;
    let eps = 0.000001f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    let norm = vec![1.0f32; hidden];

    let mut residual = vec![0.0f32; hidden];
    let embedding_scale = (hidden as f32).sqrt();
    for dim in 0..hidden {
        residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
    }

    let attn_input = rmsnorm_f32(&residual, &norm, eps);
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 2.0;
    o_proj[9 * hidden + 11] = 6.0;

    let q = gemm_f32(&attn_input, &q_proj, hidden, hidden);
    let k = gemm_f32(&attn_input, &k_proj, hidden, hidden);
    let v = gemm_f32(&attn_input, &v_proj, hidden, hidden);
    let score = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    assert!(score.is_finite());

    let attn_out = v;
    let attn_residual = gemm_f32(&attn_out, &o_proj, hidden, hidden);
    for dim in 0..hidden {
        residual[dim] += attn_residual[dim];
    }

    let final_hidden = rmsnorm_f32(&residual, &norm, eps);
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;
    let logits = gemm_f32(&final_hidden, &lm_head, vocab, hidden);
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite logits"))
        .map(|(idx, _)| idx)
        .expect("nonempty logits")
}

struct CpuGqaAttentionReference {
    residual: Vec<f32>,
    logits: Vec<f32>,
}

fn cpu_reference_gqa_attention(
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> CpuGqaAttentionReference {
    assert!(num_heads > 0);
    assert!(num_kv_heads > 0);
    assert_eq!(num_heads % num_kv_heads, 0);
    assert!(GQA_SOURCE_DIM < hidden);
    assert!(GQA_OUTPUT_DIM < hidden);
    assert!(GQA_VALUE_DIM < head_dim);
    assert!(GQA_OUTPUT_HEAD < num_heads);

    let vocab = 8usize;
    let eps = 0.000001f32;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let output_col = GQA_OUTPUT_HEAD * head_dim + GQA_VALUE_DIM;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + GQA_SOURCE_DIM] = 10.0;
    let norm = vec![1.0f32; hidden];

    let mut residual = vec![0.0f32; hidden];
    let embedding_scale = (hidden as f32).sqrt();
    for dim in 0..hidden {
        residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
    }

    let attn_input = rmsnorm_f32(&residual, &norm, eps);
    let mut q_proj = vec![0.0f32; q_dim * hidden];
    let mut k_proj = vec![0.0f32; kv_dim * hidden];
    let mut v_proj = vec![0.0f32; kv_dim * hidden];
    let mut o_proj = vec![0.0f32; hidden * q_dim];
    q_proj[GQA_SOURCE_DIM] = 0.25;
    k_proj[GQA_SOURCE_DIM] = 0.125;
    v_proj[GQA_VALUE_DIM * hidden + GQA_SOURCE_DIM] = 2.0;
    o_proj[GQA_OUTPUT_DIM * q_dim + output_col] = 6.0;

    let q = gemm_f32(&attn_input, &q_proj, q_dim, hidden);
    let k = gemm_f32(&attn_input, &k_proj, kv_dim, hidden);
    let v = gemm_f32(&attn_input, &v_proj, kv_dim, hidden);
    let mut attn_out = vec![0.0f32; q_dim];
    for head in 0..num_heads {
        let kv_head = head * num_kv_heads / num_heads;
        let q_base = head * head_dim;
        let kv_base = kv_head * head_dim;
        let score = q[q_base..q_base + head_dim]
            .iter()
            .zip(k[kv_base..kv_base + head_dim].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / (head_dim as f32).sqrt();
        assert!(score.is_finite());
        attn_out[q_base..q_base + head_dim].copy_from_slice(&v[kv_base..kv_base + head_dim]);
    }

    let attn_residual = gemm_f32(&attn_out, &o_proj, hidden, q_dim);
    for dim in 0..hidden {
        residual[dim] += attn_residual[dim];
    }

    let final_hidden = rmsnorm_f32(&residual, &norm, eps);
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + GQA_SOURCE_DIM] = 1.0;
    lm_head[3 * hidden + GQA_OUTPUT_DIM] = 4.0;
    let logits = gemm_f32(&final_hidden, &lm_head, vocab, hidden);
    CpuGqaAttentionReference { residual, logits }
}

fn cpu_reference_multihead_gqa_attention() -> CpuGqaAttentionReference {
    cpu_reference_gqa_attention(128, 4, 2, 32)
}

fn cpu_reference_multihead_gqa_attention_logits() -> Vec<f32> {
    cpu_reference_multihead_gqa_attention().logits
}

fn cpu_reference_multihead_gqa_attention_argmax() -> usize {
    cpu_full_nonzero_argmax(&cpu_reference_multihead_gqa_attention_logits())
}

fn cpu_reference_qdim_not_hidden() -> CpuGqaAttentionReference {
    cpu_reference_gqa_attention(64, 4, 2, 32)
}

fn cpu_reference_qdim_not_hidden_logits() -> Vec<f32> {
    cpu_reference_qdim_not_hidden().logits
}

fn cpu_reference_qdim_not_hidden_argmax() -> usize {
    cpu_full_nonzero_argmax(&cpu_reference_qdim_not_hidden_logits())
}

fn expected_gqa_attention_residual(hidden: usize) -> Vec<f32> {
    let mut residual = vec![0.0f32; hidden];
    residual[GQA_SOURCE_DIM] = 10.0 * (hidden as f32).sqrt();
    residual[GQA_OUTPUT_DIM] = 12.0 * (hidden as f32).sqrt();
    residual
}

#[test]
fn cpu_reference_one_layer_ffn_nonzero_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_ffn_nonzero_argmax(), 3);
}

#[test]
fn cpu_reference_one_layer_attention_nonzero_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_attention_nonzero_argmax(), 3);
}

#[test]
fn cpu_reference_multihead_gqa_attention_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_multihead_gqa_attention_argmax(), 3);
}

#[test]
fn cpu_reference_qdim_not_hidden_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_qdim_not_hidden_argmax(), 3);
}

#[test]
fn cpu_reference_multihead_gqa_attention_full_logits_are_stable() {
    let logits = cpu_reference_multihead_gqa_attention_logits();
    let expected = [0.0f32, 0.0, 7.242_86, 34.765_73, 0.0, 0.0, 0.0, 0.0];

    assert_eq!(logits.len(), 8);
    assert_eq!(cpu_full_nonzero_argmax(&logits), 3);
    assert_f32_slice_close(
        "multi-head GQA attention CPU logits",
        &logits,
        &expected,
        0.01,
    );
}

#[test]
fn cpu_reference_qdim_not_hidden_full_logits_are_stable() {
    let logits = cpu_reference_qdim_not_hidden_logits();
    let expected = [0.0f32, 0.0, 5.121_48, 24.583_08, 0.0, 0.0, 0.0, 0.0];

    assert_eq!(logits.len(), 8);
    assert_eq!(cpu_full_nonzero_argmax(&logits), 3);
    assert_f32_slice_close("q_dim != hidden CPU logits", &logits, &expected, 0.01);
}

#[test]
fn cpu_reference_multihead_gqa_attention_residual_vector_is_stable() {
    let reference = cpu_reference_multihead_gqa_attention();
    let expected = expected_gqa_attention_residual(128);

    assert_eq!(reference.residual.len(), 128);
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);
    assert_f32_slice_close(
        "multi-head GQA attention CPU residual",
        &reference.residual,
        &expected,
        0.01,
    );
}

#[test]
fn cpu_reference_qdim_not_hidden_residual_vector_is_stable() {
    let reference = cpu_reference_qdim_not_hidden();
    let expected = expected_gqa_attention_residual(64);

    assert_eq!(reference.residual.len(), 64);
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);
    assert_f32_slice_close(
        "q_dim != hidden CPU residual",
        &reference.residual,
        &expected,
        0.01,
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_noop_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_one_layer_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_noop_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_one_layer_noop_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_noop_prefill_batch_two_then_decode_batch_two_returns_token_3() {
    let dir = write_tiny_one_layer_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer tiny model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(2)],
        vec![0, 1, 2],
        vec![0, 0],
        vec![1, 1],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run batch prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let decode = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(2)],
        vec![0, 1, 2],
        vec![0, 0],
        vec![1, 1],
    );
    let decode_ticket = backend
        .launch_rollout(&decode, None)
        .expect("run batch decode");
    let logits = backend
        .debug_read_decode_logits_f32(2)
        .expect("read batched decode logits");
    assert_eq!(logits.len(), 16);
    assert_eq!(cpu_full_nonzero_argmax(&logits[0..8]), 3);
    assert_eq!(cpu_full_nonzero_argmax(&logits[8..16]), 3);
    let out = backend.collect(decode_ticket).expect("collect decode");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));
    assert_eq!(out[1].req_id, rvllm_core::ReqId(2));
    assert_eq!(out[1].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_two_layer_noop_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_two_layer_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = two_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare two-layer no-op tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_two_layer_first_ffn_nonzero_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_two_layer_first_ffn_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = two_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare two-layer first-ffn-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_two_layer_first_ffn_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_two_layer_first_ffn_nonzero_fixture();
    let plan = two_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny two-layer first-ffn-nonzero model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_three_layer_noop_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_n_layer_noop_fixture(3);
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = n_layer_plan(dir.clone(), 3);
    backend
        .prepare(&plan)
        .expect("prepare three-layer no-op tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_four_layer_noop_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_n_layer_noop_fixture(4);
    let plan = n_layer_plan(dir.clone(), 4);

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny four-layer no-op model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_two_layer_sliding_global_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_two_layer_sliding_global_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = two_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare two-layer sliding/global tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_two_layer_sliding_global_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_two_layer_sliding_global_noop_fixture();
    let plan = two_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny two-layer sliding/global model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_hf_style_noop_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    // token 3 should be chosen if dim 7 is high
    for d in 0..hidden {
        lm_head[3 * hidden + d] = if d == 7 { 2.0 } else { 0.0 };
        lm_head[2 * hidden + d] = if d == 7 { 1.0 } else { 0.0 };
    }

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    // One layer tensors (HF style separate)
    let ones = vec![1.0f32; hidden];
    let zeros_qkvo = vec![0.0f32; hidden * hidden];
    let zeros_gate_up = vec![0.0f32; intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &zeros_down,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_ffn_nonzero_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_qkvo = vec![0.0f32; hidden * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = zeros_down;
    gate_proj[7] = 0.5;
    up_proj[7] = 0.5;
    down_proj[9 * intermediate] = 4.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_zero_layer_decode_loop_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 2] = 10.0;
    embedding[3 * hidden + 3] = 10.0;
    embedding[4 * hidden + 4] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[3 * hidden + 2] = 2.0;
    lm_head[4 * hidden + 3] = 2.0;
    lm_head[5 * hidden + 4] = 2.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 0,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden,
        hidden * 2,
        hidden,
        vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_attention_nonzero_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_gate_up = vec![0.0f32; intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 2.0;
    o_proj[9 * hidden + 11] = 6.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &zeros_down,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_gqa_attention_fixture(
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> std::path::PathBuf {
    assert!(num_heads > 0);
    assert!(num_kv_heads > 0);
    assert_eq!(num_heads % num_kv_heads, 0);
    assert!(GQA_SOURCE_DIM < hidden);
    assert!(GQA_OUTPUT_DIM < hidden);
    assert!(GQA_VALUE_DIM < head_dim);
    assert!(GQA_OUTPUT_HEAD < num_heads);

    let dir = temp_fixture_dir();
    let vocab = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let output_col = GQA_OUTPUT_HEAD * head_dim + GQA_VALUE_DIM;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + GQA_SOURCE_DIM] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + GQA_SOURCE_DIM] = 1.0;
    lm_head[3 * hidden + GQA_OUTPUT_DIM] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_gate_up = vec![0.0f32; intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];
    let mut q_proj = vec![0.0f32; q_dim * hidden];
    let mut k_proj = vec![0.0f32; kv_dim * hidden];
    let mut v_proj = vec![0.0f32; kv_dim * hidden];
    let mut o_proj = vec![0.0f32; hidden * q_dim];
    q_proj[GQA_SOURCE_DIM] = 0.25;
    k_proj[GQA_SOURCE_DIM] = 0.125;
    v_proj[GQA_VALUE_DIM * hidden + GQA_SOURCE_DIM] = 2.0;
    o_proj[GQA_OUTPUT_DIM * q_dim + output_col] = 6.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[q_dim, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[kv_dim, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[kv_dim, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, q_dim],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &zeros_gate_up,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &zeros_down,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": {},
    "num_key_value_heads": {},
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, num_heads, num_kv_heads, head_dim, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_multihead_gqa_attention_fixture() -> std::path::PathBuf {
    write_tiny_gqa_attention_fixture(128, 256, 4, 2, 32)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_qdim_not_hidden_attention_fixture() -> std::path::PathBuf {
    write_tiny_gqa_attention_fixture(64, 256, 4, 2, 32)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_shared_kv_tail_poison_fixture(
    poison_tail_local_kv: bool,
    poison_source_v: bool,
) -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;
    let source_layer = 1usize;
    let tail_layer = 2usize;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + GQA_SOURCE_DIM] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + GQA_SOURCE_DIM] = 1.0;
    lm_head[3 * hidden + GQA_OUTPUT_DIM] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_gate_up = vec![0.0f32; intermediate * hidden];
    let zeros_down = vec![0.0f32; hidden * intermediate];
    let zeros_proj = vec![0.0f32; hidden * hidden];

    for layer_idx in 0..3 {
        let lprefix = format!("model.layers.{layer_idx}");
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];

        if layer_idx == source_layer {
            v_proj[GQA_VALUE_DIM * hidden + GQA_SOURCE_DIM] =
                if poison_source_v { 0.0 } else { 2.0 };
        }
        if layer_idx == tail_layer {
            q_proj[GQA_SOURCE_DIM * hidden + GQA_SOURCE_DIM] = 0.25;
            o_proj[GQA_OUTPUT_DIM * hidden + GQA_VALUE_DIM] = 6.0;
            if poison_tail_local_kv {
                k_proj[GQA_VALUE_DIM * hidden + GQA_SOURCE_DIM] = 100.0;
                v_proj[GQA_VALUE_DIM * hidden + GQA_SOURCE_DIM] = -100.0;
            }
        }

        add_tensor(
            &format!("{lprefix}.input_layernorm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.self_attn.q_proj.weight"),
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.self_attn.k_proj.weight"),
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.self_attn.v_proj.weight"),
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.self_attn.o_proj.weight"),
            if layer_idx == tail_layer {
                &o_proj
            } else {
                &zeros_proj
            },
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.mlp_norm.weight"),
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.mlp.gate_proj.weight"),
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.mlp.up_proj.weight"),
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{lprefix}.mlp.down_proj.weight"),
            &zeros_down,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );
    }

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 3,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {hidden},
    "num_kv_shared_layers": 1,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention"],
    "vocab_size": {vocab},
    "max_position_embeddings": 16,
    "sliding_window": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

fn cpu_full_nonzero_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| v * scale * w)
        .collect()
}

fn cpu_full_nonzero_matvec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(input.len(), cols);
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let base = row * cols;
        out[row] = (0..cols).map(|col| weight[base + col] * input[col]).sum();
    }
    out
}

fn cpu_full_nonzero_gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

fn cpu_full_nonzero_argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best_value {
            best_idx = idx;
            best_value = *value;
        }
    }
    best_idx
}

fn cpu_full_nonzero_top_two(values: &[f32]) -> (usize, usize) {
    assert!(values.len() >= 2);
    let mut ranked = values.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).expect("finite logits"));
    (ranked[0].0, ranked[1].0)
}

fn cpu_reference_zero_layer_decode_loop_sequence() -> Vec<usize> {
    let hidden = 128usize;
    let vocab = 8usize;
    let eps = 0.000001f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 2] = 10.0;
    embedding[3 * hidden + 3] = 10.0;
    embedding[4 * hidden + 4] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[3 * hidden + 2] = 2.0;
    lm_head[4 * hidden + 3] = 2.0;
    lm_head[5 * hidden + 4] = 2.0;

    let mut current = 2usize;
    let mut out = Vec::new();
    for _ in 0..3 {
        let mut residual = embedding[current * hidden..(current + 1) * hidden].to_vec();
        let embed_scale = (hidden as f32).sqrt();
        for value in &mut residual {
            *value *= embed_scale;
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        current = cpu_full_nonzero_argmax(&logits);
        out.push(current);
    }
    out
}

struct CpuFullNonzeroOneLayerReference {
    residual_after_attention: Vec<f32>,
    residual: Vec<f32>,
    final_hidden: Vec<f32>,
    logits: Vec<f32>,
}

fn cpu_reference_one_layer_full_nonzero() -> CpuFullNonzeroOneLayerReference {
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;
    let eps = 0.000001f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 10.0;
    let norm = vec![1.0f32; hidden];

    let mut residual = embedding[2 * hidden..3 * hidden].to_vec();
    let embed_scale = (hidden as f32).sqrt();
    for value in &mut residual {
        *value *= embed_scale;
    }

    let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    q_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.25;
    k_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.125;
    v_proj[FULL_NONZERO_VALUE_DIM * hidden + FULL_NONZERO_ORIGINAL_DIM] = 2.0;
    o_proj[FULL_NONZERO_ATTENTION_DIM * hidden + FULL_NONZERO_VALUE_DIM] = 6.0;

    let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
    let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
    let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
    let _score = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    let attn_out = v;
    let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
    for (dst, src) in residual.iter_mut().zip(projected_attn.iter()) {
        *dst += src;
    }
    let residual_after_attention = residual.clone();

    let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    gate_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
    up_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
    down_proj[FULL_NONZERO_FFN_DIM * intermediate] = 4.0;

    let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
    let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
        .collect::<Vec<_>>();
    let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
    for (dst, src) in residual.iter_mut().zip(mlp_out.iter()) {
        *dst += src;
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 1.0;
    lm_head[3 * hidden + FULL_NONZERO_ATTENTION_DIM] = 4.0;
    let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);

    CpuFullNonzeroOneLayerReference {
        residual_after_attention,
        residual,
        final_hidden,
        logits,
    }
}

fn cpu_reference_one_layer_full_nonzero_logits() -> Vec<f32> {
    cpu_reference_one_layer_full_nonzero().logits
}

fn cpu_reference_one_layer_full_nonzero_argmax() -> usize {
    cpu_full_nonzero_argmax(&cpu_reference_one_layer_full_nonzero_logits())
}

fn cpu_reference_real_hf_style_one_layer_slice_argmax() -> usize {
    cpu_reference_one_layer_full_nonzero_argmax()
}

fn cpu_reference_one_layer_qkv_norm_nonzero_argmax(apply_qkv_norm: bool) -> usize {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 1e-6f32;

    let mut residual = vec![0.0f32; hidden];
    residual[7] = 10.0 * (hidden as f32).sqrt();

    let norm = vec![1.0f32; hidden];
    let mut q_norm = vec![1.0f32; hidden];
    let mut k_norm = vec![1.0f32; hidden];
    let v_norm = vec![1.0f32; hidden];
    q_norm[0] = 0.5;
    k_norm[0] = 0.25;

    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let gate_proj = vec![0.0f32; intermediate * hidden];
    let up_proj = vec![0.0f32; intermediate * hidden];
    let down_proj = vec![0.0f32; hidden * intermediate];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.25;
    o_proj[9 * hidden + 11] = 0.5;
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 32.0;

    let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let mut q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
    let mut k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
    let mut v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
    if apply_qkv_norm {
        q = cpu_full_nonzero_rms_norm(&q, &q_norm, eps);
        k = cpu_full_nonzero_rms_norm(&k, &k_norm, eps);
        v = cpu_full_nonzero_rms_norm(&v, &v_norm, eps);
    }

    let _single_key_score =
        q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    let attn_out = v;
    let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
    for d in 0..hidden {
        residual[d] += projected_attn[d];
    }

    let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
    let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
        .collect::<Vec<_>>();
    let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
    for d in 0..hidden {
        residual[d] += mlp_out[d];
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
    cpu_full_nonzero_argmax(&logits)
}

fn cpu_reference_one_layer_extra_norms_argmax(apply_extra_norms: bool) -> usize {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 1e-6f32;

    let mut residual = vec![0.0f32; hidden];
    residual[7] = 10.0 * (hidden as f32).sqrt();

    let norm = vec![1.0f32; hidden];
    let mut post_attn_norm = vec![1.0f32; hidden];
    let pre_ff_norm = vec![1.0f32; hidden];
    let post_ff_norm = vec![1.0f32; hidden];

    post_attn_norm[7] = 0.01;
    post_attn_norm[9] = 64.0;

    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let gate_proj = vec![0.0f32; intermediate * hidden];
    let up_proj = vec![0.0f32; intermediate * hidden];
    let down_proj = vec![0.0f32; hidden * intermediate];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.25;
    o_proj[9 * hidden + 11] = 0.5;
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
    let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
    let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
    let _single_key_score =
        q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
    for d in 0..hidden {
        residual[d] += projected_attn[d];
    }

    if apply_extra_norms {
        residual = cpu_full_nonzero_rms_norm(&residual, &post_attn_norm, eps);
    }

    let mlp_normed = cpu_full_nonzero_rms_norm(
        &residual,
        if apply_extra_norms {
            &pre_ff_norm
        } else {
            &norm
        },
        eps,
    );
    let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
    let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
        .collect::<Vec<_>>();
    let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
    for d in 0..hidden {
        residual[d] += mlp_out[d];
    }

    if apply_extra_norms {
        residual = cpu_full_nonzero_rms_norm(&residual, &post_ff_norm, eps);
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
    cpu_full_nonzero_argmax(&logits)
}

fn cpu_reference_one_layer_layer_scalar_argmax(apply_layer_scalar: bool) -> usize {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 1e-6f32;
    let mut layer_scalar = vec![1.0f32; hidden];
    if apply_layer_scalar {
        layer_scalar[9] = 6.0;
    }

    let mut residual = vec![0.0f32; hidden];
    residual[7] = 10.0 * (hidden as f32).sqrt();

    let norm = vec![1.0f32; hidden];
    let q_proj = vec![0.0f32; hidden * hidden];
    let k_proj = vec![0.0f32; hidden * hidden];
    let v_proj = vec![0.0f32; hidden * hidden];
    let o_proj = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    gate_proj[7] = 0.5;
    up_proj[7] = 0.5;
    down_proj[9 * intermediate] = 1.0;
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 1.0;

    let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
    let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
    let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
    let _single_key_score =
        q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
    for d in 0..hidden {
        residual[d] += projected_attn[d];
    }

    let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
    let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
        .collect::<Vec<_>>();
    let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
    for d in 0..hidden {
        residual[d] += mlp_out[d];
    }
    for d in 0..hidden {
        residual[d] *= layer_scalar[d];
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
    cpu_full_nonzero_argmax(&logits)
}

fn cpu_reference_one_layer_integrated_gemma_probe_argmax() -> usize {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 0.000001f32;
    let mut layer_scalar = vec![1.0f32; hidden];
    layer_scalar[9] = 3.0;
    let softcap = 6.0f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let mut residual = vec![0.0f32; hidden];
    let embedding_scale = (hidden as f32).sqrt();
    for dim in 0..hidden {
        residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
    }

    let input_norm = vec![1.0f32; hidden];
    let mut q_norm = vec![1.0f32; hidden];
    let mut k_norm = vec![1.0f32; hidden];
    let mut v_norm = vec![1.0f32; hidden];
    let mut post_attn_norm = vec![1.0f32; hidden];
    let mut pre_ff_norm = vec![1.0f32; hidden];
    let mut post_ff_norm = vec![1.0f32; hidden];
    let final_norm = vec![1.0f32; hidden];
    q_norm[0] = 0.75;
    k_norm[0] = 0.5;
    v_norm[11] = 1.25;
    post_attn_norm[9] = 4.0;
    pre_ff_norm[9] = 1.0;
    post_ff_norm[9] = 2.0;

    let attn_normed = cpu_full_nonzero_rms_norm(&residual, &input_norm, eps);
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.5;
    o_proj[9 * hidden + 11] = 0.2;

    let q = cpu_full_nonzero_rms_norm(
        &cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed),
        &q_norm,
        eps,
    );
    let k = cpu_full_nonzero_rms_norm(
        &cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed),
        &k_norm,
        eps,
    );
    let v = cpu_full_nonzero_rms_norm(
        &cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed),
        &v_norm,
        eps,
    );
    let score = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
    assert!(score.is_finite());

    let attn_residual = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
    let attn_residual = cpu_full_nonzero_rms_norm(&attn_residual, &post_attn_norm, eps);
    for dim in 0..hidden {
        residual[dim] += attn_residual[dim];
    }

    let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &pre_ff_norm, eps);
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    gate_proj[9] = 0.75;
    up_proj[9] = 0.75;
    down_proj[9 * intermediate] = 1.0;

    let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
    let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
        .collect::<Vec<_>>();
    let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
    let mlp_out = cpu_full_nonzero_rms_norm(&mlp_out, &post_ff_norm, eps);
    for dim in 0..hidden {
        residual[dim] += mlp_out[dim];
    }
    for dim in 0..hidden {
        residual[dim] *= layer_scalar[dim];
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &final_norm, eps);
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 1.0;
    let mut logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
    for logit in &mut logits {
        *logit = softcap * (*logit / softcap).tanh();
    }
    cpu_full_nonzero_argmax(&logits)
}

fn cpu_reference_prompt_len_two_prefill_logits(include_first_prompt_token: bool) -> Vec<f32> {
    let hidden = 128usize;
    let vocab = 8usize;
    let eps = 0.000001f32;
    let scale = (hidden as f32).sqrt();

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    embedding[4 * hidden + 5] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    q_proj[5] = 1.0;
    k_proj[7] = 1.0;
    k_proj[5] = -1.0;
    v_proj[11 * hidden + 7] = 2.0;
    o_proj[9 * hidden + 11] = 2.0;
    lm_head[2 * hidden + 5] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let token_residual = |token: usize| -> Vec<f32> {
        let mut residual = vec![0.0f32; hidden];
        for dim in 0..hidden {
            residual[dim] = embedding[token * hidden + dim] * scale;
        }
        residual
    };

    let prompt_tokens = if include_first_prompt_token {
        vec![2usize, 4usize]
    } else {
        vec![4usize]
    };

    let mut k_cache = Vec::with_capacity(prompt_tokens.len());
    let mut v_cache = Vec::with_capacity(prompt_tokens.len());
    for &token in &prompt_tokens {
        let residual = token_residual(token);
        let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        k_cache.push(cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed));
        v_cache.push(cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed));
    }

    let mut residual = token_residual(4);
    let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &normed);
    let decode_k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed);
    let decode_v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);
    let last_slot = k_cache.len() - 1;
    k_cache[last_slot] = decode_k;
    v_cache[last_slot] = decode_v;

    let mut scores = Vec::with_capacity(k_cache.len());
    for key in &k_cache {
        let score = q
            .iter()
            .zip(key.iter())
            .map(|(qv, kv)| qv * kv)
            .sum::<f32>()
            / (hidden as f32).sqrt();
        scores.push(score);
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for score in &scores {
        denom += (*score - max_score).exp();
    }

    let mut attn_out = vec![0.0f32; hidden];
    for (idx, value) in v_cache.iter().enumerate() {
        let weight = (scores[idx] - max_score).exp() / denom;
        for dim in 0..hidden {
            attn_out[dim] += value[dim] * weight;
        }
    }

    let projected = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
    for dim in 0..hidden {
        residual[dim] += projected[dim];
    }

    let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
    cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden)
}

fn cpu_reference_prompt_len_two_prefill_argmax(include_first_prompt_token: bool) -> usize {
    cpu_full_nonzero_argmax(&cpu_reference_prompt_len_two_prefill_logits(
        include_first_prompt_token,
    ))
}

#[derive(Debug)]
struct GeneratedTinyGemma4HfDecodeLoopReference {
    generated: Vec<usize>,
    logits_by_step: Vec<Vec<f32>>,
}

fn cpu_reference_generated_tiny_gemma4_hf_decode_loop(
    prompt: &[usize],
    steps: usize,
) -> GeneratedTinyGemma4HfDecodeLoopReference {
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab = 8usize;
    let eps = 0.000001f32;
    let scale = (hidden as f32).sqrt();
    let softcap = 30.0f32;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    embedding[3 * hidden + 6] = 10.0;
    embedding[4 * hidden + 5] = 10.0;

    let norm = vec![1.0f32; hidden];
    let q_norm = vec![1.0f32; hidden];
    let k_norm = vec![1.0f32; hidden];
    let layer_scalar = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    q_proj[5] = 0.5;
    k_proj[5] = 1.0;
    v_proj[9 * hidden + 5] = 1.0;
    o_proj[10 * hidden + 9] = 1.0;
    gate_proj[7] = 0.25;
    up_proj[7] = 0.25;
    down_proj[10 * intermediate] = 0.5;
    lm_head[2 * hidden + 10] = 0.25;
    lm_head[3 * hidden + 5] = 3.0;
    lm_head[5 * hidden + 6] = 3.0;

    let token_residual = |token: usize| -> Vec<f32> {
        let mut residual = vec![0.0f32; hidden];
        for dim in 0..hidden {
            residual[dim] = embedding[token * hidden + dim] * scale;
        }
        residual
    };

    let project_kv = |token: usize| -> (Vec<f32>, Vec<f32>) {
        let residual = token_residual(token);
        let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed);
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);
        (cpu_full_nonzero_rms_norm(&k, &k_norm, eps), v)
    };

    let mut k_cache = Vec::new();
    let mut v_cache = Vec::new();
    for &token in prompt {
        let (k, v) = project_kv(token);
        k_cache.push(k);
        v_cache.push(v);
    }

    let mut current = *prompt.last().expect("nonempty prompt");
    let mut generated = Vec::new();
    let mut logits_by_step = Vec::new();
    for step in 0..steps {
        let position = prompt.len() - 1 + step;
        let mut residual = token_residual(current);
        let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let q = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &normed),
            &q_norm,
            eps,
        );
        let k = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed),
            &k_norm,
            eps,
        );
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);

        if position < k_cache.len() {
            k_cache[position] = k;
            v_cache[position] = v;
        } else {
            k_cache.push(k);
            v_cache.push(v);
        }

        let mut scores = Vec::with_capacity(k_cache.len());
        for key in &k_cache {
            let score = q
                .iter()
                .zip(key.iter())
                .map(|(qv, kv)| qv * kv)
                .sum::<f32>()
                / (hidden as f32).sqrt();
            scores.push(score);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .sum::<f32>();

        let mut attn_out = vec![0.0f32; hidden];
        for (idx, value) in v_cache.iter().enumerate() {
            let weight = (scores[idx] - max_score).exp() / denom;
            for dim in 0..hidden {
                attn_out[dim] += value[dim] * weight;
            }
        }

        let projected = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out),
            &norm,
            eps,
        );
        for dim in 0..hidden {
            residual[dim] += projected[dim] * layer_scalar[dim];
        }

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated),
            &norm,
            eps,
        );
        for dim in 0..hidden {
            residual[dim] += mlp_out[dim] * layer_scalar[dim];
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        for logit in &mut logits {
            *logit = softcap * (*logit / softcap).tanh();
        }
        let next = cpu_full_nonzero_argmax(&logits);
        logits_by_step.push(logits);
        generated.push(next);
        current = next;
    }

    GeneratedTinyGemma4HfDecodeLoopReference {
        generated,
        logits_by_step,
    }
}

fn cpu_reference_generated_tiny_gemma4_hf_sequence(prompt: &[usize], steps: usize) -> Vec<usize> {
    cpu_reference_generated_tiny_gemma4_hf_decode_loop(prompt, steps).generated
}

fn cpu_reference_generated_tiny_hf_end_to_end_decode_loop(
) -> GeneratedTinyGemma4HfDecodeLoopReference {
    cpu_reference_generated_tiny_gemma4_hf_decode_loop(&[2, 4], 2)
}

fn cpu_reference_generated_tiny_hf_end_to_end_sequence() -> Vec<usize> {
    cpu_reference_generated_tiny_hf_end_to_end_decode_loop().generated
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_full_nonzero_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 1.0;
    lm_head[3 * hidden + FULL_NONZERO_ATTENTION_DIM] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    q_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.25;
    k_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.125;
    v_proj[FULL_NONZERO_VALUE_DIM * hidden + FULL_NONZERO_ORIGINAL_DIM] = 2.0;
    o_proj[FULL_NONZERO_ATTENTION_DIM * hidden + FULL_NONZERO_VALUE_DIM] = 6.0;
    gate_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
    up_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
    down_proj[FULL_NONZERO_FFN_DIM * intermediate] = 4.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn patch_fixture_f16_tensor(
    model_dir: &std::path::Path,
    tensor_name: &str,
    linear_index: usize,
    value: f32,
) {
    let path = model_dir.join("model.safetensors");
    let mut bytes = fs::read(&path).expect("read fixture safetensors");
    let header_len = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("safetensors header length bytes"),
    ) as usize;
    let header: Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("parse safetensors header");
    let tensor = header
        .get(tensor_name)
        .and_then(Value::as_object)
        .expect("fixture tensor metadata");
    assert_eq!(tensor.get("dtype").and_then(Value::as_str), Some("F16"));
    let offsets = tensor
        .get("data_offsets")
        .and_then(Value::as_array)
        .expect("fixture tensor offsets");
    let start = offsets[0].as_u64().expect("fixture tensor start") as usize;
    let end = offsets[1].as_u64().expect("fixture tensor end") as usize;
    let byte_offset = 8 + header_len + start + linear_index * std::mem::size_of::<half::f16>();
    assert!(byte_offset + 2 <= 8 + header_len + end);
    bytes[byte_offset..byte_offset + 2]
        .copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
    fs::write(path, bytes).expect("patch fixture safetensors");
}

#[test]
fn cpu_reference_one_layer_full_nonzero_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_full_nonzero_argmax(), 3);
}

fn assert_f32_close(label: &str, got: f32, expected: f32, tolerance: f32) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tolerance,
        "{label} mismatch: got={got} expected={expected} diff={diff} tol={tolerance}"
    );
}

fn assert_selected_logits_close(
    label: &str,
    got: &[f32],
    expected: &[f32],
    indices: &[usize],
    tolerance: f32,
) {
    assert_eq!(got.len(), expected.len());
    for &idx in indices {
        assert_f32_close(
            &format!("{label} logit[{idx}]"),
            got[idx],
            expected[idx],
            tolerance,
        );
    }
}

fn assert_f32_slice_close(label: &str, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len(), "{label} length mismatch");
    for idx in 0..expected.len() {
        assert_f32_close(
            &format!("{label}[{idx}]"),
            got[idx],
            expected[idx],
            tolerance,
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Clone, Debug)]
struct FiniteSummary {
    finite_count: usize,
    total_count: usize,
    max_abs: f32,
    mean_abs: f32,
    first_nonfinite_index: Option<usize>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn finite_summary(values: &[f32]) -> FiniteSummary {
    let mut finite_count = 0usize;
    let mut abs_sum = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut first_nonfinite_index = None;
    for (idx, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            finite_count += 1;
            let abs = value.abs();
            max_abs = max_abs.max(abs);
            abs_sum += abs as f64;
        } else if first_nonfinite_index.is_none() {
            first_nonfinite_index = Some(idx);
        }
    }
    let mean_abs = if finite_count == 0 {
        0.0
    } else {
        (abs_sum / finite_count as f64) as f32
    };
    FiniteSummary {
        finite_count,
        total_count: values.len(),
        max_abs,
        mean_abs,
        first_nonfinite_index,
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn compare_trace_summary_stats(hf: &Value, metal: &Value, name: &str) {
    let hf_summary = &hf["summaries"][name];
    let metal_summary = &metal["summaries"][name];
    assert!(
        hf_summary.is_object(),
        "HF layer trace missing summary {name}"
    );
    assert!(
        metal_summary.is_object(),
        "Metal layer trace missing summary {name}"
    );
    let hf_total = hf_summary["total_count"].as_u64().expect("HF total_count");
    let metal_total = metal_summary["total_count"]
        .as_u64()
        .expect("Metal total_count");
    let hf_finite = hf_summary["finite_count"]
        .as_u64()
        .expect("HF finite_count");
    let metal_finite = metal_summary["finite_count"]
        .as_u64()
        .expect("Metal finite_count");
    assert_eq!(hf_total, metal_total, "{name} total_count");
    assert_eq!(hf_finite, hf_total, "{name} HF finite_count");
    assert_eq!(metal_finite, metal_total, "{name} Metal finite_count");

    let hf_max = hf_summary["max_abs"].as_f64().expect("HF max_abs");
    let metal_max = metal_summary["max_abs"].as_f64().expect("Metal max_abs");
    let hf_mean = hf_summary["mean_abs"].as_f64().expect("HF mean_abs");
    let metal_mean = metal_summary["mean_abs"].as_f64().expect("Metal mean_abs");
    eprintln!(
        "E2B layer4 trace {name}: hf_max={hf_max:.6e} metal_max={metal_max:.6e} delta_max={:.6e} hf_mean={hf_mean:.6e} metal_mean={metal_mean:.6e} delta_mean={:.6e}",
        (hf_max - metal_max).abs(),
        (hf_mean - metal_mean).abs()
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn parse_e2b_trace_prompt_token_ids() -> Vec<u32> {
    let Some(raw) = std::env::var_os("RVLLM_E2B_TRACE_COMPARE_PROMPT_TOKEN_IDS") else {
        return vec![2, 4];
    };
    let raw = raw.to_string_lossy();
    let tokens = raw
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            (!part.is_empty()).then(|| {
                part.parse::<u32>()
                    .unwrap_or_else(|err| panic!("invalid trace prompt token id {part:?}: {err}"))
            })
        })
        .collect::<Vec<_>>();
    assert!(
        !tokens.is_empty(),
        "RVLLM_E2B_TRACE_COMPARE_PROMPT_TOKEN_IDS must contain at least one token"
    );
    tokens
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn e2b_trace_prompt_slug(prompt_token_ids: &[u32]) -> String {
    prompt_token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(all(feature = "apple", target_os = "macos"))]
const E2B_SHARED_KV_DIFF_TRACE_LAYERS: &[usize] = &[13, 14, 15, 16, 17, 18, 19, 20];

#[cfg(all(feature = "apple", target_os = "macos"))]
const E2B_SHARED_KV_DIFF_TRACE_FIELDS: &[&str] = &[
    "input_to_layer",
    "q_projection",
    "after_q_norm",
    "after_rope_q",
    "local_kv_cache_k",
    "local_kv_cache_v",
    "attention_kv_cache_k",
    "attention_kv_cache_v",
    "attention_output",
    "after_o_proj",
    "after_post_attention_layernorm",
    "after_pre_feedforward_layernorm",
    "gate_up_out",
    "ffn_activation",
    "after_ffn_branch",
    "final_residual_after_layer",
];

#[cfg(all(feature = "apple", target_os = "macos"))]
const E2B_SHARED_KV_DIFF_DOWNSTREAM_FIELDS: &[&str] = &[
    "input_to_layer",
    "q_projection",
    "after_q_norm",
    "after_rope_q",
    "attention_kv_cache_k",
    "attention_kv_cache_v",
    "attention_output",
    "after_o_proj",
    "after_post_attention_layernorm",
    "after_pre_feedforward_layernorm",
    "gate_up_out",
    "ffn_activation",
    "after_ffn_branch",
    "final_residual_after_layer",
];

#[cfg(all(feature = "apple", target_os = "macos"))]
fn e2b_shared_kv_trace_path(mode: &str, layer_idx: usize) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "/tmp/gemma4-e2b-shared-kv-{mode}-layer{layer_idx}.json"
    ))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn read_e2b_shared_kv_trace(mode: &str, layer_idx: usize) -> Value {
    let path = e2b_shared_kv_trace_path(mode, layer_idx);
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "read E2B shared-KV {mode} layer {layer_idx} trace at {}: {err}",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse E2B shared-KV {mode} layer {layer_idx} trace: {err}"))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn trace_number_diff(
    baseline: &Value,
    candidate: &Value,
    key: &str,
    tolerance: f64,
) -> Option<String> {
    let baseline_value = baseline[key].as_f64()?;
    let candidate_value = candidate[key].as_f64()?;
    let delta = (baseline_value - candidate_value).abs();
    (delta > tolerance).then(|| {
        format!(
            "{key} baseline={baseline_value:.9e} candidate={candidate_value:.9e} delta={delta:.9e}"
        )
    })
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn trace_array_diff(
    baseline: &Value,
    candidate: &Value,
    key: &str,
    tolerance: f64,
) -> Option<String> {
    let baseline_values = baseline[key].as_array()?;
    let candidate_values = candidate[key].as_array()?;
    if baseline_values.len() != candidate_values.len() {
        return Some(format!(
            "{key} length baseline={} candidate={}",
            baseline_values.len(),
            candidate_values.len()
        ));
    }
    for (idx, (baseline_value, candidate_value)) in baseline_values
        .iter()
        .zip(candidate_values.iter())
        .enumerate()
    {
        match (baseline_value.as_f64(), candidate_value.as_f64()) {
            (Some(baseline_value), Some(candidate_value)) => {
                let delta = (baseline_value - candidate_value).abs();
                if delta > tolerance {
                    return Some(format!(
                        "{key}[{idx}] baseline={baseline_value:.9e} candidate={candidate_value:.9e} delta={delta:.9e}"
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Some(format!(
                    "{key}[{idx}] baseline={baseline_value:?} candidate={candidate_value:?}"
                ));
            }
        }
    }
    None
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn trace_selected_diff(baseline: &Value, candidate: &Value, tolerance: f64) -> Option<String> {
    let baseline_values = baseline["selected"].as_array()?;
    let candidate_values = candidate["selected"].as_array()?;
    if baseline_values.len() != candidate_values.len() {
        return Some(format!(
            "selected length baseline={} candidate={}",
            baseline_values.len(),
            candidate_values.len()
        ));
    }
    for (position, (baseline_value, candidate_value)) in baseline_values
        .iter()
        .zip(candidate_values.iter())
        .enumerate()
    {
        let baseline_index = baseline_value["index"].as_u64();
        let candidate_index = candidate_value["index"].as_u64();
        if baseline_index != candidate_index {
            return Some(format!(
                "selected[{position}] index baseline={baseline_index:?} candidate={candidate_index:?}"
            ));
        }
        match (
            baseline_value["value"].as_f64(),
            candidate_value["value"].as_f64(),
        ) {
            (Some(baseline_number), Some(candidate_number)) => {
                let delta = (baseline_number - candidate_number).abs();
                if delta > tolerance {
                    return Some(format!(
                        "selected[{position}] index={baseline_index:?} baseline={baseline_number:.9e} candidate={candidate_number:.9e} delta={delta:.9e}"
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Some(format!(
                    "selected[{position}] index={baseline_index:?} baseline={baseline_value:?} candidate={candidate_value:?}"
                ));
            }
        }
    }
    None
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn first_trace_summary_diff(
    baseline: &Value,
    candidate: &Value,
    layer_idx: usize,
    mode: &str,
    fields: &[&str],
) -> Option<String> {
    const COUNT_TOLERANCE: f64 = 0.0;
    const VALUE_TOLERANCE: f64 = 1.0e-3;
    for name in fields {
        let baseline_summary = &baseline["summaries"][name];
        let candidate_summary = &candidate["summaries"][name];
        if !baseline_summary.is_object() || !candidate_summary.is_object() {
            return Some(format!(
                "mode={mode} layer={layer_idx} field={name} missing summary baseline={} candidate={}",
                baseline_summary.is_object(),
                candidate_summary.is_object()
            ));
        }
        for key in ["total_count", "finite_count"] {
            if let Some(diff) =
                trace_number_diff(baseline_summary, candidate_summary, key, COUNT_TOLERANCE)
            {
                return Some(format!("mode={mode} layer={layer_idx} field={name} {diff}"));
            }
        }
        for key in ["max_abs", "mean_abs"] {
            if let Some(diff) =
                trace_number_diff(baseline_summary, candidate_summary, key, VALUE_TOLERANCE)
            {
                return Some(format!("mode={mode} layer={layer_idx} field={name} {diff}"));
            }
        }
        if let Some(diff) = trace_array_diff(
            baseline_summary,
            candidate_summary,
            "first_values",
            VALUE_TOLERANCE,
        ) {
            return Some(format!("mode={mode} layer={layer_idx} field={name} {diff}"));
        }
        if let Some(diff) =
            trace_selected_diff(baseline_summary, candidate_summary, VALUE_TOLERANCE)
        {
            return Some(format!("mode={mode} layer={layer_idx} field={name} {diff}"));
        }
    }
    None
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn assert_trace_summaries_finite(trace: &Value, mode: &str, layer_idx: usize, fields: &[&str]) {
    for name in fields {
        let summary = &trace["summaries"][name];
        assert!(
            summary.is_object(),
            "E2B shared-KV paired trace mode={mode} layer={layer_idx} missing summary {name}"
        );
        let total = summary["total_count"]
            .as_u64()
            .unwrap_or_else(|| panic!("mode={mode} layer={layer_idx} {name} total_count"));
        let finite = summary["finite_count"]
            .as_u64()
            .unwrap_or_else(|| panic!("mode={mode} layer={layer_idx} {name} finite_count"));
        assert_eq!(
            finite, total,
            "E2B shared-KV paired trace mode={mode} layer={layer_idx} {name} has non-finite values"
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_e2b_shared_kv_trace_mode(
    model_dir: &std::path::Path,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    mode: &'static str,
) {
    for layer_idx in E2B_SHARED_KV_DIFF_TRACE_LAYERS {
        let _ = fs::remove_file(e2b_shared_kv_trace_path(mode, *layer_idx));
    }

    let trace_path_template = std::path::PathBuf::from(format!(
        "/tmp/gemma4-e2b-shared-kv-{mode}-layer{{layer}}.json"
    ));
    let env_guard = MetalDebugEnvGuard::new(&[
        RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV,
        RVLLM_METAL_DEBUG_TRACE_LAYER_ENV,
        RVLLM_METAL_DEBUG_TRACE_JSON_ENV,
        RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV,
        RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE_ENV,
    ]);
    env_guard.set(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV, "13,14,15,16,17,18,19,20");
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_JSON_ENV, &trace_path_template);
    env_guard.set(RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE_ENV, mode);

    let mut plan = n_layer_plan(model_dir.to_path_buf(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir.to_path_buf());
    backend.prepare(&plan).unwrap_or_else(|err| {
        panic!("real Gemma4 E2B prepare should complete for mode={mode}: {err}")
    });

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend
        .launch_prefill(&prefill)
        .unwrap_or_else(|err| panic!("real E2B shared-KV paired trace prefill mode={mode}: {err}"));
    let prefill_out = backend.collect(prefill_ticket).unwrap_or_else(|err| {
        panic!("real E2B shared-KV paired trace prefill collect mode={mode}: {err}")
    });
    assert!(prefill_out.is_empty());

    let decode = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(4)],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let decode_ticket = backend
        .launch_rollout(&decode, None)
        .unwrap_or_else(|err| panic!("real E2B shared-KV paired trace decode mode={mode}: {err}"));
    let decode_out = backend.collect(decode_ticket).unwrap_or_else(|err| {
        panic!("real E2B shared-KV paired trace decode collect mode={mode}: {err}")
    });
    assert_eq!(
        decode_out.len(),
        1,
        "skip-final-logits debug path must return one placeholder token for mode={mode}"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Debug)]
struct E2bHfReferenceStep {
    selected_token_ids: Vec<u32>,
    selected_logits: Vec<f32>,
    full_logits: Option<Vec<f32>>,
    next_token: u32,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Debug)]
struct E2bHfReferenceLogits {
    prompt_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    steps: Vec<E2bHfReferenceStep>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn read_e2b_hf_reference_logits(
    path: &std::path::Path,
    expected_prompt_token_ids: &[u32],
    expected_decode_steps: u64,
) -> E2bHfReferenceLogits {
    let raw = fs::read_to_string(path).expect("read HF reference logits artifact");
    let value: Value = serde_json::from_str(&raw).expect("parse HF reference logits artifact");

    assert_eq!(
        value["schema"].as_str(),
        Some("rvllm.gemma4_hf_reference_logits.v1")
    );
    let prompt_token_ids = value["prompt_token_ids"]
        .as_array()
        .expect("prompt ids")
        .iter()
        .map(|item| item.as_u64().expect("prompt token id") as u32)
        .collect::<Vec<_>>();
    assert_eq!(prompt_token_ids.as_slice(), expected_prompt_token_ids);
    assert_eq!(value["decode_steps"].as_u64(), Some(expected_decode_steps));

    let generated_tokens = value["generated_tokens"]
        .as_array()
        .expect("generated tokens")
        .iter()
        .map(|item| item.as_u64().expect("generated token id") as u32)
        .collect::<Vec<_>>();
    let steps = value["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .enumerate()
        .map(|(step_idx, step)| {
            assert_eq!(step["step"].as_u64(), Some(step_idx as u64));
            let next_token = step["next_token"].as_u64().expect("next token") as u32;
            let selected = step["selected_logits"].as_array().expect("selected logits");
            let mut token_ids = Vec::with_capacity(selected.len());
            let mut logits = Vec::with_capacity(selected.len());
            for item in selected {
                token_ids.push(item["token_id"].as_u64().expect("selected token id") as u32);
                logits.push(item["logit"].as_f64().expect("selected logit") as f32);
            }
            let full_logits = step.get("logits").and_then(|logits_value| {
                logits_value.as_array().map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_f64().expect("full logit") as f32)
                        .collect::<Vec<_>>()
                })
            });
            E2bHfReferenceStep {
                selected_token_ids: token_ids,
                selected_logits: logits,
                full_logits,
                next_token,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(steps.len(), expected_decode_steps as usize);
    assert_eq!(generated_tokens.len(), steps.len());
    for (step, &generated) in steps.iter().zip(generated_tokens.iter()) {
        assert_eq!(step.next_token, generated);
    }
    E2bHfReferenceLogits {
        prompt_token_ids,
        generated_tokens,
        steps,
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn read_e2b_hf_reference_selected_logits(path: &std::path::Path) -> (Vec<u32>, Vec<f32>, u32) {
    let reference = read_e2b_hf_reference_logits(path, &[2, 4], 1);
    let step = &reference.steps[0];
    (
        step.selected_token_ids.clone(),
        step.selected_logits.clone(),
        step.next_token,
    )
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn assert_e2b_full_vocab_logits_close(
    label: &str,
    metal_logits: &[f32],
    hf_logits: &[f32],
    tolerance: f32,
) {
    assert_eq!(
        metal_logits.len(),
        hf_logits.len(),
        "{label} length mismatch"
    );

    let mut max_delta = 0.0f32;
    let mut max_delta_token = 0usize;
    let mut sum_delta = 0.0f64;
    let mut deltas = Vec::with_capacity(hf_logits.len());
    for (idx, (&metal, &hf)) in metal_logits.iter().zip(hf_logits.iter()).enumerate() {
        assert!(
            metal.is_finite(),
            "{label} metal logit[{idx}] is not finite"
        );
        assert!(hf.is_finite(), "{label} HF logit[{idx}] is not finite");
        let delta = (metal - hf).abs();
        if delta > max_delta {
            max_delta = delta;
            max_delta_token = idx;
        }
        sum_delta += f64::from(delta);
        deltas.push(delta);
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).expect("finite delta"));
    let p99_idx = ((deltas.len() - 1) * 99) / 100;
    let p999_idx = ((deltas.len() - 1) * 999) / 1000;
    let mean_delta = sum_delta / hf_logits.len() as f64;
    eprintln!(
        "{label}: vocab={} max_delta={} max_delta_token={} mean_delta={:.6e} p99_delta={} p999_delta={}",
        hf_logits.len(),
        max_delta,
        max_delta_token,
        mean_delta,
        deltas[p99_idx],
        deltas[p999_idx]
    );
    assert!(
        max_delta <= tolerance,
        "{label} max delta exceeds tolerance: max_delta={max_delta} token={max_delta_token} tol={tolerance}"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn assert_e2b_sample_matches_or_hf_tie(
    label: &str,
    hf_logits: &[f32],
    expected_token: u32,
    sampled_token: u32,
) {
    if sampled_token == expected_token {
        return;
    }
    let expected_idx = expected_token as usize;
    let sampled_idx = sampled_token as usize;
    let expected_logit = hf_logits[expected_idx];
    let sampled_logit = hf_logits[sampled_idx];
    eprintln!(
        "{label}: sampled token {sampled_token} differs from HF token {expected_token}; HF logits sampled={sampled_logit} expected={expected_logit}"
    );
    assert_f32_close(
        &format!("{label} HF tied sampled token"),
        sampled_logit,
        expected_logit,
        0.0001,
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_real_e2b_model_backend_decode_loop(
    model_dir: std::path::PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    prompt_token_ids: &[u32],
    decode_steps: usize,
) -> Result<(Vec<Vec<f32>>, Vec<rvllm_apple::StepToken>)> {
    run_real_e2b_model_backend_decode_loop_with_forced_next_tokens(
        model_dir,
        arch,
        prompt_token_ids,
        decode_steps,
        None,
    )
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_real_e2b_model_backend_decode_loop_with_forced_next_tokens(
    model_dir: std::path::PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    prompt_token_ids: &[u32],
    decode_steps: usize,
    forced_next_tokens: Option<&[u32]>,
) -> Result<(Vec<Vec<f32>>, Vec<rvllm_apple::StepToken>)> {
    assert!(!prompt_token_ids.is_empty());
    assert!(
        prompt_token_ids.len() + decode_steps <= 16,
        "real E2B probe arena currently supports prompt_len + decode_steps <= 16"
    );
    if let Some(tokens) = forced_next_tokens {
        assert!(
            tokens.len() >= decode_steps,
            "forced next-token list must cover every decode step"
        );
    }

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<rvllm_apple::StepToken>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let mut backend = ModelMetalBackend::new(model_dir);
        backend.prepare(&plan)?;

        let prompt_tokens = prompt_token_ids
            .iter()
            .map(|&token| rvllm_core::TokenId(token))
            .collect::<Vec<_>>();
        let prompt_len = prompt_tokens.len();
        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            prompt_tokens.clone(),
            vec![0, prompt_len as u32],
            vec![(prompt_len - 1) as u32],
            vec![prompt_len as u32],
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());

        let mut current = *prompt_tokens.last().expect("prompt token");
        let mut logits_by_step = Vec::with_capacity(decode_steps);
        let mut generated = Vec::with_capacity(decode_steps);
        for step_idx in 0..decode_steps {
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(1)],
                vec![current],
                vec![0, 1],
                vec![(prompt_len - 1 + step_idx) as u32],
                vec![(prompt_len + step_idx) as u32],
            );
            let decode_ticket = backend.launch_rollout(&decode, None)?;
            let logits = backend.debug_read_decode_logits_f32(1)?;
            let residual = backend.debug_read_residual_f32(1)?;
            assert_eq!(logits.len(), arch.vocab_size);
            assert_eq!(residual.len(), arch.hidden_size);
            assert!(logits.iter().all(|v| v.is_finite()));
            assert!(residual.iter().all(|v| v.is_finite()));
            let out = backend.collect(decode_ticket)?;
            assert_eq!(out.len(), 1);
            current = forced_next_tokens
                .and_then(|tokens| tokens.get(step_idx).copied())
                .map(rvllm_core::TokenId)
                .unwrap_or(out[0].token_id);
            logits_by_step.push(logits);
            generated.push(out[0].clone());
        }
        Ok((logits_by_step, generated))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    result
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_real_e2b_model_backend_batch_one_step(
    model_dir: std::path::PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    prompts: &[Vec<u32>],
) -> Result<(Vec<Vec<f32>>, Vec<rvllm_apple::StepToken>)> {
    assert!(!prompts.is_empty());
    assert!(
        prompts
            .iter()
            .all(|prompt| !prompt.is_empty() && prompt.len() + 1 <= 16),
        "real E2B probe arena currently supports each batch prompt_len + decode_steps <= 16"
    );

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<rvllm_apple::StepToken>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let mut backend = ModelMetalBackend::new(model_dir);
        backend.prepare(&plan)?;

        let req_ids = (0..prompts.len())
            .map(|idx| rvllm_core::ReqId((idx + 1) as u64))
            .collect::<Vec<_>>();
        let mut prefill_tokens = Vec::new();
        let mut cu_seqlens = Vec::with_capacity(prompts.len() + 1);
        let mut positions = Vec::with_capacity(prompts.len());
        let mut context_lens = Vec::with_capacity(prompts.len());
        cu_seqlens.push(0);
        for prompt in prompts {
            prefill_tokens.extend(prompt.iter().map(|&token| rvllm_core::TokenId(token)));
            positions.push((prompt.len() - 1) as u32);
            context_lens.push(prompt.len() as u32);
            cu_seqlens.push(prefill_tokens.len() as u32);
        }

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            req_ids.clone(),
            prefill_tokens,
            cu_seqlens,
            positions.clone(),
            context_lens.clone(),
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());

        let decode_tokens = prompts
            .iter()
            .map(|prompt| rvllm_core::TokenId(*prompt.last().expect("prompt token")))
            .collect::<Vec<_>>();
        let decode_cu_seqlens = (0..=prompts.len())
            .map(|idx| idx as u32)
            .collect::<Vec<_>>();
        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            req_ids,
            decode_tokens,
            decode_cu_seqlens,
            positions,
            context_lens,
        );
        let decode_ticket = backend.launch_rollout(&decode, None)?;
        let flat_logits = backend.debug_read_decode_logits_f32(prompts.len())?;
        let residual = backend.debug_read_residual_f32(prompts.len())?;
        assert_eq!(flat_logits.len(), prompts.len() * arch.vocab_size);
        assert_eq!(residual.len(), prompts.len() * arch.hidden_size);
        assert!(flat_logits.iter().all(|v| v.is_finite()));
        assert!(residual.iter().all(|v| v.is_finite()));
        let out = backend.collect(decode_ticket)?;
        assert_eq!(out.len(), prompts.len());
        let logits_by_seq = flat_logits
            .chunks_exact(arch.vocab_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        Ok((logits_by_seq, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    result
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_real_e2b_model_backend_batch_decode_loop_with_forced_next_tokens(
    model_dir: std::path::PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    prompts: &[Vec<u32>],
    decode_steps: usize,
    forced_next_tokens: &[Vec<u32>],
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<rvllm_apple::StepToken>>)> {
    assert!(!prompts.is_empty());
    assert_eq!(forced_next_tokens.len(), prompts.len());
    assert!(forced_next_tokens
        .iter()
        .all(|tokens| tokens.len() >= decode_steps));
    assert!(
        prompts
            .iter()
            .all(|prompt| !prompt.is_empty() && prompt.len() + decode_steps <= 16),
        "real E2B probe arena currently supports each batch prompt_len + decode_steps <= 16"
    );

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<rvllm_apple::StepToken>>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let mut backend = ModelMetalBackend::new(model_dir);
        backend.prepare(&plan)?;

        let req_ids = (0..prompts.len())
            .map(|idx| rvllm_core::ReqId((idx + 1) as u64))
            .collect::<Vec<_>>();
        let mut prefill_tokens = Vec::new();
        let mut prefill_cu_seqlens = Vec::with_capacity(prompts.len() + 1);
        prefill_cu_seqlens.push(0);
        for prompt in prompts {
            prefill_tokens.extend(prompt.iter().map(|&token| rvllm_core::TokenId(token)));
            prefill_cu_seqlens.push(prefill_tokens.len() as u32);
        }
        let prefill_positions = prompts
            .iter()
            .map(|prompt| (prompt.len() - 1) as u32)
            .collect::<Vec<_>>();
        let prefill_context_lens = prompts
            .iter()
            .map(|prompt| prompt.len() as u32)
            .collect::<Vec<_>>();

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            req_ids.clone(),
            prefill_tokens,
            prefill_cu_seqlens,
            prefill_positions,
            prefill_context_lens,
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());

        let decode_cu_seqlens = (0..=prompts.len())
            .map(|idx| idx as u32)
            .collect::<Vec<_>>();
        let mut current_tokens = prompts
            .iter()
            .map(|prompt| rvllm_core::TokenId(*prompt.last().expect("prompt token")))
            .collect::<Vec<_>>();
        let mut logits_by_step = Vec::with_capacity(decode_steps);
        let mut outputs_by_step = Vec::with_capacity(decode_steps);
        for step_idx in 0..decode_steps {
            let positions = prompts
                .iter()
                .map(|prompt| (prompt.len() - 1 + step_idx) as u32)
                .collect::<Vec<_>>();
            let context_lens = prompts
                .iter()
                .map(|prompt| (prompt.len() + step_idx) as u32)
                .collect::<Vec<_>>();
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                req_ids.clone(),
                current_tokens.clone(),
                decode_cu_seqlens.clone(),
                positions,
                context_lens,
            );
            let decode_ticket = backend.launch_rollout(&decode, None)?;
            let flat_logits = backend.debug_read_decode_logits_f32(prompts.len())?;
            let residual = backend.debug_read_residual_f32(prompts.len())?;
            assert_eq!(flat_logits.len(), prompts.len() * arch.vocab_size);
            assert_eq!(residual.len(), prompts.len() * arch.hidden_size);
            assert!(flat_logits.iter().all(|v| v.is_finite()));
            assert!(residual.iter().all(|v| v.is_finite()));
            let out = backend.collect(decode_ticket)?;
            assert_eq!(out.len(), prompts.len());
            current_tokens = forced_next_tokens
                .iter()
                .map(|tokens| rvllm_core::TokenId(tokens[step_idx]))
                .collect();
            logits_by_step.push(
                flat_logits
                    .chunks_exact(arch.vocab_size)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            );
            outputs_by_step.push(out);
        }
        Ok((logits_by_step, outputs_by_step))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    result
}

#[test]
fn cpu_reference_one_layer_full_nonzero_selected_hidden_values_are_expected() {
    let reference = cpu_reference_one_layer_full_nonzero();

    assert_eq!(reference.residual_after_attention.len(), 128);
    assert_eq!(reference.residual.len(), 128);
    assert_eq!(reference.final_hidden.len(), 128);
    assert_f32_close(
        "residual zero dim",
        reference.residual[FULL_NONZERO_ZERO_DIM],
        0.0,
        0.0001,
    );
    assert_f32_close(
        "hidden zero dim",
        reference.final_hidden[FULL_NONZERO_ZERO_DIM],
        0.0,
        0.0001,
    );
    assert_f32_close(
        "residual original dim",
        reference.residual[FULL_NONZERO_ORIGINAL_DIM],
        113.137_085,
        0.0001,
    );
    assert_f32_close(
        "hidden original dim",
        reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM],
        6.943_472_4,
        0.0001,
    );
    assert_f32_close(
        "residual attention dim",
        reference.residual[FULL_NONZERO_ATTENTION_DIM],
        135.764_5,
        0.0001,
    );
    assert_f32_close(
        "hidden attention dim",
        reference.final_hidden[FULL_NONZERO_ATTENTION_DIM],
        8.332_167,
        0.0001,
    );
    assert_f32_close(
        "residual ffn pre-update dim",
        reference.residual_after_attention[FULL_NONZERO_FFN_DIM],
        0.0,
        0.0001,
    );
    assert_f32_close(
        "residual ffn dim",
        reference.residual[FULL_NONZERO_FFN_DIM],
        52.453_545,
        0.0001,
    );
    assert_f32_close(
        "hidden ffn dim",
        reference.final_hidden[FULL_NONZERO_FFN_DIM],
        3.219_189_6,
        0.0001,
    );
    assert!(
        reference.final_hidden[FULL_NONZERO_ATTENTION_DIM]
            > reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM]
    );
    assert!(
        reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM]
            > reference.final_hidden[FULL_NONZERO_FFN_DIM]
    );
    assert!(
        reference.final_hidden[FULL_NONZERO_FFN_DIM]
            > reference.final_hidden[FULL_NONZERO_ZERO_DIM]
    );
}

#[test]
fn cpu_reference_one_layer_full_nonzero_residual_vector_is_stable() {
    let reference = cpu_reference_one_layer_full_nonzero();
    let mut expected = vec![0.0f32; 128];
    expected[FULL_NONZERO_ORIGINAL_DIM] = 113.137_085;
    expected[FULL_NONZERO_ATTENTION_DIM] = 135.764_5;
    expected[FULL_NONZERO_FFN_DIM] = 52.453_545;

    assert_f32_slice_close(
        "full nonzero residual",
        &reference.residual,
        &expected,
        0.0001,
    );
}

#[test]
fn cpu_reference_one_layer_full_nonzero_selected_logits_pick_token_3() {
    let logits = cpu_reference_one_layer_full_nonzero_logits();
    assert_eq!(logits.len(), 8);
    assert_eq!(cpu_full_nonzero_argmax(&logits), 3);
    assert_eq!(cpu_full_nonzero_top_two(&logits), (3, 2));
    assert_eq!(logits[0], 0.0);
    assert!(logits[3] > logits[2]);
    assert!(logits[2] > logits[0]);
}

#[test]
fn cpu_reference_real_hf_style_one_layer_slice_argmax_is_3() {
    assert_eq!(cpu_reference_real_hf_style_one_layer_slice_argmax(), 3);
}

#[test]
fn cpu_reference_one_layer_qkv_norm_nonzero_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_qkv_norm_nonzero_argmax(false), 2);
    assert_eq!(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true), 3);
}

#[test]
fn cpu_reference_one_layer_extra_norms_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_extra_norms_argmax(false), 2);
    assert_eq!(cpu_reference_one_layer_extra_norms_argmax(true), 3);
}

#[test]
fn cpu_reference_one_layer_layer_scalar_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_layer_scalar_argmax(false), 2);
    assert_eq!(cpu_reference_one_layer_layer_scalar_argmax(true), 3);
}

#[test]
fn cpu_reference_one_layer_integrated_gemma_probe_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_one_layer_integrated_gemma_probe_argmax(), 3);
}

#[test]
fn cpu_reference_prompt_len_two_prefill_fixture_argmax_is_3() {
    assert_eq!(cpu_reference_prompt_len_two_prefill_argmax(false), 2);
    assert_eq!(cpu_reference_prompt_len_two_prefill_argmax(true), 3);
}

#[test]
fn cpu_reference_prompt_len_two_prefill_selected_logits_pick_token_3() {
    let without_first_logits = cpu_reference_prompt_len_two_prefill_logits(false);
    let include_first_logits = cpu_reference_prompt_len_two_prefill_logits(true);
    let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&include_first_logits);
    let low_idx = 0usize;

    assert_eq!(without_first_logits.len(), 8);
    assert_eq!(include_first_logits.len(), 8);
    assert_eq!(cpu_full_nonzero_argmax(&without_first_logits), 2);
    assert_eq!(expected_idx, 3);
    assert_eq!(runner_up_idx, 2);
    assert_eq!(cpu_full_nonzero_argmax(&include_first_logits), expected_idx);
    assert_eq!(include_first_logits[low_idx], 0.0);
    assert!(include_first_logits[expected_idx] > include_first_logits[runner_up_idx]);
    assert!(include_first_logits[runner_up_idx] > include_first_logits[low_idx]);
    assert!(without_first_logits[2] > without_first_logits[3]);
    assert!(include_first_logits[3] > without_first_logits[3]);
}

#[test]
fn cpu_reference_generated_tiny_hf_end_to_end_sequence_is_3_5() {
    assert_eq!(
        cpu_reference_generated_tiny_hf_end_to_end_sequence(),
        vec![3, 5]
    );
}

#[test]
fn cpu_reference_generated_tiny_hf_full_logits_are_stable() {
    let reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
    assert_eq!(reference.generated, vec![3, 5]);
    assert_eq!(reference.logits_by_step.len(), 2);

    let expected_logits = [
        [0.0f32, 0.0, 0.281_43, 24.286_85, 0.0, 0.0, 0.0, 0.0],
        [0.0f32, 0.0, 0.281_43, 0.0, 0.0, 24.286_85, 0.0, 0.0],
    ];

    for (step_idx, expected) in expected_logits.iter().enumerate() {
        let logits = &reference.logits_by_step[step_idx];
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(logits);
        let low_idx = 0usize;

        assert_eq!(logits.len(), 8);
        assert_eq!(expected_idx, reference.generated[step_idx]);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(logits[low_idx], 0.0);
        assert!(logits[expected_idx] > logits[runner_up_idx]);
        assert!(logits[runner_up_idx] > logits[low_idx]);

        assert_f32_slice_close(
            &format!("decode step {} logits", step_idx + 1),
            logits,
            expected,
            0.05,
        );
    }

    let max_step_diff = reference.logits_by_step[0]
        .iter()
        .zip(reference.logits_by_step[1].iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_step_diff > 1.0,
        "decode steps should produce different logits, max_diff={max_step_diff}"
    );

    let cold_token_three = cpu_reference_generated_tiny_gemma4_hf_decode_loop(&[3], 1);
    let max_context_diff = reference.logits_by_step[1]
        .iter()
        .zip(cold_token_three.logits_by_step[0].iter())
        .map(|(persistent, cold)| (persistent - cold).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_context_diff > 0.05,
        "second decode step should depend on retained KV/context, max_diff={max_context_diff}"
    );
}

#[test]
fn generated_tiny_hf_reference_bundle_can_be_exported() {
    let Some(bundle_dir) = std::env::var_os("RVLLM_GENERATED_TINY_HF_REFERENCE_DIR") else {
        eprintln!("skipping: RVLLM_GENERATED_TINY_HF_REFERENCE_DIR is not set");
        return;
    };
    let bundle_dir = std::path::PathBuf::from(bundle_dir);
    fs::create_dir_all(&bundle_dir).expect("create reference bundle dir");

    let fixture_dir = write_generated_tiny_hf_end_to_end_fixture();
    let reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
    assert_eq!(reference.generated, vec![3, 5]);
    assert_eq!(reference.logits_by_step.len(), 2);

    fs::copy(
        fixture_dir.join("config.json"),
        bundle_dir.join("config.json"),
    )
    .expect("copy generated tiny config");
    fs::copy(
        fixture_dir.join("model.safetensors"),
        bundle_dir.join("model.safetensors"),
    )
    .expect("copy generated tiny safetensors");

    let manifest = serde_json::json!({
        "fixture": "generated_tiny_gemma4_hf",
        "evidence_class": "GENERATED-HF-NUMERIC",
        "model_scope": "generated HF/Gemma-shaped synthetic fixture; not a real checkpoint",
        "prompt_tokens": [2, 4],
        "decode_steps": 2,
        "generated_tokens": reference.generated,
        "logits_by_step": reference.logits_by_step,
        "files": {
            "config": "config.json",
            "weights": "model.safetensors"
        },
        "comparison_note": "External reference code should load the exported safetensors/config, run the same prompt [2, 4] for two decode steps, and compare full logits by step."
    });
    fs::write(
        bundle_dir.join("expected_reference.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize reference manifest"),
    )
    .expect("write reference manifest");

    eprintln!(
        "exported generated tiny HF reference bundle to {}",
        bundle_dir.display()
    );

    let _ = fs::remove_dir_all(fixture_dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_real_hf_style_one_layer_slice_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 2.0;
    o_proj[9 * hidden + 11] = 6.0;
    gate_proj[7] = 0.5;
    up_proj[7] = 0.5;
    down_proj[9 * intermediate] = 4.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_extra_norms_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut post_attn_norm = vec![1.0f32; hidden];
    let pre_ff_norm = vec![1.0f32; hidden];
    let post_ff_norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    post_attn_norm[7] = 0.01;
    post_attn_norm[9] = 64.0;
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let gate_up = vec![0.0f32; 2 * intermediate * hidden];
    let down_proj = vec![0.0f32; hidden * intermediate];

    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.25;
    o_proj[9 * hidden + 11] = 0.5;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        &post_attn_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.pre_feedforward_layernorm.weight",
        &pre_ff_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.post_feedforward_layernorm.weight",
        &post_ff_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_up.weight",
        &gate_up,
        &[2 * intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_layer_scalar_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut layer_scalar = vec![1.0f32; hidden];
    layer_scalar[9] = 6.0;
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 1.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let zeros_qkvo = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];

    gate_proj[7] = 0.5;
    up_proj[7] = 0.5;
    down_proj[9 * intermediate] = 1.0;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &zeros_qkvo,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.layer_scalar",
        &layer_scalar,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_integrated_gemma_probe_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let input_norm = vec![1.0f32; hidden];
    let mut q_norm = vec![1.0f32; hidden];
    let mut k_norm = vec![1.0f32; hidden];
    let mut v_norm = vec![1.0f32; hidden];
    let mut post_attn_norm = vec![1.0f32; hidden];
    let mut pre_ff_norm = vec![1.0f32; hidden];
    let mut post_ff_norm = vec![1.0f32; hidden];
    let final_norm = vec![1.0f32; hidden];
    q_norm[0] = 0.75;
    k_norm[0] = 0.5;
    v_norm[11] = 1.25;
    post_attn_norm[9] = 4.0;
    pre_ff_norm[9] = 1.0;
    post_ff_norm[9] = 2.0;

    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.5;
    o_proj[9 * hidden + 11] = 0.2;

    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];
    gate_proj[9] = 0.75;
    up_proj[9] = 0.75;
    down_proj[9 * intermediate] = 1.0;

    let mut layer_scalar = vec![1.0f32; hidden];
    layer_scalar[9] = 3.0;
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 1.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &final_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &input_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_norm.weight",
        &q_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_norm.weight",
        &k_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_norm.weight",
        &v_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &pre_ff_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        &post_attn_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.pre_feedforward_layernorm.weight",
        &pre_ff_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.post_feedforward_layernorm.weight",
        &post_ff_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.layer_scalar",
        &layer_scalar,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 6.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_prompt_len_two_prefill_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    embedding[4 * hidden + 5] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let gate_proj = vec![0.0f32; intermediate * hidden];
    let up_proj = vec![0.0f32; intermediate * hidden];
    let down_proj = vec![0.0f32; hidden * intermediate];
    let mut lm_head = vec![0.0f32; vocab * hidden];

    q_proj[5] = 1.0;
    k_proj[7] = 1.0;
    k_proj[5] = -1.0;
    v_proj[11 * hidden + 7] = 2.0;
    o_proj[9 * hidden + 11] = 2.0;
    lm_head[2 * hidden + 5] = 1.0;
    lm_head[3 * hidden + 9] = 4.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn write_tiny_one_layer_qkv_norm_nonzero_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;

    let norm = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 7] = 1.0;
    lm_head[3 * hidden + 9] = 32.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let mut add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        "model.embed_tokens.weight",
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.norm.weight",
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "lm_head.weight",
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let ones = vec![1.0f32; hidden];
    let mut q_norm = vec![1.0f32; hidden];
    let mut k_norm = vec![1.0f32; hidden];
    let v_norm = vec![1.0f32; hidden];
    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let gate_up = vec![0.0f32; 2 * intermediate * hidden];
    let down_proj = vec![0.0f32; hidden * intermediate];

    q_norm[0] = 0.5;
    k_norm[0] = 0.25;
    q_proj[7] = 0.25;
    k_proj[7] = 0.125;
    v_proj[11 * hidden + 7] = 0.25;
    o_proj[9 * hidden + 11] = 0.5;

    add_tensor(
        "model.layers.0.input_layernorm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.q_norm.weight",
        &q_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.k_norm.weight",
        &k_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.v_norm.weight",
        &v_norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp_norm.weight",
        &ones,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.gate_up.weight",
        &gate_up,
        &[2 * intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
        hidden, intermediate, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

fn write_generated_tiny_hf_end_to_end_fixture() -> std::path::PathBuf {
    let dir = temp_fixture_dir();
    let hidden = 128;
    let intermediate = 256;
    let vocab = 8;
    let prefix = "model.language_model";

    let mut embedding = vec![0.0f32; vocab * hidden];
    embedding[2 * hidden + 7] = 10.0;
    embedding[3 * hidden + 6] = 10.0;
    embedding[4 * hidden + 5] = 10.0;

    let norm = vec![1.0f32; hidden];
    let layer_scalar = vec![1.0f32; hidden];
    let mut lm_head = vec![0.0f32; vocab * hidden];
    lm_head[2 * hidden + 10] = 0.25;
    lm_head[3 * hidden + 5] = 3.0;
    lm_head[5 * hidden + 6] = 3.0;

    let mut header = Map::<String, Value>::new();
    let mut payload = Vec::new();

    let add_tensor = |name: &str,
                      data: &[f32],
                      shape: &[usize],
                      payload: &mut Vec<u8>,
                      header: &mut Map<String, Value>| {
        let start = payload.len();
        let bytes = f16_bytes(data);
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    };

    add_tensor(
        &format!("{prefix}.embed_tokens.weight"),
        &embedding,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.norm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.lm_head.weight"),
        &lm_head,
        &[vocab, hidden],
        &mut payload,
        &mut header,
    );

    let mut q_proj = vec![0.0f32; hidden * hidden];
    let mut k_proj = vec![0.0f32; hidden * hidden];
    let mut v_proj = vec![0.0f32; hidden * hidden];
    let mut o_proj = vec![0.0f32; hidden * hidden];
    let mut gate_proj = vec![0.0f32; intermediate * hidden];
    let mut up_proj = vec![0.0f32; intermediate * hidden];
    let mut down_proj = vec![0.0f32; hidden * intermediate];

    q_proj[5] = 0.5;
    k_proj[5] = 1.0;
    v_proj[9 * hidden + 5] = 1.0;
    o_proj[10 * hidden + 9] = 1.0;
    gate_proj[7] = 0.25;
    up_proj[7] = 0.25;
    down_proj[10 * intermediate] = 0.5;

    add_tensor(
        &format!("{prefix}.layers.0.input_layernorm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.q_proj.weight"),
        &q_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.k_proj.weight"),
        &k_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.v_proj.weight"),
        &v_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.o_proj.weight"),
        &o_proj,
        &[hidden, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.q_norm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.self_attn.k_norm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.post_attention_layernorm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.pre_feedforward_layernorm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.post_feedforward_layernorm.weight"),
        &norm,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.layer_scalar"),
        &layer_scalar,
        &[hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.mlp.gate_proj.weight"),
        &gate_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.mlp.up_proj.weight"),
        &up_proj,
        &[intermediate, hidden],
        &mut payload,
        &mut header,
    );
    add_tensor(
        &format!("{prefix}.layers.0.mlp.down_proj.weight"),
        &down_proj,
        &[hidden, intermediate],
        &mut payload,
        &mut header,
    );

    let config = format!(
        r#"{{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "head_dim": {},
    "global_head_dim": {},
    "layer_types": ["full_attention"],
    "vocab_size": {},
    "max_position_embeddings": 16,
    "sliding_window": 8,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": false,
    "attention_k_eq_v": false,
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0}}
    }}
  }}
}}"#,
        hidden, intermediate, hidden, hidden, vocab
    );

    fs::write(dir.join("config.json"), config).expect("write config");

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut out = File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
    out.write_all(&(header_json.len() as u64).to_le_bytes())
        .expect("write header len");
    out.write_all(header_json.as_bytes())
        .expect("write header bytes");
    out.write_all(&payload).expect("write payload");
    dir
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_hf_style_noop_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_one_layer_hf_style_noop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer hf-style tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_hf_style_noop_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_one_layer_hf_style_noop_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny hf-style one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_real_hf_style_one_layer_slice_model_backend_decodes_cpu_expected_token() {
    let expected = rvllm_core::TokenId(cpu_reference_real_hf_style_one_layer_slice_argmax() as u32);
    let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare real-hf-style one-layer slice");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_real_hf_style_one_layer_slice_prefill_then_decode_cpu_expected_token() {
    let expected = rvllm_core::TokenId(cpu_reference_real_hf_style_one_layer_slice_argmax() as u32);
    let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with real-hf-style one-layer slice plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_extra_norms_model_backend_decodes_token_2_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_one_layer_extra_norms_argmax(true) as u32);
    let dir = write_tiny_one_layer_extra_norms_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer extra-norms tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_extra_norms_prefill_then_decode_token_2_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_one_layer_extra_norms_argmax(true) as u32);
    let dir = write_tiny_one_layer_extra_norms_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny extra-norms one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_layer_scalar_model_backend_decodes_token_2_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_one_layer_layer_scalar_argmax(true) as u32);
    let dir = write_tiny_one_layer_layer_scalar_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer layer-scalar tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_layer_scalar_prefill_then_decode_token_2_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_one_layer_layer_scalar_argmax(true) as u32);
    let dir = write_tiny_one_layer_layer_scalar_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny layer-scalar one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_integrated_gemma_probe_model_backend_decodes_token_2_to_3() {
    let expected =
        rvllm_core::TokenId(cpu_reference_one_layer_integrated_gemma_probe_argmax() as u32);
    let dir = write_tiny_one_layer_integrated_gemma_probe_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer integrated Gemma probe tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cpu_reference_zero_layer_decode_loop_sequence_is_3_4_5() {
    assert_eq!(
        cpu_reference_zero_layer_decode_loop_sequence(),
        vec![3, 4, 5]
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_zero_layer_decode_once(
    dir: &std::path::Path,
) -> (Vec<StepToken>, MetalProbePerfStats, bool) {
    let mut backend = ModelMetalBackend::new(dir.to_path_buf());
    let plan = zero_layer_plan(dir.to_path_buf());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");
    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect rollout");
    let stats = backend.probe_perf_stats();
    (out, stats, backend.metal_debug_sync_enabled())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn metal_debug_sync_env_preserves_zero_layer_decode_output() {
    let env_guard = MetalDebugSyncEnvGuard::new();
    let dir = write_tiny_zero_layer_decode_loop_fixture();

    env_guard.set_current(None);
    let (normal_out, normal_stats, normal_debug_sync) = run_zero_layer_decode_once(&dir);

    env_guard.set_current(Some("1"));
    let (debug_out, debug_stats, debug_debug_sync) = run_zero_layer_decode_once(&dir);

    assert!(!normal_debug_sync);
    assert!(debug_debug_sync);
    assert_eq!(normal_out, debug_out);
    assert_eq!(debug_out[0].token_id, rvllm_core::TokenId(3));
    assert!(normal_stats.command_buffers > 0);
    assert!(normal_stats.encoders > 0);
    assert!(normal_stats.forced_waits > 0);
    assert!(debug_stats.forced_waits > normal_stats.forced_waits);
    assert_eq!(debug_stats.decode_steps, 1);
    assert_eq!(debug_stats.last_step_tokens, 1);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn metal_probe_perf_counters_are_populated_and_monotonic() {
    let env_guard = MetalDebugSyncEnvGuard::new();
    env_guard.set_current(None);
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");

    let first = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let first_ticket = backend
        .launch_rollout(&first, None)
        .expect("run first rollout");
    let first_out = backend
        .collect(first_ticket)
        .expect("collect first rollout");
    assert_eq!(first_out[0].token_id, rvllm_core::TokenId(3));
    let first_stats = backend.probe_perf_stats();
    assert_eq!(first_stats.decode_steps, 1);
    assert_eq!(first_stats.tokens, 1);
    assert_eq!(first_stats.last_step_tokens, 1);
    assert!(first_stats.command_buffers > 0);
    assert!(first_stats.encoders > 0);
    assert!(first_stats.forced_waits > 0);
    assert!(first_stats.last_step_command_buffers > 0);
    assert!(first_stats.last_step_encoders > 0);
    assert!(first_stats.last_step_forced_waits > 0);
    assert!(first_stats.last_step_cpu_wall_ns > 0);

    let second = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![first_out[0].token_id],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let second_ticket = backend
        .launch_rollout(&second, None)
        .expect("run second rollout");
    let second_out = backend
        .collect(second_ticket)
        .expect("collect second rollout");
    assert_eq!(second_out[0].token_id, rvllm_core::TokenId(4));
    let second_stats = backend.probe_perf_stats();
    assert_eq!(second_stats.decode_steps, 2);
    assert_eq!(second_stats.tokens, 2);
    assert!(second_stats.command_buffers > first_stats.command_buffers);
    assert!(second_stats.encoders > first_stats.encoders);
    assert!(second_stats.forced_waits > first_stats.forced_waits);
    assert!(second_stats.cpu_wall_ns >= first_stats.cpu_wall_ns);
    assert_eq!(second_stats.last_step_tokens, 1);
    assert!(second_stats.last_step_command_buffers > 0);
    assert!(second_stats.last_step_encoders > 0);
    assert!(second_stats.last_step_forced_waits > 0);
    assert!(second_stats.last_step_cpu_wall_ns > 0);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn metal_probe_pipeline_compilation_counters_do_not_change_after_rollout() {
    let env_guard = MetalDebugSyncEnvGuard::new();
    env_guard.set_current(None);
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");

    let after_prepare = backend.probe_perf_stats();
    assert_eq!(
        after_prepare.library_compiles, 1,
        "prepare should compile the Metal library exactly once"
    );
    assert_eq!(
        after_prepare.pipeline_state_compiles,
        kernels::KERNEL_COUNT as u64,
        "prepare should compile all known PSOs before rollout"
    );
    assert_eq!(after_prepare.command_buffers, 0);
    assert_eq!(after_prepare.encoders, 0);

    let first = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let first_ticket = backend
        .launch_rollout(&first, None)
        .expect("run first rollout");
    let first_out = backend
        .collect(first_ticket)
        .expect("collect first rollout");
    assert_eq!(first_out[0].token_id, rvllm_core::TokenId(3));
    let after_first = backend.probe_perf_stats();
    assert_eq!(
        after_first.library_compiles, after_prepare.library_compiles,
        "rollout must not compile a Metal library"
    );
    assert_eq!(
        after_first.pipeline_state_compiles, after_prepare.pipeline_state_compiles,
        "rollout must not compile PSOs"
    );
    assert!(after_first.command_buffers > after_prepare.command_buffers);
    assert!(after_first.encoders > after_prepare.encoders);

    let second = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![first_out[0].token_id],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let second_ticket = backend
        .launch_rollout(&second, None)
        .expect("run second rollout");
    let second_out = backend
        .collect(second_ticket)
        .expect("collect second rollout");
    assert_eq!(second_out[0].token_id, rvllm_core::TokenId(4));
    let after_second = backend.probe_perf_stats();
    assert_eq!(
        after_second.library_compiles, after_prepare.library_compiles,
        "subsequent rollout must not compile a Metal library"
    );
    assert_eq!(
        after_second.pipeline_state_compiles, after_prepare.pipeline_state_compiles,
        "subsequent rollout must not compile PSOs"
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn metal_probe_arena_regions_do_not_change_after_rollout() {
    let env_guard = MetalDebugSyncEnvGuard::new();
    env_guard.set_current(None);
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");

    let after_prepare = backend
        .probe_arena_stats()
        .expect("arena stats after prepare");
    assert!(after_prepare.region_count > 0);
    assert!(after_prepare.allocated_bytes > 0);
    assert!(after_prepare.capacity_bytes >= after_prepare.allocated_bytes);

    let first = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let first_ticket = backend
        .launch_rollout(&first, None)
        .expect("run first rollout");
    let first_out = backend
        .collect(first_ticket)
        .expect("collect first rollout");
    assert_eq!(first_out[0].token_id, rvllm_core::TokenId(3));
    let after_first = backend
        .probe_arena_stats()
        .expect("arena stats after first rollout");
    assert_eq!(
        after_first, after_prepare,
        "rollout must reuse the prepared Metal arena without allocating regions"
    );

    let second = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![first_out[0].token_id],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let second_ticket = backend
        .launch_rollout(&second, None)
        .expect("run second rollout");
    let second_out = backend
        .collect(second_ticket)
        .expect("collect second rollout");
    assert_eq!(second_out[0].token_id, rvllm_core::TokenId(4));
    let after_second = backend
        .probe_arena_stats()
        .expect("arena stats after second rollout");
    assert_eq!(
        after_second, after_prepare,
        "subsequent rollout must not grow the Metal arena"
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_zero_layer_decode_loop_model_backend_generates_3_4_5() {
    let expected = cpu_reference_zero_layer_decode_loop_sequence()
        .into_iter()
        .map(|token| rvllm_core::TokenId(token as u32))
        .collect::<Vec<_>>();
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![
            rvllm_core::TokenId(0),
            rvllm_core::TokenId(1),
            rvllm_core::TokenId(2),
        ],
        vec![0, 3],
        vec![2],
        vec![3],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let mut current = rvllm_core::TokenId(2);
    let mut generated = Vec::new();
    for (position, context_len) in [(2, 3), (3, 4), (4, 5)] {
        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![position],
            vec![context_len],
        );
        let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect rollout");
        assert_eq!(out.len(), 1);
        current = out[0].token_id;
        generated.push(current);
    }

    assert_eq!(generated, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_zero_layer_decode_loop_generates_3_4_5() {
    let expected = cpu_reference_zero_layer_decode_loop_sequence()
        .into_iter()
        .map(|token| rvllm_core::TokenId(token as u32))
        .collect::<Vec<_>>();
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let plan = zero_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with zero-layer decode-loop tiny model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![
            rvllm_core::TokenId(0),
            rvllm_core::TokenId(1),
            rvllm_core::TokenId(2),
        ],
        3,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let mut generated = Vec::new();
    for expected_token in &expected {
        let step = engine.step_launch().expect("launch decode");
        let out = step.collect().expect("collect decode");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(&out[0].new_token, expected_token);
        generated.push(out[0].new_token);
    }

    assert_eq!(generated, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_zero_layer_decode_batch_two_returns_independent_tokens() {
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare zero-layer decode-loop tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(3)],
        vec![0, 1, 2],
        vec![0, 1],
        vec![1, 2],
    );
    let ticket = backend
        .launch_rollout(&handoff, None)
        .expect("run batched rollout");
    let out = backend.collect(ticket).expect("collect batched rollout");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));
    assert_eq!(out[1].req_id, rvllm_core::ReqId(2));
    assert_eq!(out[1].token_id, rvllm_core::TokenId(4));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_zero_layer_decode_batch_two_returns_exact_tokens() {
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let plan = zero_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with zero-layer decode-loop tiny model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));
    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(2),
        vec![rvllm_core::TokenId(0), rvllm_core::TokenId(3)],
        1,
    ));

    let prefill = engine.step_launch().expect("launch batched prefill");
    match prefill.plan().expect("prefill plan") {
        crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
            assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
        }
        other => panic!("expected Prefill, got {other:?}"),
    }
    assert!(prefill
        .collect()
        .expect("collect batched prefill")
        .is_empty());

    let decode = engine.step_launch().expect("launch batched decode");
    match decode.plan().expect("decode plan") {
        crate::scheduler::BatchPlan::Decode {
            req_ids,
            bucket,
            positions,
            context_lens,
            ..
        } => {
            assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            assert_eq!(*bucket, 2);
            assert_eq!(positions, &vec![0, 1]);
            assert_eq!(context_lens, &vec![1, 2]);
        }
        other => panic!("expected Decode, got {other:?}"),
    }
    let out = decode.collect().expect("collect batched decode");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out[0].new_token, rvllm_core::TokenId(3));
    assert_eq!(out[1].req_id, rvllm_core::ReqId(2));
    assert_eq!(out[1].new_token, rvllm_core::TokenId(4));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_zero_layer_decode_batch_four_returns_exact_tokens() {
    let dir = write_tiny_zero_layer_decode_loop_fixture();
    let plan = zero_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with zero-layer decode-loop tiny model plan");

    for (req_id, prompt) in [
        (1, vec![rvllm_core::TokenId(2)]),
        (2, vec![rvllm_core::TokenId(0), rvllm_core::TokenId(3)]),
        (
            3,
            vec![
                rvllm_core::TokenId(0),
                rvllm_core::TokenId(1),
                rvllm_core::TokenId(4),
            ],
        ),
        (4, vec![rvllm_core::TokenId(2)]),
    ] {
        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(req_id),
            prompt,
            1,
        ));
    }

    let prefill = engine.step_launch().expect("launch batched prefill");
    assert!(prefill
        .collect()
        .expect("collect batched prefill")
        .is_empty());

    let decode = engine.step_launch().expect("launch batched decode");
    match decode.plan().expect("decode plan") {
        crate::scheduler::BatchPlan::Decode {
            bucket,
            positions,
            context_lens,
            ..
        } => {
            assert_eq!(*bucket, 4);
            assert_eq!(positions, &vec![0, 1, 2, 0]);
            assert_eq!(context_lens, &vec![1, 2, 3, 1]);
        }
        other => panic!("expected Decode, got {other:?}"),
    }
    let out = decode.collect().expect("collect batched decode");
    let got = out
        .iter()
        .map(|step| (step.req_id, step.new_token))
        .collect::<Vec<_>>();
    assert_eq!(
        got,
        vec![
            (rvllm_core::ReqId(1), rvllm_core::TokenId(3)),
            (rvllm_core::ReqId(2), rvllm_core::TokenId(4)),
            (rvllm_core::ReqId(3), rvllm_core::TokenId(5)),
            (rvllm_core::ReqId(4), rvllm_core::TokenId(3)),
        ]
    );
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_integrated_gemma_probe_prefill_then_decode_token_2_to_3() {
    let expected =
        rvllm_core::TokenId(cpu_reference_one_layer_integrated_gemma_probe_argmax() as u32);
    let dir = write_tiny_one_layer_integrated_gemma_probe_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny integrated Gemma probe one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_prompt_len_two_model_backend_prefill_then_decode_token_2_4_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_prompt_len_two_prefill_argmax(true) as u32);
    let dir = write_tiny_prompt_len_two_prefill_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare prompt length two tiny model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let decode = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(4)],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let decode_ticket = backend.launch_rollout(&decode, None).expect("run rollout");
    let out = backend.collect(decode_ticket).expect("collect rollout");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_prompt_len_two_prefill_selected_logits_match_cpu() {
    let cpu_logits = cpu_reference_prompt_len_two_prefill_logits(true);
    let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
    let low_idx = 0usize;
    assert_eq!(expected_idx, 3);
    assert_eq!(runner_up_idx, 2);
    assert_eq!(cpu_logits[low_idx], 0.0);

    let dir = write_tiny_prompt_len_two_prefill_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare prompt length two tiny model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let decode = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(4)],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let decode_ticket = backend.launch_rollout(&decode, None).expect("run rollout");
    let metal_logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read decode logits");
    let out = backend.collect(decode_ticket).expect("collect rollout");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));

    const LOGIT_TOLERANCE: f32 = 0.05;
    assert_selected_logits_close(
        "prompt length two direct backend",
        &metal_logits,
        &cpu_logits,
        &[expected_idx, runner_up_idx, low_idx],
        LOGIT_TOLERANCE,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_prompt_len_two_prefill_then_decode_token_2_4_to_3() {
    let expected = rvllm_core::TokenId(cpu_reference_prompt_len_two_prefill_argmax(true) as u32);
    let dir = write_tiny_prompt_len_two_prefill_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with prompt length two tiny model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_prompt_len_two_prefill_selected_logits_match_cpu() {
    let cpu_logits = cpu_reference_prompt_len_two_prefill_logits(true);
    let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
    let low_idx = 0usize;
    assert_eq!(expected_idx, 3);
    assert_eq!(runner_up_idx, 2);
    assert_eq!(cpu_logits[low_idx], 0.0);

    let dir = write_tiny_prompt_len_two_prefill_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared prompt length two tiny model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(expected_idx as u32));
    assert!(!engine.has_pending_work());

    let metal_logits = shared_backend
        .debug_read_decode_logits_f32(1)
        .expect("read shared backend decode logits");
    const LOGIT_TOLERANCE: f32 = 0.05;
    assert_selected_logits_close(
        "prompt length two engine backend",
        &metal_logits,
        &cpu_logits,
        &[expected_idx, runner_up_idx, low_idx],
        LOGIT_TOLERANCE,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_qkv_norm_nonzero_model_backend_decodes_token_2_to_3() {
    let expected =
        rvllm_core::TokenId(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true) as u32);
    let dir = write_tiny_one_layer_qkv_norm_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer qkv-norm tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_qkv_norm_nonzero_prefill_then_decode_token_2_to_3() {
    let expected =
        rvllm_core::TokenId(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true) as u32);
    let dir = write_tiny_one_layer_qkv_norm_nonzero_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny qkv-norm one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_generated_gemma4_hf_model_backend_one_prompt_token_matches_cpu_token() {
    let expected =
        rvllm_core::TokenId(cpu_reference_generated_tiny_gemma4_hf_sequence(&[2], 1)[0] as u32);
    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare generated tiny Gemma4 HF-named model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_generated_gemma4_hf_end_to_end_model_backend_matches_cpu_tokens() {
    let expected = cpu_reference_generated_tiny_hf_end_to_end_sequence()
        .into_iter()
        .map(|token| rvllm_core::TokenId(token as u32))
        .collect::<Vec<_>>();
    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare generated tiny Gemma4 HF-named model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let mut current = rvllm_core::TokenId(4);
    let mut generated = Vec::new();
    for (idx, expected_token) in expected.iter().enumerate() {
        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![1 + idx as u32],
            vec![2 + idx as u32],
        );
        let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect rollout");
        assert_eq!(out.len(), 1);
        assert_eq!(&out[0].token_id, expected_token);
        current = out[0].token_id;
        generated.push(current);
    }

    assert_eq!(generated, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_generated_gemma4_hf_end_to_end_model_backend_full_logits_match_cpu() {
    let cpu_reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
    assert_eq!(cpu_reference.generated, vec![3, 5]);

    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare generated tiny Gemma4 HF-named model");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
    let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
    assert!(prefill_out.is_empty());

    let mut current = rvllm_core::TokenId(4);
    let mut generated = Vec::new();
    for (step_idx, expected_token) in cpu_reference.generated.iter().enumerate() {
        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![1 + step_idx as u32],
            vec![2 + step_idx as u32],
        );
        let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
        let metal_logits = backend
            .debug_read_decode_logits_f32(1)
            .expect("read decode logits");
        let out = backend.collect(ticket).expect("collect rollout");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(*expected_token as u32));
        current = out[0].token_id;
        generated.push(current);

        let cpu_logits = &cpu_reference.logits_by_step[step_idx];
        let expected_idx = cpu_full_nonzero_argmax(cpu_logits);
        assert_eq!(expected_idx, *expected_token);
        assert_eq!(metal_logits.len(), cpu_logits.len());

        const LOGIT_TOLERANCE: f32 = 0.05;
        assert_f32_slice_close(
            &format!("generated tiny HF direct decode step {}", step_idx + 1),
            &metal_logits,
            cpu_logits,
            LOGIT_TOLERANCE,
        );
    }

    let expected = cpu_reference
        .generated
        .iter()
        .map(|token| rvllm_core::TokenId(*token as u32))
        .collect::<Vec<_>>();
    assert_eq!(generated, expected);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_generated_gemma4_hf_end_to_end_matches_cpu_tokens() {
    let expected = cpu_reference_generated_tiny_hf_end_to_end_sequence()
        .into_iter()
        .map(|token| rvllm_core::TokenId(token as u32))
        .collect::<Vec<_>>();
    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with generated tiny Gemma4 HF-named model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        expected.len() as u32,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let mut generated = Vec::new();
    for expected_token in &expected {
        let step = engine.step_launch().expect("launch decode");
        let out = step.collect().expect("collect decode");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(&out[0].new_token, expected_token);
        generated.push(out[0].new_token);
    }

    assert_eq!(generated, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_generated_gemma4_hf_end_to_end_full_logits_match_cpu() {
    let cpu_reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
    assert_eq!(cpu_reference.generated, vec![3, 5]);

    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared generated tiny Gemma4 HF-named model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        cpu_reference.generated.len() as u32,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let mut generated = Vec::new();
    for (step_idx, expected_token) in cpu_reference.generated.iter().enumerate() {
        let step = engine.step_launch().expect("launch decode");
        let out = step.collect().expect("collect decode");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(
            out[0].new_token,
            rvllm_core::TokenId(*expected_token as u32)
        );
        generated.push(out[0].new_token);

        let metal_logits = shared_backend
            .debug_read_decode_logits_f32(1)
            .expect("read shared backend decode logits");
        let cpu_logits = &cpu_reference.logits_by_step[step_idx];
        let expected_idx = cpu_full_nonzero_argmax(cpu_logits);
        assert_eq!(expected_idx, *expected_token);
        assert_eq!(metal_logits.len(), cpu_logits.len());

        const LOGIT_TOLERANCE: f32 = 0.05;
        assert_f32_slice_close(
            &format!("generated tiny HF engine decode step {}", step_idx + 1),
            &metal_logits,
            cpu_logits,
            LOGIT_TOLERANCE,
        );
    }

    let expected = cpu_reference
        .generated
        .iter()
        .map(|token| rvllm_core::TokenId(*token as u32))
        .collect::<Vec<_>>();
    assert_eq!(generated, expected);
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_ffn_nonzero_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_one_layer_ffn_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer ffn-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_ffn_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_one_layer_ffn_nonzero_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny ffn-nonzero one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_attention_nonzero_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_one_layer_attention_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer attention-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_attention_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_one_layer_attention_nonzero_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny attention-nonzero one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_multihead_gqa_attention_model_backend_decodes_token_2_to_3() {
    assert_eq!(cpu_reference_multihead_gqa_attention_argmax(), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare multi-head GQA attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_multihead_gqa_attention_model_backend_full_logits_match_cpu() {
    let cpu_logits = cpu_reference_multihead_gqa_attention_logits();
    assert_eq!(cpu_full_nonzero_argmax(&cpu_logits), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare multi-head GQA attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let metal_logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read decode logits");
    assert_eq!(metal_logits.len(), cpu_logits.len());
    assert_f32_slice_close(
        "multi-head GQA direct decode logits",
        &metal_logits,
        &cpu_logits,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_multihead_gqa_attention_model_backend_full_residual_matches_cpu() {
    let reference = cpu_reference_multihead_gqa_attention();
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare multi-head GQA attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let metal_residual = backend
        .debug_read_residual_f32(1)
        .expect("read decode residual");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));
    assert_eq!(metal_residual.len(), reference.residual.len());
    assert_f32_slice_close(
        "multi-head GQA direct residual",
        &metal_residual,
        &reference.residual,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_multihead_gqa_attention_prefill_then_decode_token_2_to_3() {
    assert_eq!(cpu_reference_multihead_gqa_attention_argmax(), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny multi-head GQA attention model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_multihead_gqa_attention_full_logits_match_cpu() {
    let cpu_logits = cpu_reference_multihead_gqa_attention_logits();
    assert_eq!(cpu_full_nonzero_argmax(&cpu_logits), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared multi-head GQA attention model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let metal_logits = shared_backend
        .debug_read_decode_logits_f32(1)
        .expect("read shared backend decode logits");
    assert_eq!(metal_logits.len(), cpu_logits.len());
    assert_f32_slice_close(
        "multi-head GQA engine decode logits",
        &metal_logits,
        &cpu_logits,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_multihead_gqa_attention_full_residual_matches_cpu() {
    let reference = cpu_reference_multihead_gqa_attention();
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);

    let dir = write_tiny_multihead_gqa_attention_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared multi-head GQA attention model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let metal_residual = shared_backend
        .debug_read_residual_f32(1)
        .expect("read shared backend decode residual");
    assert_eq!(metal_residual.len(), reference.residual.len());
    assert_f32_slice_close(
        "multi-head GQA engine residual",
        &metal_residual,
        &reference.residual,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_qdim_not_hidden_model_backend_decodes_token_2_to_3() {
    assert_eq!(cpu_reference_qdim_not_hidden_argmax(), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare q_dim != hidden attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_qdim_not_hidden_model_backend_full_logits_match_cpu() {
    let cpu_logits = cpu_reference_qdim_not_hidden_logits();
    assert_eq!(cpu_full_nonzero_argmax(&cpu_logits), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare q_dim != hidden attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let metal_logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read decode logits");
    assert_eq!(metal_logits.len(), cpu_logits.len());
    assert_f32_slice_close(
        "q_dim != hidden direct decode logits",
        &metal_logits,
        &cpu_logits,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_qdim_not_hidden_model_backend_full_residual_matches_cpu() {
    let reference = cpu_reference_qdim_not_hidden();
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare q_dim != hidden attention tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let metal_residual = backend
        .debug_read_residual_f32(1)
        .expect("read decode residual");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));
    assert_eq!(metal_residual.len(), reference.residual.len());
    assert_f32_slice_close(
        "q_dim != hidden direct residual",
        &metal_residual,
        &reference.residual,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_tiny_shared_kv_tail_poison_fixture(
    poison_tail_local_kv: bool,
    poison_source_v: bool,
) -> (Vec<f32>, rvllm_core::TokenId) {
    let dir = write_tiny_shared_kv_tail_poison_fixture(poison_tail_local_kv, poison_source_v);
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = n_layer_plan(dir.clone(), 3);
    backend
        .prepare(&plan)
        .expect("prepare tiny shared-KV poison model");
    let state = backend.state.as_ref().expect("prepared model state");
    assert_eq!(state.layers[2].shared_kv_source_layer, Some(1));

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    let logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read tiny shared-KV poison logits");
    assert_eq!(logits.len(), 8);

    let _ = fs::remove_dir_all(dir);
    (logits, out[0].token_id)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_shared_kv_tail_local_kv_poison_does_not_change_logits_but_source_v_does() {
    let (base_logits, base_token) = run_tiny_shared_kv_tail_poison_fixture(false, false);
    let (tail_poison_logits, tail_poison_token) =
        run_tiny_shared_kv_tail_poison_fixture(true, false);
    let (source_poison_logits, source_poison_token) =
        run_tiny_shared_kv_tail_poison_fixture(false, true);

    assert_eq!(base_token, rvllm_core::TokenId(3));
    assert_eq!(tail_poison_token, base_token);
    assert_f32_slice_close(
        "tail-local K/V poison should not affect shared-KV tail logits",
        &tail_poison_logits,
        &base_logits,
        0.02,
    );

    let source_delta = (source_poison_logits[3] - base_logits[3]).abs();
    assert!(
        source_delta > 1.0,
        "poisoning source-layer V should affect tail logits: base_token={base_token:?} source_token={source_poison_token:?} base_logit3={} source_logit3={} delta={source_delta}",
        base_logits[3],
        source_poison_logits[3]
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_qdim_not_hidden_prefill_then_decode_token_2_to_3() {
    assert_eq!(cpu_reference_qdim_not_hidden_argmax(), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny q_dim != hidden attention model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_qdim_not_hidden_full_logits_match_cpu() {
    let cpu_logits = cpu_reference_qdim_not_hidden_logits();
    assert_eq!(cpu_full_nonzero_argmax(&cpu_logits), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared q_dim != hidden attention model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let metal_logits = shared_backend
        .debug_read_decode_logits_f32(1)
        .expect("read shared backend decode logits");
    assert_eq!(metal_logits.len(), cpu_logits.len());
    assert_f32_slice_close(
        "q_dim != hidden engine decode logits",
        &metal_logits,
        &cpu_logits,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_qdim_not_hidden_full_residual_matches_cpu() {
    let reference = cpu_reference_qdim_not_hidden();
    assert_eq!(cpu_full_nonzero_argmax(&reference.logits), 3);

    let dir = write_tiny_qdim_not_hidden_attention_fixture();
    let plan = one_layer_plan(dir.clone());
    let shared_backend = SharedModelMetalBackend::new(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_backend(Box::new(shared_backend.clone()))
        .with_apple_runtime_plan(plan)
        .expect("engine with shared q_dim != hidden attention model backend");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let metal_residual = shared_backend
        .debug_read_residual_f32(1)
        .expect("read shared backend decode residual");
    assert_eq!(metal_residual.len(), reference.residual.len());
    assert_f32_slice_close(
        "q_dim != hidden engine residual",
        &metal_residual,
        &reference.residual,
        0.05,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_full_nonzero_model_backend_decodes_token_2_to_3() {
    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer full-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal and RVLLM_TEST_APPLE_METALLIB_ROOT"]
fn schema_v3_w4_w8_down_proj_sidecars_are_selected_and_execute_real_layer() {
    use rvllm_apple::model_package_builder::{
        build_apple_model_package, AppleLowBitExportRequest, AppleModelPackageBuildConfig,
    };

    let Some(metallib_root) = std::env::var_os("RVLLM_TEST_APPLE_METALLIB_ROOT") else {
        eprintln!("skipping: RVLLM_TEST_APPLE_METALLIB_ROOT is not set");
        return;
    };
    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let config_path = dir.join("config.json");
    let mut config: Value =
        serde_json::from_slice(&fs::read(&config_path).expect("read fixture config"))
            .expect("parse fixture config");
    config
        .as_object_mut()
        .expect("fixture config object")
        .insert("model_type".to_owned(), Value::String("gemma4".to_owned()));
    fs::write(
        &config_path,
        serde_json::to_vec_pretty(&config).expect("encode package fixture config"),
    )
    .expect("write package fixture config");
    fs::write(dir.join("tokenizer.json"), b"tiny-tokenizer").expect("write fixture tokenizer");
    let hidden = 128_usize;
    let intermediate = 256_usize;

    // Add a second active lane to one down-projection quantization group.
    // The resulting W4 and W8 projections both differ materially from native
    // F16 while the LM head still ignores the affected FFN output dimension.
    patch_fixture_f16_tensor(
        &dir,
        "model.layers.0.mlp.gate_proj.weight",
        hidden + FULL_NONZERO_ORIGINAL_DIM,
        1.0,
    );
    patch_fixture_f16_tensor(
        &dir,
        "model.layers.0.mlp.up_proj.weight",
        hidden + FULL_NONZERO_ORIGINAL_DIM,
        1.0,
    );
    patch_fixture_f16_tensor(
        &dir,
        "model.layers.0.mlp.down_proj.weight",
        FULL_NONZERO_FFN_DIM * intermediate + 1,
        0.992_187_5,
    );

    let env_guard =
        MetalDebugEnvGuard::new(&[RVLLM_METAL_DTYPE_ENV, RVLLM_METAL_DEBUG_TRACE_LAYER_ENV]);
    env_guard.set(RVLLM_METAL_DTYPE_ENV, "f16");
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV, "0");

    for format in [
        rvllm_apple::AppleLowBitWeightFormat::W4A16,
        rvllm_apple::AppleLowBitWeightFormat::W8A16,
    ] {
        let package_root = dir.with_file_name(format!(
            "{}-{}-package",
            dir.file_name()
                .and_then(|name| name.to_str())
                .expect("UTF-8 fixture name"),
            format.name()
        ));
        let build = AppleModelPackageBuildConfig {
            model_dir: dir.clone(),
            metallib_root: std::path::PathBuf::from(&metallib_root),
            output_dir: package_root.clone(),
            package_id: format!("tiny-one-layer-{}", format.name()),
            weight_format: None,
            low_bit_down_projections: vec![AppleLowBitExportRequest {
                tensor_name: "model.layers.0.mlp.down_proj.weight".to_owned(),
                format,
            }],
        };
        let report = build_apple_model_package(&build).expect("build schema-v3 hybrid package");
        assert_eq!(report.low_bit_tensors, 1);
        assert_eq!(report.low_bit_formats, vec![format]);

        let package =
            rvllm_apple::AppleModelPackage::open(&package_root).expect("open hybrid package");
        assert_eq!(
            package.manifest().schema_version,
            rvllm_apple::AppleModelPackageManifest::SCHEMA_V3
        );
        let tensor = package
            .low_bit_tensor("model.layers.0.mlp.down_proj.weight")
            .expect("authenticated low-bit descriptor");
        assert_eq!(tensor.format, format);
        assert_eq!(tensor.shape, [hidden as u32, intermediate as u32]);
        let packed = package
            .load_low_bit_tensor("model.layers.0.mlp.down_proj.weight")
            .expect("load canonical low-bit tensor");

        let mut backend = ModelMetalBackend::from_model_package_path(package_root.clone())
            .expect("construct packaged Metal backend");
        backend
            .prepare(&one_layer_plan(package_root.clone()))
            .expect("prepare packaged low-bit model");
        let capacity = backend.model_capacity().expect("capacity report");
        let hybrid_weights_bytes = capacity.weights_bytes;
        let hybrid_numeric_fingerprint = capacity.numeric_abi_fingerprint;
        assert_eq!(capacity.low_bit_projection_count, 1);
        assert_eq!(
            capacity.low_bit_weight_bytes,
            (packed.packed_values().len()
                + packed.scales().len() * std::mem::size_of::<half::f16>()) as u64
        );
        assert_ne!(
            capacity.numeric_abi_fingerprint,
            metal_numeric_abi_fingerprint(MetalFloatType::F16, false, false)
        );
        let selected = backend.state.as_ref().expect("prepared state").layers[0]
            .low_bit_down_proj
            .expect("selected low-bit down projection");
        assert_eq!(selected.format(), format);
        assert_eq!(selected.shape(), [hidden as u32, intermediate as u32]);
        assert_eq!(
            selected.kernel_name(),
            match format {
                rvllm_apple::AppleLowBitWeightFormat::W4A16 => "projection_w4a16_f16",
                rvllm_apple::AppleLowBitWeightFormat::W8A16 => "projection_w8a16_f16",
            }
        );
        assert!(
            backend.state.as_ref().expect("prepared state").layers[0]
                .down_proj
                .as_ref()
                .expect("hybrid native fallback")
                .size
                > 0,
            "native F16 fallback must remain resident"
        );

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![
                rvllm_core::TokenId(2),
                rvllm_core::TokenId(2),
                rvllm_core::TokenId(2),
            ],
            vec![0, 3],
            vec![2],
            vec![3],
        );
        let ticket = backend
            .launch_prefill(&handoff)
            .expect("launch real multi-token low-bit prefill");
        let out = backend.collect(ticket).expect("collect low-bit prefill");
        assert!(out.is_empty());

        let state = backend.state.as_ref().expect("prepared state");
        let trace = state.layers[0].trace.as_ref().expect("layer trace");
        let arena = backend.arena.as_ref().expect("prepared arena");
        let activation = unsafe {
            std::slice::from_raw_parts(
                arena.host_ptr(&trace.ffn_activation).cast::<u16>(),
                3 * intermediate,
            )
            .iter()
            .map(|bits| half::f16::from_bits(*bits))
            .collect::<Vec<_>>()
        };
        let actual = unsafe {
            std::slice::from_raw_parts(
                arena.host_ptr(&trace.after_ffn_branch).cast::<u16>(),
                3 * hidden,
            )
            .iter()
            .map(|bits| half::f16::from_bits(*bits))
            .collect::<Vec<_>>()
        };
        let expected = rvllm_apple::project_apple_low_bit_reference(&packed, &activation, 3)
            .expect("CPU low-bit projection");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let expected = expected.to_f32();
            let tolerance = 0.02 + expected.abs() * 0.002;
            assert!(
                (actual.to_f32() - expected).abs() <= tolerance,
                "{format:?} real layer output[{index}] actual={} expected={expected} tolerance={tolerance}",
                actual.to_f32()
            );
        }
        let dense = activation[0].to_f32() * 4.0 + activation[1].to_f32() * 0.992_187_5;
        let low_bit = expected[FULL_NONZERO_FFN_DIM].to_f32();
        let metal = actual[FULL_NONZERO_FFN_DIM].to_f32();
        assert!(
            (dense - low_bit).abs() > 0.5,
            "{format:?} fixture must distinguish native dense ({dense}) from low-bit ({low_bit})"
        );
        assert!((metal - low_bit).abs() < (metal - dense).abs());

        drop(backend);
        let mut replacement_backend =
            ModelMetalBackend::from_model_package_path(package_root.clone())
                .expect("construct replacement Metal backend")
                .with_low_bit_residency_policy(MetalLowBitResidencyPolicy::ReplaceNative);
        replacement_backend
            .prepare(&one_layer_plan(package_root.clone()))
            .expect("prepare native-replacement low-bit model");
        let replacement_capacity = replacement_backend
            .model_capacity()
            .expect("replacement capacity report");
        assert_eq!(replacement_capacity.low_bit_projection_count, 1);
        assert!(replacement_capacity.weights_bytes < hybrid_weights_bytes);
        assert_ne!(
            replacement_capacity.numeric_abi_fingerprint,
            hybrid_numeric_fingerprint
        );
        let replacement_layer = &replacement_backend
            .state
            .as_ref()
            .expect("replacement state")
            .layers[0];
        assert!(replacement_layer.down_proj.is_none());
        assert!(replacement_layer.low_bit_down_proj.is_some());
        let replacement_ticket = replacement_backend
            .launch_prefill(&handoff)
            .expect("launch native-replacement low-bit prefill");
        assert!(replacement_backend
            .collect(replacement_ticket)
            .expect("collect native-replacement low-bit prefill")
            .is_empty());
        let _ = fs::remove_dir_all(package_root);
    }
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal and RVLLM_TEST_APPLE_METALLIB_ROOT"]
fn schema_v3_two_layer_mixed_low_bit_package_replaces_both_native_projections() {
    use rvllm_apple::model_package_builder::{
        build_apple_model_package, AppleLowBitExportRequest, AppleModelPackageBuildConfig,
    };

    let Some(metallib_root) = std::env::var_os("RVLLM_TEST_APPLE_METALLIB_ROOT") else {
        eprintln!("skipping: RVLLM_TEST_APPLE_METALLIB_ROOT is not set");
        return;
    };
    let dir = write_tiny_two_layer_fixture(false);
    let config_path = dir.join("config.json");
    let mut config: Value =
        serde_json::from_slice(&fs::read(&config_path).expect("read fixture config"))
            .expect("parse fixture config");
    config
        .as_object_mut()
        .expect("fixture config object")
        .insert("model_type".to_owned(), Value::String("gemma4".to_owned()));
    fs::write(
        &config_path,
        serde_json::to_vec_pretty(&config).expect("encode package fixture config"),
    )
    .expect("write package fixture config");
    fs::write(dir.join("tokenizer.json"), b"tiny-tokenizer").expect("write fixture tokenizer");

    let package_root = dir.with_file_name(format!(
        "{}-mixed-low-bit-package",
        dir.file_name()
            .and_then(|name| name.to_str())
            .expect("UTF-8 fixture name")
    ));
    let report = build_apple_model_package(&AppleModelPackageBuildConfig {
        model_dir: dir.clone(),
        metallib_root: std::path::PathBuf::from(metallib_root),
        output_dir: package_root.clone(),
        package_id: "tiny-two-layer-mixed-low-bit".to_owned(),
        weight_format: None,
        // Deliberately reverse manifest input order. Runtime preflight and
        // arena installation must use canonical tensor-name order.
        low_bit_down_projections: vec![
            AppleLowBitExportRequest {
                tensor_name: "model.layers.1.mlp.down_proj.weight".to_owned(),
                format: AppleLowBitWeightFormat::W8A16,
            },
            AppleLowBitExportRequest {
                tensor_name: "model.layers.0.mlp.down_proj.weight".to_owned(),
                format: AppleLowBitWeightFormat::W4A16,
            },
        ],
    })
    .expect("build two-layer mixed low-bit package");
    assert_eq!(report.low_bit_tensors, 2);

    let env_guard = MetalDebugEnvGuard::new(&[RVLLM_METAL_DTYPE_ENV]);
    env_guard.set(RVLLM_METAL_DTYPE_ENV, "f16");
    let mut backend = ModelMetalBackend::from_model_package_path(package_root.clone())
        .expect("construct two-layer package backend")
        .with_low_bit_residency_policy(MetalLowBitResidencyPolicy::ReplaceNative);
    backend
        .prepare(&two_layer_plan(package_root.clone()))
        .expect("prepare two-layer replacement model");
    let state = backend.state.as_ref().expect("prepared state");
    assert_eq!(state.layers.len(), 2);
    assert!(state
        .layers
        .iter()
        .all(|layer| layer.down_proj.is_none() && layer.low_bit_down_proj.is_some()));
    assert_eq!(
        state.layers[0]
            .low_bit_down_proj
            .expect("layer zero sidecar")
            .format(),
        AppleLowBitWeightFormat::W4A16
    );
    assert_eq!(
        state.layers[1]
            .low_bit_down_proj
            .expect("layer one sidecar")
            .format(),
        AppleLowBitWeightFormat::W8A16
    );
    assert_eq!(
        backend
            .model_capacity()
            .expect("two-layer capacity")
            .low_bit_projection_count,
        2
    );

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );
    let ticket = backend
        .launch_prefill(&handoff)
        .expect("launch two-layer mixed low-bit prefill");
    assert!(backend
        .collect(ticket)
        .expect("collect two-layer mixed low-bit prefill")
        .is_empty());

    let _ = fs::remove_dir_all(package_root);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_full_nonzero_model_backend_selected_logits_match_cpu() {
    let cpu_logits = cpu_reference_one_layer_full_nonzero_logits();
    let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
    let low_idx = 0usize;
    assert_eq!(expected_idx, 3);
    assert_eq!(runner_up_idx, 2);
    assert_eq!(cpu_logits[low_idx], 0.0);

    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer full-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let metal_logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read decode logits");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));
    assert_eq!(metal_logits.len(), cpu_logits.len());

    const LOGIT_TOLERANCE: f32 = 0.05;
    for idx in [expected_idx, runner_up_idx, low_idx] {
        let diff = (metal_logits[idx] - cpu_logits[idx]).abs();
        assert!(
            diff <= LOGIT_TOLERANCE,
            "logit[{idx}] mismatch: metal={} cpu={} diff={} tol={}",
            metal_logits[idx],
            cpu_logits[idx],
            diff,
            LOGIT_TOLERANCE
        );
    }

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_full_nonzero_model_backend_selected_hidden_matches_cpu() {
    let reference = cpu_reference_one_layer_full_nonzero();
    let cpu_logits = &reference.logits;
    let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(cpu_logits);
    let low_idx = FULL_NONZERO_ZERO_DIM;
    assert_eq!(expected_idx, 3);
    assert_eq!(runner_up_idx, 2);
    assert_eq!(cpu_logits[low_idx], 0.0);

    let selected_dims = [
        FULL_NONZERO_ZERO_DIM,
        FULL_NONZERO_ORIGINAL_DIM,
        FULL_NONZERO_ATTENTION_DIM,
        FULL_NONZERO_FFN_DIM,
    ];

    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer full-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let metal_residual = backend
        .debug_read_residual_f32(1)
        .expect("read decode residual");
    let metal_logits = backend
        .debug_read_decode_logits_f32(1)
        .expect("read decode logits");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));
    assert_eq!(metal_residual.len(), reference.residual.len());
    assert_eq!(metal_logits.len(), cpu_logits.len());

    const HIDDEN_TOLERANCE: f32 = 0.05;
    for dim in selected_dims {
        assert_f32_close(
            &format!("residual[{dim}]"),
            metal_residual[dim],
            reference.residual[dim],
            HIDDEN_TOLERANCE,
        );
    }

    const LOGIT_TOLERANCE: f32 = 0.05;
    for idx in [expected_idx, runner_up_idx, low_idx] {
        assert_f32_close(
            &format!("logit[{idx}]"),
            metal_logits[idx],
            cpu_logits[idx],
            LOGIT_TOLERANCE,
        );
    }

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn tiny_one_layer_full_nonzero_model_backend_full_residual_matches_cpu() {
    let reference = cpu_reference_one_layer_full_nonzero();
    let expected_idx = cpu_full_nonzero_argmax(&reference.logits);
    assert_eq!(expected_idx, 3);

    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = one_layer_plan(dir.clone());
    backend
        .prepare(&plan)
        .expect("prepare one-layer full-nonzero tiny model");

    let handoff = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2)],
        vec![0, 1],
        vec![0],
        vec![1],
    );

    let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
    let metal_residual = backend
        .debug_read_residual_f32(1)
        .expect("read decode residual");
    let out = backend.collect(ticket).expect("collect");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));

    const RESIDUAL_TOLERANCE: f32 = 0.05;
    assert_f32_slice_close(
        "full nonzero Metal residual",
        &metal_residual,
        &reference.residual,
        RESIDUAL_TOLERANCE,
    );

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Silicon Metal device"]
fn engine_one_layer_full_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
    let dir = write_tiny_one_layer_full_nonzero_fixture();
    let plan = one_layer_plan(dir.clone());

    let mut engine = crate::engine::Engine::new()
        .with_apple_runtime_plan(plan)
        .expect("engine with tiny full-nonzero one-layer model plan");

    engine.scheduler.enqueue(crate::sched_state::Request::new(
        rvllm_core::ReqId(1),
        vec![rvllm_core::TokenId(2)],
        1,
    ));

    let step1 = engine.step_launch().expect("launch prefill");
    let out1 = step1.collect().expect("collect prefill");
    assert!(out1.is_empty());

    let step2 = engine.step_launch().expect("launch decode");
    let out2 = step2.collect().expect("collect decode");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
    assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
    assert!(!engine.has_pending_work());

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn generated_tiny_gemma4_hf_fixture_uses_real_names_and_dry_run_validates() {
    let dir = write_generated_tiny_hf_end_to_end_fixture();
    let tensors = rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
    let prefix = "model.language_model";

    assert!(tensors.contains_key(&format!("{prefix}.embed_tokens.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.norm.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.lm_head.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.q_proj.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.k_proj.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.v_proj.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.q_norm.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.k_norm.weight")));
    assert!(tensors.contains_key(&format!(
        "{prefix}.layers.0.post_attention_layernorm.weight"
    )));
    assert!(tensors.contains_key(&format!(
        "{prefix}.layers.0.pre_feedforward_layernorm.weight"
    )));
    assert!(tensors.contains_key(&format!(
        "{prefix}.layers.0.post_feedforward_layernorm.weight"
    )));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.layer_scalar")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.gate_proj.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.up_proj.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.down_proj.weight")));
    assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
    assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));

    let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
        .expect("generated Gemma4 fixture dry-run validates");
    assert_eq!(validation.weight_prefix, prefix);
    assert_eq!(validation.final_logit_softcap, Some(30.0));
    assert_eq!(
        validation.layers[0].attention_kind,
        rvllm_apple_metal::gemma4_model::MetalProbeLayerAttentionKind::Full
    );
    assert_eq!(validation.layers[0].layer_scalar_dim, 128);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn hf_style_one_layer_fixture_has_separate_tensors() {
    let dir = write_tiny_one_layer_hf_style_noop_fixture();
    let tensors = rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.mlp.up_proj.weight"));
    assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
    assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn real_hf_style_one_layer_slice_fixture_has_hf_names_and_norm_alias() {
    let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
    let tensors = rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.o_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.mlp.up_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.mlp.down_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.input_layernorm.weight"));
    assert!(tensors.contains_key("model.layers.0.post_attention_layernorm.weight"));
    assert!(!tensors.contains_key("model.layers.0.mlp_norm.weight"));
    assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
    assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_backend_prepare_rejects_missing_dir() {
    let dir = std::env::temp_dir().join("rvllm-definitely-missing-model-dir");
    let _ = fs::remove_dir_all(&dir);

    let mut backend = ModelMetalBackend::new(dir.clone());
    let plan = zero_layer_plan(dir);
    let err = backend.prepare(&plan).expect_err("missing dir should fail");
    let s = format!("{err}");
    assert!(s.contains("InvalidWeightBlob") || s.contains("missing model path"));
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory in RVLLM_GEMMA4_MODEL_DIR"]
fn real_gemma4_e2b_model_backend_prepare_reports_current_large_model_gate() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let validation = rvllm_loader::Gemma4DryRunValidation::from_model_dir(&model_dir)
        .expect("real Gemma4 E2B dry-run metadata should validate before prepare");
    assert_eq!(validation.num_layers, 35);
    assert_eq!(validation.hidden_size, 1536);
    assert_eq!(validation.vocab_size, 262144);

    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before prepare");
    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    let mut backend = ModelMetalBackend::new(model_dir);
    let err = backend
        .prepare(&plan)
        .expect_err("current Metal prepare should report the large-model layer gate");
    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    }
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported_probe_num_layers_without_large_model_opt_in"),
        "unexpected prepare error: {msg}"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory in RVLLM_GEMMA4_MODEL_DIR and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prepare_with_large_model_opt_in() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before prepare");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    let prepare = backend.prepare(&plan);

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    prepare.expect("real Gemma4 E2B Metal prepare/load should complete under explicit opt-in");
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device"]
fn real_gemma4_e2b_model_capacity_reports_budgeted_kv_accounts() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before prepare");
    let env_guard = MetalDebugEnvGuard::new(&[
        "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE",
        "RVLLM_METAL_MAX_TOTAL_TOKENS",
        "RVLLM_METAL_MAX_BATCH_TOKENS",
        "RVLLM_METAL_MAX_BATCH_SEQUENCES",
    ]);
    env_guard.set("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");
    env_guard.set("RVLLM_METAL_MAX_TOTAL_TOKENS", "128");
    env_guard.set("RVLLM_METAL_MAX_BATCH_TOKENS", "128");
    env_guard.set("RVLLM_METAL_MAX_BATCH_SEQUENCES", "8");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    backend.prepare(&plan).expect("budgeted prepare");
    let capacity = backend.model_capacity().expect("prepared capacity report");

    assert_eq!(capacity.max_context_tokens, 128);
    assert_eq!(capacity.max_batch_tokens, 128);
    assert_eq!(capacity.max_batch_sequences, 8);
    assert_eq!(capacity.kv_page_size, 32);
    assert_eq!(capacity.max_useful_kv_pages, 32);
    assert_eq!(capacity.admission_required_kv_pages, 32);
    assert_eq!(capacity.physical_kv_pages, 32);
    assert_eq!(
        capacity.reserve_bytes,
        capacity.recommended_working_set_bytes * 20 / 100
    );
    assert_eq!(
        capacity.usable_bytes,
        capacity.recommended_working_set_bytes - capacity.reserve_bytes
    );
    assert_eq!(
        capacity.scratch_budget_bytes,
        capacity.scratch_slot_bytes * 3
    );
    assert_eq!(
        capacity.allocated_kv_bytes,
        capacity.kv_page_bytes * u64::from(capacity.physical_kv_pages)
    );
    assert_eq!(
        capacity.kv_budget_bytes,
        capacity.usable_bytes
            - capacity.weights_bytes
            - capacity.scratch_budget_bytes
            - capacity.metadata_bytes
    );
    assert!(capacity.prepared_arena_bytes <= capacity.usable_bytes);
    assert_eq!(
        capacity.metal_float_type,
        backend.metal_compute_dtype_report()
    );
    assert_eq!(capacity.kv_storage_format, capacity.metal_float_type);
    assert!(!capacity.experimental_kv_int8_active);
    assert_eq!(capacity.numeric_abi_version, METAL_NUMERIC_ABI_VERSION);
    assert_ne!(capacity.numeric_abi_fingerprint, [0; 32]);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and bounded layer finite debug opt-in"]
fn real_gemma4_e2b_prefill_layers_0_to_4_residuals_are_finite() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before bounded finite run");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);

    let env_guard = MetalDebugEnvGuard::new(&[
        RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV,
        RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV,
        RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
    ]);
    env_guard.set(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV, "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    backend
        .prepare(&plan)
        .expect("real Gemma4 E2B Metal prepare/load should complete before finite run");

    for stop_after_layer in 0usize..=4 {
        env_guard.set(
            RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
            stop_after_layer.to_string(),
        );
        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let ticket = backend.launch_prefill(&prefill).unwrap_or_else(|err| {
            panic!("bounded E2B prefill failed at layer {stop_after_layer}: {err}")
        });
        let out = backend.collect(ticket).unwrap_or_else(|err| {
            panic!("bounded E2B prefill collect failed at layer {stop_after_layer}: {err}")
        });
        assert!(out.is_empty(), "bounded prefill must not sample logits");

        let residual = backend.debug_read_residual_f32(2).unwrap_or_else(|err| {
            panic!("bounded E2B residual read failed at layer {stop_after_layer}: {err}")
        });
        let summary = finite_summary(&residual);
        eprintln!(
            "bounded E2B residual after layer {stop_after_layer}: finite={}/{} max_abs={:e} mean_abs={:e} first_nonfinite_index={:?}",
            summary.finite_count,
            summary.total_count,
            summary.max_abs,
            summary.mean_abs,
            summary.first_nonfinite_index
        );
        assert_eq!(
            summary.first_nonfinite_index, None,
            "first non-finite residual after local kernel fixes appears at or before layer {stop_after_layer}: {summary:?}"
        );
        assert_eq!(summary.finite_count, summary.total_count);
    }
    env_guard.remove(RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and bounded layer finite debug opt-in"]
fn real_gemma4_e2b_batch_two_prefill_layers_0_to_4_residuals_are_finite() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before bounded batch finite run");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);

    let env_guard = MetalDebugEnvGuard::new(&[
        RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV,
        RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV,
        RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
    ]);
    env_guard.set(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV, "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    backend
        .prepare(&plan)
        .expect("real Gemma4 E2B Metal prepare/load should complete before batch finite run");

    for stop_after_layer in 0usize..=4 {
        env_guard.set(
            RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
            stop_after_layer.to_string(),
        );
        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)],
            vec![
                rvllm_core::TokenId(2),
                rvllm_core::TokenId(4),
                rvllm_core::TokenId(2),
                rvllm_core::TokenId(17),
            ],
            vec![0, 2, 4],
            vec![1, 1],
            vec![2, 2],
        );
        let ticket = backend.launch_prefill(&prefill).unwrap_or_else(|err| {
            panic!("bounded E2B batch prefill failed at layer {stop_after_layer}: {err}")
        });
        let out = backend.collect(ticket).unwrap_or_else(|err| {
            panic!("bounded E2B batch prefill collect failed at layer {stop_after_layer}: {err}")
        });
        assert!(
            out.is_empty(),
            "bounded batch prefill must not sample logits"
        );

        let residual = backend.debug_read_residual_f32(4).unwrap_or_else(|err| {
            panic!("bounded E2B batch residual read failed at layer {stop_after_layer}: {err}")
        });
        assert_eq!(residual.len(), 4 * arch.hidden_size);
        let summary = finite_summary(&residual);
        eprintln!(
            "bounded E2B batch residual after layer {stop_after_layer}: finite={}/{} max_abs={:e} mean_abs={:e} first_nonfinite_index={:?}",
            summary.finite_count,
            summary.total_count,
            summary.max_abs,
            summary.mean_abs,
            summary.first_nonfinite_index
        );
        assert_eq!(
            summary.first_nonfinite_index, None,
            "first non-finite batch residual appears at or before layer {stop_after_layer}: {summary:?}"
        );
        assert_eq!(summary.finite_count, summary.total_count);
    }
    env_guard.remove(RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and /tmp/gemma4-e2b-hf-layer4-trace.json"]
fn real_gemma4_e2b_layer4_metal_trace_compares_to_hf_summary() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let trace_layer = std::env::var("RVLLM_E2B_TRACE_COMPARE_LAYER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(4);
    let prompt_token_ids = parse_e2b_trace_prompt_token_ids();
    let prompt_slug = e2b_trace_prompt_slug(&prompt_token_ids);
    let hf_trace_path = std::env::var_os("RVLLM_E2B_TRACE_COMPARE_HF_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            if prompt_token_ids.as_slice() == [2, 4] {
                std::path::PathBuf::from(format!(
                    "/tmp/gemma4-e2b-hf-layer{trace_layer}-trace.json"
                ))
            } else {
                std::path::PathBuf::from(format!(
                    "/tmp/gemma4-e2b-hf-prompt-{prompt_slug}-layer{trace_layer}-trace.json"
                ))
            }
        });
    if !hf_trace_path.exists() {
        eprintln!(
            "skipping: HF layer trace artifact is missing at {}",
            hf_trace_path.display()
        );
        return;
    }
    let metal_trace_path = std::env::var_os("RVLLM_E2B_TRACE_COMPARE_METAL_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            if prompt_token_ids.as_slice() == [2, 4] {
                std::path::PathBuf::from(format!(
                    "/tmp/gemma4-e2b-metal-layer{trace_layer}-trace.json"
                ))
            } else {
                std::path::PathBuf::from(format!(
                    "/tmp/gemma4-e2b-metal-prompt-{prompt_slug}-layer{trace_layer}-trace.json"
                ))
            }
        });
    let _ = fs::remove_file(&metal_trace_path);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before layer trace");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);

    let env_guard = MetalDebugEnvGuard::new(&[
        RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV,
        RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV,
        RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
        RVLLM_METAL_DEBUG_TRACE_LAYER_ENV,
        RVLLM_METAL_DEBUG_TRACE_JSON_ENV,
    ]);
    env_guard.set(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV, "1");
    env_guard.set(
        RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV,
        trace_layer.to_string(),
    );
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV, trace_layer.to_string());
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_JSON_ENV, &metal_trace_path);

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    backend
        .prepare(&plan)
        .expect("real Gemma4 E2B Metal prepare/load should complete before layer trace");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        prompt_token_ids
            .iter()
            .copied()
            .map(rvllm_core::TokenId)
            .collect(),
        vec![0, prompt_token_ids.len() as u32],
        vec![(prompt_token_ids.len() - 1) as u32],
        vec![prompt_token_ids.len() as u32],
    );
    let ticket = backend
        .launch_prefill(&prefill)
        .expect("bounded E2B layer trace prefill should launch");
    let out = backend
        .collect(ticket)
        .expect("bounded E2B layer trace prefill collect should succeed");
    assert!(out.is_empty(), "layer trace prefill must not sample logits");

    let hf_raw = fs::read_to_string(&hf_trace_path).expect("read HF layer trace");
    let metal_raw = fs::read_to_string(&metal_trace_path).expect("read Metal layer trace");
    let hf: Value = serde_json::from_str(&hf_raw).expect("parse HF layer trace");
    let metal: Value = serde_json::from_str(&metal_raw).expect("parse Metal layer trace");
    assert_eq!(
        hf["schema"].as_str(),
        Some("rvllm.gemma4_hf_layer_trace.v1")
    );
    assert_eq!(
        metal["schema"].as_str(),
        Some("rvllm.gemma4_metal_layer_trace.v1")
    );
    let expected_prompt_values = prompt_token_ids
        .iter()
        .map(|&token| Value::from(token))
        .collect::<Vec<_>>();
    assert_eq!(
        hf["prompt_token_ids"]
            .as_array()
            .expect("prompt ids")
            .as_slice(),
        expected_prompt_values.as_slice()
    );
    assert_eq!(hf["layer"].as_u64(), Some(trace_layer as u64));
    assert_eq!(metal["layer"].as_u64(), Some(trace_layer as u64));
    assert_eq!(metal["phase"].as_str(), Some("prefill"));

    for name in [
        "input_to_layer",
        "after_input_layernorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "after_q_norm",
        "after_k_norm",
        "after_v_norm",
        "after_rope_q",
        "after_rope_k",
        "attention_output",
        "after_o_proj",
        "after_post_attention_layernorm",
        "after_pre_feedforward_layernorm",
        "gate_up_out",
        "ffn_activation",
        "after_ffn_branch",
        "after_post_feedforward_layernorm",
        "per_layer_input",
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
        "final_residual_after_layer",
    ] {
        if !hf["summaries"][name].is_object() {
            eprintln!("E2B layer trace: skipping HF-missing summary {name}");
            continue;
        }
        compare_trace_summary_stats(&hf, &metal, name);
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and large Metal arena opt-in"]
fn real_gemma4_e2b_shared_kv_layers_13_to_16_decode_trace_explains_tail_cache_use() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before shared-KV trace");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);

    let trace_path_template =
        std::path::PathBuf::from("/tmp/gemma4-e2b-shared-kv-layer{layer}-decode-trace.json");
    for layer_idx in [13usize, 14, 15, 16] {
        let path = std::path::PathBuf::from(format!(
            "/tmp/gemma4-e2b-shared-kv-layer{layer_idx}-decode-trace.json"
        ));
        let _ = fs::remove_file(path);
    }

    let env_guard = MetalDebugEnvGuard::new(&[
        RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV,
        RVLLM_METAL_DEBUG_TRACE_LAYER_ENV,
        RVLLM_METAL_DEBUG_TRACE_JSON_ENV,
        RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV,
    ]);
    env_guard.set(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV, "1");
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV, "13,14,15,16");
    env_guard.set(RVLLM_METAL_DEBUG_TRACE_JSON_ENV, &trace_path_template);
    env_guard.set(RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV, "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    backend
        .prepare(&plan)
        .expect("real Gemma4 E2B Metal prepare/load should complete before shared-KV trace");

    let prefill = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
    );
    let prefill_ticket = backend
        .launch_prefill(&prefill)
        .expect("real E2B shared-KV trace prefill should launch");
    let prefill_out = backend
        .collect(prefill_ticket)
        .expect("real E2B shared-KV trace prefill collect should succeed");
    assert!(prefill_out.is_empty());

    let decode = rvllm_apple::HandoffCapsule::new(
        rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
        vec![rvllm_core::ReqId(1)],
        vec![rvllm_core::TokenId(4)],
        vec![0, 1],
        vec![1],
        vec![2],
    );
    let decode_ticket = backend
        .launch_rollout(&decode, None)
        .expect("real E2B shared-KV trace decode should launch");
    let decode_out = backend
        .collect(decode_ticket)
        .expect("real E2B shared-KV trace decode collect should succeed");
    assert_eq!(
        decode_out.len(),
        1,
        "debug skip-final-logits path must still return one placeholder token"
    );

    for (layer_idx, expected_source) in
        [(13usize, None), (14, None), (15, Some(13)), (16, Some(13))]
    {
        let path = std::path::PathBuf::from(format!(
            "/tmp/gemma4-e2b-shared-kv-layer{layer_idx}-decode-trace.json"
        ));
        let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!(
                "read real E2B shared-KV layer {layer_idx} trace at {}: {err}",
                path.display()
            )
        });
        let trace: Value = serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("parse shared-KV layer {layer_idx} trace: {err}"));
        assert_eq!(
            trace["schema"].as_str(),
            Some("rvllm.gemma4_metal_layer_trace.v1")
        );
        assert_eq!(trace["phase"].as_str(), Some("decode"));
        assert_eq!(trace["layer"].as_u64(), Some(layer_idx as u64));
        assert_eq!(
            trace["shared_kv_source_layer"]
                .as_u64()
                .map(|value| value as usize),
            expected_source,
            "unexpected shared-KV source layer for E2B layer {layer_idx}"
        );
        for name in [
            "q_projection",
            "k_projection",
            "v_projection",
            "after_q_norm",
            "after_k_norm",
            "after_v_norm",
            "after_rope_q",
            "after_rope_k",
            "attention_output",
            "final_residual_after_layer",
            "local_kv_cache_k",
            "local_kv_cache_v",
            "attention_kv_cache_k",
            "attention_kv_cache_v",
        ] {
            let summary = &trace["summaries"][name];
            assert!(
                summary.is_object(),
                "shared-KV layer {layer_idx} missing summary {name}"
            );
            let total = summary["total_count"]
                .as_u64()
                .unwrap_or_else(|| panic!("shared-KV layer {layer_idx} {name} total_count"));
            let finite = summary["finite_count"]
                .as_u64()
                .unwrap_or_else(|| panic!("shared-KV layer {layer_idx} {name} finite_count"));
            assert_eq!(
                finite, total,
                "shared-KV layer {layer_idx} {name} has non-finite values"
            );
            let max_abs = summary["max_abs"].as_f64().unwrap_or(0.0);
            let mean_abs = summary["mean_abs"].as_f64().unwrap_or(0.0);
            eprintln!(
                "E2B shared-KV trace layer {layer_idx} {name}: finite={finite}/{total} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
            );
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and large Metal arena opt-in"]
fn real_gemma4_e2b_shared_kv_baseline_vs_skip_trace_identifies_first_divergence() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before shared-KV paired trace");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);

    let modes = [
        "none",
        "skip_local_kv_cache_write_only",
        "skip_tail_kv_projection_and_cache",
    ];
    for mode in modes {
        eprintln!("E2B shared-KV paired trace: running mode={mode}");
        run_e2b_shared_kv_trace_mode(&model_dir, &arch, mode);
        for layer_idx in E2B_SHARED_KV_DIFF_TRACE_LAYERS {
            let trace = read_e2b_shared_kv_trace(mode, *layer_idx);
            assert_eq!(
                trace["schema"].as_str(),
                Some("rvllm.gemma4_metal_layer_trace.v1")
            );
            assert_eq!(trace["phase"].as_str(), Some("decode"));
            assert_eq!(trace["layer"].as_u64(), Some(*layer_idx as u64));
            assert_trace_summaries_finite(
                &trace,
                mode,
                *layer_idx,
                E2B_SHARED_KV_DIFF_TRACE_FIELDS,
            );
        }
    }

    for mode in [
        "skip_local_kv_cache_write_only",
        "skip_tail_kv_projection_and_cache",
    ] {
        let mut first_any = None;
        let mut first_downstream = None;
        for layer_idx in E2B_SHARED_KV_DIFF_TRACE_LAYERS {
            let baseline = read_e2b_shared_kv_trace("none", *layer_idx);
            let candidate = read_e2b_shared_kv_trace(mode, *layer_idx);
            if first_any.is_none() {
                first_any = first_trace_summary_diff(
                    &baseline,
                    &candidate,
                    *layer_idx,
                    mode,
                    E2B_SHARED_KV_DIFF_TRACE_FIELDS,
                );
            }
            if first_downstream.is_none() {
                first_downstream = first_trace_summary_diff(
                    &baseline,
                    &candidate,
                    *layer_idx,
                    mode,
                    E2B_SHARED_KV_DIFF_DOWNSTREAM_FIELDS,
                );
            }
        }
        match &first_any {
            Some(diff) => {
                eprintln!("E2B shared-KV paired trace mode={mode} first_any_diff: {diff}")
            }
            None => eprintln!(
                "E2B shared-KV paired trace mode={mode} first_any_diff: none through layers 13-20"
            ),
        }
        match &first_downstream {
            Some(diff) => {
                eprintln!("E2B shared-KV paired trace mode={mode} first_downstream_diff: {diff}")
            }
            None => eprintln!(
                "E2B shared-KV paired trace mode={mode} first_downstream_diff: none through layers 13-20; if selected logits previously failed, trace later layers before retrying the gate"
            ),
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prefill_decode_selected_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path = std::path::PathBuf::from("/tmp/gemma4-e2b-hf-reference-logits.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let (selected_token_ids, expected_logits, expected_next_token) =
        read_e2b_hf_reference_selected_logits(&reference_path);
    assert_eq!(selected_token_ids, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(expected_next_token, 954);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    let result = (|| -> Result<(Vec<f32>, Vec<f32>, Vec<rvllm_apple::StepToken>)> {
        backend.prepare(&plan)?;

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());

        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(4)],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let decode_ticket = backend.launch_rollout(&decode, None)?;
        let logits = backend.debug_read_decode_logits_f32(1)?;
        let residual = backend.debug_read_residual_f32(1)?;
        let out = backend.collect(decode_ticket)?;
        Ok((logits, residual, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits, metal_residual, out) =
        result.expect("real Gemma4 E2B prefill/decode should launch");
    assert_eq!(out.len(), 1);
    assert_eq!(metal_logits.len(), arch.vocab_size);
    assert_eq!(metal_residual.len(), arch.hidden_size);
    let residual_nonfinite = metal_residual.iter().filter(|v| !v.is_finite()).count();
    let logits_nonfinite = metal_logits.iter().filter(|v| !v.is_finite()).count();
    if residual_nonfinite > 0 || logits_nonfinite > 0 {
        eprintln!(
            "real E2B nonfinite summary: residual={} logits={}",
            residual_nonfinite, logits_nonfinite
        );
    }
    assert!(metal_logits.iter().all(|v| v.is_finite()));

    let selected_indices: Vec<usize> = selected_token_ids
        .iter()
        .map(|&token_id| token_id as usize)
        .collect();
    const FIRST_E2B_LOGIT_TOLERANCE: f32 = 1.0;
    for (&idx, &expected) in selected_indices.iter().zip(expected_logits.iter()) {
        eprintln!(
            "real E2B selected logit[{idx}]: metal={} hf={} delta={}",
            metal_logits[idx],
            expected,
            (metal_logits[idx] - expected).abs()
        );
        assert_f32_close(
            &format!("real E2B selected logit[{idx}]"),
            metal_logits[idx],
            expected,
            FIRST_E2B_LOGIT_TOLERANCE,
        );
    }
    assert_eq!(
        out[0].token_id,
        rvllm_core::TokenId(expected_next_token),
        "real E2B sampled token should match HF reference once selected logits are stable"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, full HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prefill_decode_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path =
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: full HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let reference = read_e2b_hf_reference_logits(&reference_path, &[2, 4], 1);
    let step = &reference.steps[0];
    assert_eq!(reference.prompt_token_ids, vec![2, 4]);
    assert_eq!(reference.generated_tokens, vec![954]);
    assert_eq!(step.selected_token_ids, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(step.next_token, 954);
    let expected_full_logits = step
        .full_logits
        .as_ref()
        .expect("HF reference artifact must include full logits");

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);
    assert_eq!(expected_full_logits.len(), arch.vocab_size);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);
    let result = (|| -> Result<(Vec<f32>, Vec<f32>, Vec<rvllm_apple::StepToken>)> {
        backend.prepare(&plan)?;

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());

        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(4)],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let decode_ticket = backend.launch_rollout(&decode, None)?;
        let logits = backend.debug_read_decode_logits_f32(1)?;
        let residual = backend.debug_read_residual_f32(1)?;
        let out = backend.collect(decode_ticket)?;
        Ok((logits, residual, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits, metal_residual, out) =
        result.expect("real Gemma4 E2B prefill/decode should launch");
    assert_eq!(out.len(), 1);
    assert_eq!(metal_logits.len(), arch.vocab_size);
    assert_eq!(metal_residual.len(), arch.hidden_size);
    assert!(metal_residual.iter().all(|v| v.is_finite()));
    assert!(metal_logits.iter().all(|v| v.is_finite()));

    for (&token_id, &expected) in step
        .selected_token_ids
        .iter()
        .zip(step.selected_logits.iter())
    {
        let idx = token_id as usize;
        eprintln!(
            "real E2B full-vocab guard selected logit[{idx}]: metal={} hf={} delta={}",
            metal_logits[idx],
            expected,
            (metal_logits[idx] - expected).abs()
        );
    }

    const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
    assert_e2b_full_vocab_logits_close(
        "real E2B full-vocab logits",
        &metal_logits,
        expected_full_logits,
        FULL_E2B_LOGIT_TOLERANCE,
    );
    assert_eq!(
        cpu_full_nonzero_argmax(&metal_logits),
        cpu_full_nonzero_argmax(expected_full_logits),
        "real E2B full-vocab argmax should match HF reference"
    );
    assert_eq!(
        out[0].token_id,
        rvllm_core::TokenId(step.next_token),
        "real E2B sampled token should match HF reference after full-vocab comparison"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, two-step full HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prefill_decode_two_steps_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path =
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: two-step full HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let reference = read_e2b_hf_reference_logits(&reference_path, &[2, 4], 2);
    assert_eq!(reference.prompt_token_ids, vec![2, 4]);
    assert_eq!(reference.steps.len(), 2);
    assert_eq!(reference.generated_tokens[0], 954);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before decode loop");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_step, out) = run_real_e2b_model_backend_decode_loop(
        model_dir,
        &arch,
        &reference.prompt_token_ids,
        reference.steps.len(),
    )
    .expect("real Gemma4 E2B two-step prefill/decode should launch");

    assert_eq!(metal_logits_by_step.len(), reference.steps.len());
    assert_eq!(out.len(), reference.steps.len());
    for (step_idx, (metal_logits, step)) in metal_logits_by_step
        .iter()
        .zip(reference.steps.iter())
        .enumerate()
    {
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);
        for (&token_id, &expected) in step
            .selected_token_ids
            .iter()
            .zip(step.selected_logits.iter())
        {
            let idx = token_id as usize;
            eprintln!(
                "real E2B two-step guard step={} selected logit[{idx}]: metal={} hf={} delta={}",
                step_idx,
                metal_logits[idx],
                expected,
                (metal_logits[idx] - expected).abs()
            );
        }

        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!("real E2B step {} full-vocab logits", step_idx + 1),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_eq!(
            cpu_full_nonzero_argmax(metal_logits),
            cpu_full_nonzero_argmax(expected_full_logits),
            "real E2B full-vocab argmax should match HF reference at step {}",
            step_idx + 1
        );
        assert_eq!(
            out[step_idx].token_id,
            rvllm_core::TokenId(step.next_token),
            "real E2B sampled token should match HF reference at step {}",
            step_idx + 1
        );
    }
    let generated = out
        .iter()
        .map(|token| token.token_id.raw())
        .collect::<Vec<_>>();
    assert_eq!(generated, reference.generated_tokens);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, broader-prompt full HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_broader_prompt_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path =
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: broader-prompt full HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let reference = read_e2b_hf_reference_logits(&reference_path, &[2, 17, 42, 4], 1);
    assert_eq!(reference.prompt_token_ids, vec![2, 17, 42, 4]);
    assert_eq!(reference.steps.len(), 1);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before broader prompt decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_step, out) = run_real_e2b_model_backend_decode_loop(
        model_dir,
        &arch,
        &reference.prompt_token_ids,
        reference.steps.len(),
    )
    .expect("real Gemma4 E2B broader prompt prefill/decode should launch");

    assert_eq!(metal_logits_by_step.len(), 1);
    assert_eq!(out.len(), 1);
    let step = &reference.steps[0];
    let expected_full_logits = step
        .full_logits
        .as_ref()
        .expect("HF reference artifact must include full logits");
    assert_eq!(expected_full_logits.len(), arch.vocab_size);
    let metal_logits = &metal_logits_by_step[0];
    for (&token_id, &expected) in step
        .selected_token_ids
        .iter()
        .zip(step.selected_logits.iter())
    {
        let idx = token_id as usize;
        eprintln!(
            "real E2B broader prompt selected logit[{idx}]: metal={} hf={} delta={}",
            metal_logits[idx],
            expected,
            (metal_logits[idx] - expected).abs()
        );
    }

    const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
    assert_e2b_full_vocab_logits_close(
        "real E2B broader prompt full-vocab logits",
        metal_logits,
        expected_full_logits,
        FULL_E2B_LOGIT_TOLERANCE,
    );
    assert_eq!(
        cpu_full_nonzero_argmax(metal_logits),
        cpu_full_nonzero_argmax(expected_full_logits),
        "real E2B broader prompt argmax should match HF reference"
    );
    assert_eq!(
        out[0].token_id,
        rvllm_core::TokenId(step.next_token),
        "real E2B broader prompt sampled token should match HF reference"
    );
    assert_eq!(vec![out[0].token_id.raw()], reference.generated_tokens);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, four-step full HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prefill_decode_four_steps_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path =
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps4.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: four-step full HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let reference = read_e2b_hf_reference_logits(&reference_path, &[2, 4], 4);
    assert_eq!(reference.prompt_token_ids, vec![2, 4]);
    assert_eq!(reference.steps.len(), 4);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before four-step decode loop");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_step, out) = run_real_e2b_model_backend_decode_loop(
        model_dir,
        &arch,
        &reference.prompt_token_ids,
        reference.steps.len(),
    )
    .expect("real Gemma4 E2B four-step prefill/decode should launch");

    assert_eq!(metal_logits_by_step.len(), reference.steps.len());
    assert_eq!(out.len(), reference.steps.len());
    for (step_idx, (metal_logits, step)) in metal_logits_by_step
        .iter()
        .zip(reference.steps.iter())
        .enumerate()
    {
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);
        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!(
                "real E2B four-step decode step {} full-vocab logits",
                step_idx + 1
            ),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_eq!(
            cpu_full_nonzero_argmax(metal_logits),
            cpu_full_nonzero_argmax(expected_full_logits),
            "real E2B four-step argmax should match HF reference at step {}",
            step_idx + 1
        );
        assert_eq!(
            out[step_idx].token_id,
            rvllm_core::TokenId(step.next_token),
            "real E2B four-step sampled token should match HF reference at step {}",
            step_idx + 1
        );
    }
    let generated = out
        .iter()
        .map(|token| token.token_id.raw())
        .collect::<Vec<_>>();
    assert_eq!(generated, reference.generated_tokens);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, eight-step full HF reference artifact, and large Metal arena opt-in"]
fn real_gemma4_e2b_model_backend_prefill_decode_eight_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference(
) {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_path =
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps8.json");
    if !reference_path.exists() {
        eprintln!(
            "skipping: eight-step full HF reference logits artifact is missing at {}",
            reference_path.display()
        );
        return;
    }

    let reference = read_e2b_hf_reference_logits(&reference_path, &[2, 4], 8);
    assert_eq!(reference.prompt_token_ids, vec![2, 4]);
    assert_eq!(reference.steps.len(), 8);

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before eight-step decode loop");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_step, out) =
        run_real_e2b_model_backend_decode_loop_with_forced_next_tokens(
            model_dir,
            &arch,
            &reference.prompt_token_ids,
            reference.steps.len(),
            Some(&reference.generated_tokens),
        )
        .expect("real Gemma4 E2B eight-step forced-token prefill/decode should launch");

    assert_eq!(metal_logits_by_step.len(), reference.steps.len());
    assert_eq!(out.len(), reference.steps.len());
    for (step_idx, (metal_logits, step)) in metal_logits_by_step
        .iter()
        .zip(reference.steps.iter())
        .enumerate()
    {
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);
        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!(
                "real E2B eight-step decode step {} full-vocab logits",
                step_idx + 1
            ),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_e2b_sample_matches_or_hf_tie(
            &format!("real E2B eight-step logit argmax at step {}", step_idx + 1),
            expected_full_logits,
            cpu_full_nonzero_argmax(expected_full_logits) as u32,
            cpu_full_nonzero_argmax(metal_logits) as u32,
        );
        assert_e2b_sample_matches_or_hf_tie(
            &format!("real E2B eight-step sampled token at step {}", step_idx + 1),
            expected_full_logits,
            step.next_token,
            out[step_idx].token_id.raw(),
        );
    }
    let sampled = out
        .iter()
        .map(|token| token.token_id.raw())
        .collect::<Vec<_>>();
    eprintln!(
        "real E2B eight-step forced-token run: sampled={sampled:?} forced_hf={:?}",
        reference.generated_tokens
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, batch full HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_batch_two_prefill_decode_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-step1.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: batch full HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 1))
        .collect::<Vec<_>>();
    let prompts = references
        .iter()
        .map(|reference| reference.prompt_token_ids.clone())
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before batch decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_seq, out) =
        run_real_e2b_model_backend_batch_one_step(model_dir, &arch, &prompts)
            .expect("real Gemma4 E2B batch prefill/decode should launch");

    assert_eq!(metal_logits_by_seq.len(), references.len());
    assert_eq!(out.len(), references.len());
    for (seq_idx, ((metal_logits, reference), token)) in metal_logits_by_seq
        .iter()
        .zip(references.iter())
        .zip(out.iter())
        .enumerate()
    {
        let step = &reference.steps[0];
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);
        for (&token_id, &expected) in step
            .selected_token_ids
            .iter()
            .zip(step.selected_logits.iter())
        {
            let idx = token_id as usize;
            eprintln!(
                "real E2B batch seq={seq_idx} selected logit[{idx}]: metal={} hf={} delta={}",
                metal_logits[idx],
                expected,
                (metal_logits[idx] - expected).abs()
            );
        }

        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!("real E2B batch seq {} full-vocab logits", seq_idx + 1),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_eq!(
            cpu_full_nonzero_argmax(metal_logits),
            cpu_full_nonzero_argmax(expected_full_logits),
            "real E2B batch seq {} argmax should match HF reference",
            seq_idx + 1
        );
        assert_eq!(
            token.token_id,
            rvllm_core::TokenId(step.next_token),
            "real E2B batch seq {} sampled token should match HF reference",
            seq_idx + 1
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, batch full HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_engine_batch_two_prefill_decode_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-step1.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: batch full HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 1))
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before Engine batch decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<crate::engine::StepOutput>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared real Gemma4 E2B model backend");

        for (req_idx, reference) in references.iter().enumerate() {
            let prompt = reference
                .prompt_token_ids
                .iter()
                .map(|&token| rvllm_core::TokenId(token))
                .collect::<Vec<_>>();
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId((req_idx + 1) as u64),
                prompt,
                1,
            ));
        }

        let prefill = engine.step_launch().expect("launch Engine batch prefill");
        match prefill.plan().expect("Engine prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill.collect()?.is_empty());

        let decode = engine.step_launch().expect("launch Engine batch decode");
        match decode.plan().expect("Engine decode plan") {
            crate::scheduler::BatchPlan::Decode {
                req_ids,
                bucket,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
                assert_eq!(*bucket, 2);
                assert_eq!(positions, &vec![1, 1]);
                assert_eq!(context_lens, &vec![2, 2]);
            }
            other => panic!("expected Engine Decode, got {other:?}"),
        }
        let out = decode.collect()?;
        assert_eq!(out.len(), references.len());
        assert!(!engine.has_pending_work());

        let flat_logits = shared_backend.debug_read_decode_logits_f32(references.len())?;
        assert_eq!(flat_logits.len(), references.len() * arch.vocab_size);
        let logits_by_seq = flat_logits
            .chunks_exact(arch.vocab_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        Ok((logits_by_seq, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits_by_seq, out) =
        result.expect("Engine real Gemma4 E2B batch prefill/decode should launch");
    assert_eq!(metal_logits_by_seq.len(), references.len());
    assert_eq!(out.len(), references.len());
    for (seq_idx, ((metal_logits, reference), token)) in metal_logits_by_seq
        .iter()
        .zip(references.iter())
        .zip(out.iter())
        .enumerate()
    {
        let step = &reference.steps[0];
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);

        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!(
                "real E2B Engine batch seq {} full-vocab logits",
                seq_idx + 1
            ),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_eq!(
            cpu_full_nonzero_argmax(metal_logits),
            cpu_full_nonzero_argmax(expected_full_logits),
            "real E2B Engine batch seq {} argmax should match HF reference",
            seq_idx + 1
        );
        assert_eq!(token.req_id, rvllm_core::ReqId((seq_idx + 1) as u64));
        assert_eq!(
            token.new_token,
            rvllm_core::TokenId(step.next_token),
            "real E2B Engine batch seq {} sampled token should match HF reference",
            seq_idx + 1
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, mixed-length batch HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_engine_batch_two_mixed_prompt_lengths_full_vocab_logits_match_hf_reference() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17, 42, 4]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: Engine mixed-length batch HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 1))
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before Engine mixed-length batch decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<crate::engine::StepOutput>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared real Gemma4 E2B model backend");

        for (req_idx, reference) in references.iter().enumerate() {
            let prompt = reference
                .prompt_token_ids
                .iter()
                .map(|&token| rvllm_core::TokenId(token))
                .collect::<Vec<_>>();
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId((req_idx + 1) as u64),
                prompt,
                1,
            ));
        }

        let prefill = engine
            .step_launch()
            .expect("launch Engine mixed-length batch prefill");
        match prefill.plan().expect("Engine prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill.collect()?.is_empty());

        let decode = engine
            .step_launch()
            .expect("launch Engine mixed-length batch decode");
        match decode.plan().expect("Engine decode plan") {
            crate::scheduler::BatchPlan::Decode {
                req_ids,
                bucket,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
                assert_eq!(*bucket, 2);
                assert_eq!(positions, &vec![1, 3]);
                assert_eq!(context_lens, &vec![2, 4]);
            }
            other => panic!("expected Engine Decode, got {other:?}"),
        }
        let out = decode.collect()?;
        assert_eq!(out.len(), references.len());
        assert!(!engine.has_pending_work());

        let flat_logits = shared_backend.debug_read_decode_logits_f32(references.len())?;
        assert_eq!(flat_logits.len(), references.len() * arch.vocab_size);
        let logits_by_seq = flat_logits
            .chunks_exact(arch.vocab_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        Ok((logits_by_seq, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits_by_seq, out) =
        result.expect("Engine real Gemma4 E2B mixed-length batch decode should launch");
    assert_eq!(metal_logits_by_seq.len(), references.len());
    assert_eq!(out.len(), references.len());
    for (seq_idx, ((metal_logits, reference), token)) in metal_logits_by_seq
        .iter()
        .zip(references.iter())
        .zip(out.iter())
        .enumerate()
    {
        let step = &reference.steps[0];
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);

        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!(
                "real E2B Engine mixed-length batch seq {} full-vocab logits",
                seq_idx + 1
            ),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_e2b_sample_matches_or_hf_tie(
            &format!(
                "real E2B Engine mixed-length batch seq {} logit argmax",
                seq_idx + 1
            ),
            expected_full_logits,
            cpu_full_nonzero_argmax(expected_full_logits) as u32,
            cpu_full_nonzero_argmax(metal_logits) as u32,
        );
        assert_eq!(token.req_id, rvllm_core::ReqId((seq_idx + 1) as u64));
        assert_e2b_sample_matches_or_hf_tie(
            &format!(
                "real E2B Engine mixed-length batch seq {} sampled token",
                seq_idx + 1
            ),
            expected_full_logits,
            step.next_token,
            token.new_token.raw(),
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, mixed-length two-step batch HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_engine_batch_two_mixed_prompt_lengths_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference(
) {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-steps2.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17, 42, 4]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: Engine mixed-length two-step batch HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 2))
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before Engine mixed-length two-step decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<crate::engine::StepOutput>>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared real Gemma4 E2B model backend");

        for (req_idx, reference) in references.iter().enumerate() {
            let prompt = reference
                .prompt_token_ids
                .iter()
                .map(|&token| rvllm_core::TokenId(token))
                .collect::<Vec<_>>();
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId((req_idx + 1) as u64),
                prompt,
                2,
            ));
        }

        let prefill = engine
            .step_launch()
            .expect("launch Engine mixed-length two-step batch prefill");
        match prefill.plan().expect("Engine prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill.collect()?.is_empty());

        let expected_positions = [vec![1, 3], vec![2, 4]];
        let expected_context_lens = [vec![2, 4], vec![3, 5]];
        let mut logits_by_step = Vec::new();
        let mut outputs_by_step = Vec::new();
        for step_idx in 0..2usize {
            let decode = engine
                .step_launch()
                .expect("launch Engine mixed-length two-step batch decode");
            match decode.plan().expect("Engine decode plan") {
                crate::scheduler::BatchPlan::Decode {
                    req_ids,
                    bucket,
                    positions,
                    context_lens,
                    ..
                } => {
                    assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
                    assert_eq!(*bucket, 2);
                    assert_eq!(positions, &expected_positions[step_idx]);
                    assert_eq!(context_lens, &expected_context_lens[step_idx]);
                }
                other => panic!("expected Engine Decode, got {other:?}"),
            }
            let out = decode.collect()?;
            assert_eq!(out.len(), references.len());

            let flat_logits = shared_backend.debug_read_decode_logits_f32(references.len())?;
            assert_eq!(flat_logits.len(), references.len() * arch.vocab_size);
            let step_logits = flat_logits
                .chunks_exact(arch.vocab_size)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            logits_by_step.push(step_logits);
            outputs_by_step.push(out);
        }
        assert!(!engine.has_pending_work());
        Ok((logits_by_step, outputs_by_step))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits_by_step, out_by_step) =
        result.expect("Engine real Gemma4 E2B mixed-length two-step batch decode should launch");
    assert_eq!(metal_logits_by_step.len(), 2);
    assert_eq!(out_by_step.len(), 2);
    for step_idx in 0..2usize {
        assert_eq!(metal_logits_by_step[step_idx].len(), references.len());
        assert_eq!(out_by_step[step_idx].len(), references.len());
        for seq_idx in 0..references.len() {
            let reference_step = &references[seq_idx].steps[step_idx];
            let expected_full_logits = reference_step
                .full_logits
                .as_ref()
                .expect("HF reference artifact must include full logits");
            assert_eq!(expected_full_logits.len(), arch.vocab_size);
            const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
            assert_e2b_full_vocab_logits_close(
                &format!(
                    "real E2B Engine mixed-length batch seq {} two-step decode step {} full-vocab logits",
                    seq_idx + 1,
                    step_idx + 1
                ),
                &metal_logits_by_step[step_idx][seq_idx],
                expected_full_logits,
                FULL_E2B_LOGIT_TOLERANCE,
            );
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B Engine mixed-length batch seq {} two-step logit argmax at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                cpu_full_nonzero_argmax(expected_full_logits) as u32,
                cpu_full_nonzero_argmax(&metal_logits_by_step[step_idx][seq_idx]) as u32,
            );
            let token = &out_by_step[step_idx][seq_idx];
            assert_eq!(token.req_id, rvllm_core::ReqId((seq_idx + 1) as u64));
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B Engine mixed-length batch seq {} two-step sampled token at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                reference_step.next_token,
                token.new_token.raw(),
            );
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, batch-three HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_engine_batch_three_mixed_prompt_lengths_one_step_full_vocab_logits_match_hf_reference(
) {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-step1.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json"),
    ];
    let expected_prompts: [&[u32]; 3] = [&[2, 4], &[2, 17], &[2, 17, 42, 4]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: Engine batch-three HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 1))
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before Engine batch-three decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<crate::engine::StepOutput>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared real Gemma4 E2B model backend");

        for (req_idx, reference) in references.iter().enumerate() {
            let prompt = reference
                .prompt_token_ids
                .iter()
                .map(|&token| rvllm_core::TokenId(token))
                .collect::<Vec<_>>();
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId((req_idx + 1) as u64),
                prompt,
                1,
            ));
        }

        let prefill = engine
            .step_launch()
            .expect("launch Engine batch-three prefill");
        match prefill.plan().expect("Engine prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(
                    req_ids,
                    &vec![
                        rvllm_core::ReqId(1),
                        rvllm_core::ReqId(2),
                        rvllm_core::ReqId(3)
                    ]
                );
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill.collect()?.is_empty());

        let decode = engine
            .step_launch()
            .expect("launch Engine batch-three decode");
        match decode.plan().expect("Engine decode plan") {
            crate::scheduler::BatchPlan::Decode {
                req_ids,
                bucket,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(
                    req_ids,
                    &vec![
                        rvllm_core::ReqId(1),
                        rvllm_core::ReqId(2),
                        rvllm_core::ReqId(3)
                    ]
                );
                assert_eq!(*bucket, 4);
                assert_eq!(positions, &vec![1, 1, 3]);
                assert_eq!(context_lens, &vec![2, 2, 4]);
            }
            other => panic!("expected Engine Decode, got {other:?}"),
        }
        let out = decode.collect()?;
        assert_eq!(out.len(), references.len());
        assert!(!engine.has_pending_work());

        let flat_logits = shared_backend.debug_read_decode_logits_f32(references.len())?;
        assert_eq!(flat_logits.len(), references.len() * arch.vocab_size);
        let logits_by_seq = flat_logits
            .chunks_exact(arch.vocab_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        Ok((logits_by_seq, out))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits_by_seq, out) =
        result.expect("Engine real Gemma4 E2B batch-three decode should launch");
    assert_eq!(metal_logits_by_seq.len(), references.len());
    assert_eq!(out.len(), references.len());
    for (seq_idx, ((metal_logits, reference), token)) in metal_logits_by_seq
        .iter()
        .zip(references.iter())
        .zip(out.iter())
        .enumerate()
    {
        let step = &reference.steps[0];
        let expected_full_logits = step
            .full_logits
            .as_ref()
            .expect("HF reference artifact must include full logits");
        assert_eq!(expected_full_logits.len(), arch.vocab_size);

        const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
        assert_e2b_full_vocab_logits_close(
            &format!(
                "real E2B Engine batch-three seq {} full-vocab logits",
                seq_idx + 1
            ),
            metal_logits,
            expected_full_logits,
            FULL_E2B_LOGIT_TOLERANCE,
        );
        assert_e2b_sample_matches_or_hf_tie(
            &format!(
                "real E2B Engine batch-three seq {} logit argmax",
                seq_idx + 1
            ),
            expected_full_logits,
            cpu_full_nonzero_argmax(expected_full_logits) as u32,
            cpu_full_nonzero_argmax(metal_logits) as u32,
        );
        assert_eq!(token.req_id, rvllm_core::ReqId((seq_idx + 1) as u64));
        assert_e2b_sample_matches_or_hf_tie(
            &format!(
                "real E2B Engine batch-three seq {} sampled token",
                seq_idx + 1
            ),
            expected_full_logits,
            step.next_token,
            token.new_token.raw(),
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, batch two-step HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_engine_batch_two_prefill_decode_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference(
) {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-steps2.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: Engine batch two-step HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 2))
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before Engine batch two-step decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let result = (|| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<crate::engine::StepOutput>>)> {
        let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
        plan.ane_hidden_size = arch.hidden_size;
        plan.ane_intermediate_size = arch.intermediate_size;
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared real Gemma4 E2B model backend");

        for (req_idx, reference) in references.iter().enumerate() {
            let prompt = reference
                .prompt_token_ids
                .iter()
                .map(|&token| rvllm_core::TokenId(token))
                .collect::<Vec<_>>();
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId((req_idx + 1) as u64),
                prompt,
                2,
            ));
        }

        let prefill = engine
            .step_launch()
            .expect("launch Engine batch two-step prefill");
        match prefill.plan().expect("Engine prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill.collect()?.is_empty());

        let mut logits_by_step = Vec::new();
        let mut outputs_by_step = Vec::new();
        for step_idx in 0..2usize {
            let decode = engine
                .step_launch()
                .expect("launch Engine batch two-step decode");
            match decode.plan().expect("Engine decode plan") {
                crate::scheduler::BatchPlan::Decode {
                    req_ids,
                    bucket,
                    positions,
                    context_lens,
                    ..
                } => {
                    assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
                    assert_eq!(*bucket, 2);
                    assert_eq!(positions, &vec![1 + step_idx as u32, 1 + step_idx as u32]);
                    assert_eq!(
                        context_lens,
                        &vec![2 + step_idx as u32, 2 + step_idx as u32]
                    );
                }
                other => panic!("expected Engine Decode, got {other:?}"),
            }
            let out = decode.collect()?;
            assert_eq!(out.len(), references.len());

            let flat_logits = shared_backend.debug_read_decode_logits_f32(references.len())?;
            assert_eq!(flat_logits.len(), references.len() * arch.vocab_size);
            let step_logits = flat_logits
                .chunks_exact(arch.vocab_size)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            logits_by_step.push(step_logits);
            outputs_by_step.push(out);
        }
        assert!(!engine.has_pending_work());
        Ok((logits_by_step, outputs_by_step))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (metal_logits_by_step, out_by_step) =
        result.expect("Engine real Gemma4 E2B batch two-step decode should launch");
    assert_eq!(metal_logits_by_step.len(), 2);
    assert_eq!(out_by_step.len(), 2);
    for step_idx in 0..2usize {
        assert_eq!(metal_logits_by_step[step_idx].len(), references.len());
        assert_eq!(out_by_step[step_idx].len(), references.len());
        for seq_idx in 0..references.len() {
            let reference_step = &references[seq_idx].steps[step_idx];
            let expected_full_logits = reference_step
                .full_logits
                .as_ref()
                .expect("HF reference artifact must include full logits");
            assert_eq!(expected_full_logits.len(), arch.vocab_size);
            const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
            assert_e2b_full_vocab_logits_close(
                &format!(
                    "real E2B Engine batch seq {} two-step decode step {} full-vocab logits",
                    seq_idx + 1,
                    step_idx + 1
                ),
                &metal_logits_by_step[step_idx][seq_idx],
                expected_full_logits,
                FULL_E2B_LOGIT_TOLERANCE,
            );
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B Engine batch seq {} two-step logit argmax at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                cpu_full_nonzero_argmax(expected_full_logits) as u32,
                cpu_full_nonzero_argmax(&metal_logits_by_step[step_idx][seq_idx]) as u32,
            );
            let token = &out_by_step[step_idx][seq_idx];
            assert_eq!(token.req_id, rvllm_core::ReqId((seq_idx + 1) as u64));
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B Engine batch seq {} two-step sampled token at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                reference_step.next_token,
                token.new_token.raw(),
            );
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory, batch two-step HF reference artifacts, and large Metal arena opt-in"]
fn real_gemma4_e2b_batch_two_prefill_decode_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference(
) {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let reference_paths = [
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json"),
        std::path::PathBuf::from("/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-steps2.json"),
    ];
    let expected_prompts: [&[u32]; 2] = [&[2, 4], &[2, 17]];
    if let Some(missing) = reference_paths.iter().find(|path| !path.exists()) {
        eprintln!(
            "skipping: batch two-step HF reference logits artifact is missing at {}",
            missing.display()
        );
        return;
    }

    let references = reference_paths
        .iter()
        .zip(expected_prompts.iter())
        .map(|(path, expected)| read_e2b_hf_reference_logits(path, expected, 2))
        .collect::<Vec<_>>();
    let prompts = references
        .iter()
        .map(|reference| reference.prompt_token_ids.clone())
        .collect::<Vec<_>>();
    let forced_tokens = references
        .iter()
        .map(|reference| reference.generated_tokens.clone())
        .collect::<Vec<_>>();

    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before batch two-step decode");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let (metal_logits_by_step, out_by_step) =
        run_real_e2b_model_backend_batch_decode_loop_with_forced_next_tokens(
            model_dir,
            &arch,
            &prompts,
            2,
            &forced_tokens,
        )
        .expect("real Gemma4 E2B batch two-step forced-token prefill/decode should launch");

    assert_eq!(metal_logits_by_step.len(), 2);
    assert_eq!(out_by_step.len(), 2);
    for step_idx in 0..2usize {
        assert_eq!(metal_logits_by_step[step_idx].len(), references.len());
        assert_eq!(out_by_step[step_idx].len(), references.len());
        for seq_idx in 0..references.len() {
            let reference_step = &references[seq_idx].steps[step_idx];
            let expected_full_logits = reference_step
                .full_logits
                .as_ref()
                .expect("HF reference artifact must include full logits");
            assert_eq!(expected_full_logits.len(), arch.vocab_size);
            const FULL_E2B_LOGIT_TOLERANCE: f32 = 1.0;
            assert_e2b_full_vocab_logits_close(
                &format!(
                    "real E2B batch seq {} two-step decode step {} full-vocab logits",
                    seq_idx + 1,
                    step_idx + 1
                ),
                &metal_logits_by_step[step_idx][seq_idx],
                expected_full_logits,
                FULL_E2B_LOGIT_TOLERANCE,
            );
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B batch seq {} two-step logit argmax at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                cpu_full_nonzero_argmax(expected_full_logits) as u32,
                cpu_full_nonzero_argmax(&metal_logits_by_step[step_idx][seq_idx]) as u32,
            );
            assert_e2b_sample_matches_or_hf_tie(
                &format!(
                    "real E2B batch seq {} two-step sampled token at step {}",
                    seq_idx + 1,
                    step_idx + 1
                ),
                expected_full_logits,
                reference_step.next_token,
                out_by_step[step_idx][seq_idx].token_id.raw(),
            );
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
const RVLLM_E2B_PROFILE_JSON_ENV: &str = "RVLLM_E2B_PROFILE_JSON";
#[cfg(all(feature = "apple", target_os = "macos"))]
const RVLLM_E2B_PROFILE_SAMPLE_ID_ENV: &str = "RVLLM_E2B_PROFILE_SAMPLE_ID";

#[cfg(all(feature = "apple", target_os = "macos"))]
fn real_e2b_probe_profile_artifact_json(
    stats: MetalProbePerfStats,
    prepare_ms: u128,
    prefill_ms: u128,
    decode_ms: u128,
    debug_sync: bool,
) -> Value {
    let decode_seconds = (decode_ms as f64 / 1000.0).max(f64::EPSILON);
    let prefill_seconds = (prefill_ms as f64 / 1000.0).max(f64::EPSILON);
    let decode_tok_s = stats.decode_steps as f64 / decode_seconds;
    let prefill_tok_s = 2.0 / prefill_seconds;
    let command_buffers_per_token = stats.command_buffers as f64 / stats.tokens as f64;

    let sample_id = std::env::var(RVLLM_E2B_PROFILE_SAMPLE_ID_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "real-e2b-metal-probe-json-artifact".to_string());
    let mut sample = rvllm_apple::BackendProfileSample::new(
        sample_id,
        rvllm_apple::BenchmarkCategory::MetalOnly,
        "rvllm-runtime-model-metal-backend-probe",
        "google/gemma-4-E2B",
        rvllm_apple::BackendProfileMetrics {
            first_token_latency_ms: rvllm_apple::OptionalMetric::unmeasured(
                "probe artifact records aggregate prefill wall time, not isolated first-token latency",
            ),
            steady_decode_tokens_per_second: rvllm_apple::OptionalMetric::measured(decode_tok_s),
            prefill_tokens_per_second: rvllm_apple::OptionalMetric::measured(prefill_tok_s),
            memory_peak_bytes: rvllm_apple::OptionalMetric::unmeasured(
                "probe artifact has no external peak RSS/GPU memory profiler capture",
            ),
            command_buffers_per_token: rvllm_apple::OptionalMetric::measured(
                command_buffers_per_token,
            ),
            cpu_utilization_percent: rvllm_apple::OptionalMetric::unmeasured(
                "probe artifact has no Instruments or powermetrics CPU utilization capture",
            ),
            gpu_utilization_percent: rvllm_apple::OptionalMetric::unmeasured(
                "probe artifact has no Metal System Trace GPU utilization capture",
            ),
            ane_utilization_percent: rvllm_apple::OptionalMetric::unsupported(
                "real E2B probe currently uses Metal only; private ANE execution is not established",
            ),
            energy_joules: rvllm_apple::OptionalMetric::unmeasured(
                "probe artifact has no powermetrics energy capture",
            ),
        },
    );
    sample.prompt_tokens = 2;
    sample.generated_tokens = stats.decode_steps;

    serde_json::json!({
        "schema_version": 1,
        "artifact_kind": "rvllm-real-e2b-metal-probe-profile",
        "generated_by": "real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters",
        "sample": sample,
        "timings_ms": {
            "prepare": prepare_ms,
            "prefill": prefill_ms,
            "decode": decode_ms,
        },
        "metal_probe_counters": {
            "prefill_steps": stats.prefill_steps,
            "decode_steps": stats.decode_steps,
            "tokens": stats.tokens,
            "library_compiles": stats.library_compiles,
            "pipeline_state_compiles": stats.pipeline_state_compiles,
            "command_buffers": stats.command_buffers,
            "encoders": stats.encoders,
            "embedding_encoders": stats.embedding_encoders,
            "ple_encoders": stats.ple_encoders,
            "layer_encoders": stats.layer_encoders,
            "layer_scale_encoder_fusions": stats.layer_scale_encoder_fusions,
            "final_sample_encoders": stats.final_sample_encoders,
            "final_logits_encoders": stats.final_logits_encoders,
            "encoder_counts_by_kernel_family": {
                "embedding": stats.embedding_encoders,
                "ple_input": stats.ple_encoders,
                "layer_body": stats.layer_encoders,
                "layer_scale_fused": stats.layer_scale_encoder_fusions,
                "final_sample": stats.final_sample_encoders,
                "final_logits_diagnostic": stats.final_logits_encoders,
            },
            "forced_waits": stats.forced_waits,
            "cpu_wall_ns": stats.cpu_wall_ns,
            "cpu_encode_ns": stats.cpu_encode_ns,
            "command_buffer_wait_ns": stats.command_buffer_wait_ns,
            "last_step_tokens": stats.last_step_tokens,
            "last_step_command_buffers": stats.last_step_command_buffers,
            "last_step_encoders": stats.last_step_encoders,
            "last_step_forced_waits": stats.last_step_forced_waits,
            "last_step_cpu_wall_ns": stats.last_step_cpu_wall_ns,
            "last_step_cpu_encode_ns": stats.last_step_cpu_encode_ns,
            "last_step_command_buffer_wait_ns": stats.last_step_command_buffer_wait_ns,
        },
        "debug_sync": debug_sync,
        "claim_boundary": "single-host probe artifact; not production performance, ANE, or regression evidence",
    })
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn real_e2b_probe_profile_artifact_schema_records_unmeasured_slots() {
    let stats = MetalProbePerfStats {
        last_step_gpu_execution_ns: None,
        prefill_steps: 1,
        decode_steps: 4,
        tokens: 6,
        library_compiles: 1,
        pipeline_state_compiles: kernels::KERNEL_COUNT as u64,
        command_buffers: 5,
        encoders: 3187,
        embedding_encoders: 5,
        ple_encoders: 20,
        layer_encoders: 3150,
        layer_scale_encoder_fusions: 0,
        final_sample_encoders: 12,
        final_logits_encoders: 0,
        forced_waits: 5,
        cpu_wall_ns: 123,
        cpu_encode_ns: 100,
        command_buffer_wait_ns: 23,
        last_step_tokens: 1,
        last_step_command_buffers: 1,
        last_step_encoders: 700,
        last_step_forced_waits: 1,
        last_step_cpu_wall_ns: 456,
        last_step_cpu_encode_ns: 400,
        last_step_command_buffer_wait_ns: 56,
    };

    let artifact = real_e2b_probe_profile_artifact_json(stats, 478_027, 586, 1_767, false);
    assert_eq!(artifact["schema_version"], 1);
    assert_eq!(
        artifact["artifact_kind"],
        "rvllm-real-e2b-metal-probe-profile"
    );
    assert_eq!(
        artifact["claim_boundary"],
        "single-host probe artifact; not production performance, ANE, or regression evidence"
    );
    assert_eq!(artifact["metal_probe_counters"]["command_buffers"], 5);
    assert_eq!(artifact["sample"]["prompt_tokens"], 2);
    assert_eq!(artifact["sample"]["generated_tokens"], 4);
    assert!(
        artifact["sample"]["metrics"]["memory_peak_bytes"]
            .get("Unmeasured")
            .is_some(),
        "profile artifact must keep missing external memory profiling explicit"
    );
    assert!(
        artifact["sample"]["metrics"]["gpu_utilization_percent"]
            .get("Unmeasured")
            .is_some(),
        "profile artifact must keep missing external GPU profiling explicit"
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and large Metal arena opt-in"]
fn real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before profiling harness");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);

    let prepare_start = Instant::now();
    let result = (|| -> Result<(Vec<rvllm_core::TokenId>, MetalProbePerfStats, u128, u128, u128)> {
        backend.prepare(&plan)?;
        let prepare_ms = prepare_start.elapsed().as_millis();

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_start = Instant::now();
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        let prefill_ms = prefill_start.elapsed().as_millis();
        assert!(prefill_out.is_empty());

        let mut current = rvllm_core::TokenId(4);
        let mut generated = Vec::new();
        let decode_start = Instant::now();
        for step_idx in 0..4usize {
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(1)],
                vec![current],
                vec![0, 1],
                vec![1 + step_idx as u32],
                vec![2 + step_idx as u32],
            );
            let ticket = backend.launch_rollout(&decode, None)?;
            let out = backend.collect(ticket)?;
            assert_eq!(out.len(), 1);
            current = out[0].token_id;
            generated.push(current);
        }
        let decode_ms = decode_start.elapsed().as_millis();
        Ok((
            generated,
            backend.probe_perf_stats(),
            prepare_ms,
            prefill_ms,
            decode_ms,
        ))
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    let (generated, stats, prepare_ms, prefill_ms, decode_ms) =
        result.expect("real Gemma4 E2B profiling harness should run");
    assert_eq!(
        generated,
        vec![
            rvllm_core::TokenId(954),
            rvllm_core::TokenId(1289),
            rvllm_core::TokenId(236813),
            rvllm_core::TokenId(655),
        ]
    );
    assert_eq!(stats.prefill_steps, 1);
    assert_eq!(stats.decode_steps, 4);
    assert_eq!(stats.tokens, 6);
    assert!(stats.command_buffers > 0);
    assert_eq!(
        stats.command_buffers,
        stats.prefill_steps + stats.decode_steps,
        "combined E2B probe submission should use one command buffer per prefill/decode step"
    );
    assert_eq!(stats.library_compiles, 1);
    assert_eq!(stats.pipeline_state_compiles, kernels::KERNEL_COUNT as u64);
    assert!(stats.encoders > 0);
    assert!(stats.forced_waits > 0);
    assert!(stats.cpu_wall_ns > 0);

    let decode_seconds = (decode_ms as f64 / 1000.0).max(f64::EPSILON);
    let decode_tok_s = stats.decode_steps as f64 / decode_seconds;
    let command_buffers_per_token = stats.command_buffers as f64 / stats.tokens as f64;
    let encoders_per_token = stats.encoders as f64 / stats.tokens as f64;
    eprintln!(
        "real E2B profile probe: prepare_ms={prepare_ms} prefill_ms={prefill_ms} decode_ms={decode_ms} decode_tok_s={decode_tok_s:.4} total_tokens={} library_compiles={} pipeline_state_compiles={} command_buffers={} encoders={} forced_waits={} command_buffers_per_token={command_buffers_per_token:.4} encoders_per_token={encoders_per_token:.4} last_step_cpu_wall_ns={} debug_sync={}",
        stats.tokens,
        stats.library_compiles,
        stats.pipeline_state_compiles,
        stats.command_buffers,
        stats.encoders,
        stats.forced_waits,
        stats.last_step_cpu_wall_ns,
        metal_debug_sync_enabled()
    );

    if let Some(path) = std::env::var_os(RVLLM_E2B_PROFILE_JSON_ENV) {
        let artifact = real_e2b_probe_profile_artifact_json(
            stats,
            prepare_ms,
            prefill_ms,
            decode_ms,
            metal_debug_sync_enabled(),
        );
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create E2B profile artifact parent dir");
        }
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&artifact).expect("serialize E2B profile artifact"),
        )
        .expect("write E2B profile artifact");
        eprintln!("wrote real E2B profile artifact to {}", path.display());
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
#[ignore = "requires cached Gemma4 E2B model directory and large Metal arena opt-in"]
fn real_gemma4_e2b_arena_and_pipeline_counters_do_not_change_after_rollout() {
    let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
        eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
        .expect("real Gemma4 E2B arch should parse before hot-path invariant test");
    assert_eq!(arch.num_hidden_layers, 35);
    assert_eq!(arch.hidden_size, 1536);
    assert_eq!(arch.vocab_size, 262144);

    let previous_large_probe_opt_in = std::env::var_os("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");

    let mut plan = n_layer_plan(model_dir.clone(), arch.num_hidden_layers);
    plan.ane_hidden_size = arch.hidden_size;
    plan.ane_intermediate_size = arch.intermediate_size;
    let mut backend = ModelMetalBackend::new(model_dir);

    let result = (|| -> Result<()> {
        backend.prepare(&plan)?;
        let after_prepare_stats = backend.probe_perf_stats();
        let after_prepare_arena = backend
            .probe_arena_stats()
            .expect("arena stats after real E2B prepare");
        assert_eq!(after_prepare_stats.library_compiles, 1);
        assert_eq!(
            after_prepare_stats.pipeline_state_compiles,
            kernels::KERNEL_COUNT as u64
        );
        assert_eq!(after_prepare_stats.command_buffers, 0);
        assert_eq!(after_prepare_stats.encoders, 0);
        assert!(after_prepare_arena.region_count > 0);
        assert!(after_prepare_arena.allocated_bytes > 0);
        assert!(after_prepare_arena.capacity_bytes >= after_prepare_arena.allocated_bytes);

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill)?;
        let prefill_out = backend.collect(prefill_ticket)?;
        assert!(prefill_out.is_empty());
        let after_prefill_stats = backend.probe_perf_stats();
        let after_prefill_arena = backend
            .probe_arena_stats()
            .expect("arena stats after real E2B prefill");
        assert_eq!(
            after_prefill_stats.library_compiles, after_prepare_stats.library_compiles,
            "real E2B prefill must not compile a Metal library after prepare"
        );
        assert_eq!(
            after_prefill_stats.pipeline_state_compiles,
            after_prepare_stats.pipeline_state_compiles,
            "real E2B prefill must not compile PSOs after prepare"
        );
        assert_eq!(
            after_prefill_arena, after_prepare_arena,
            "real E2B prefill must reuse the prepared Metal arena"
        );

        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(4)],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let decode_ticket = backend.launch_rollout(&decode, None)?;
        let decode_out = backend.collect(decode_ticket)?;
        assert_eq!(decode_out.len(), 1);
        assert_eq!(decode_out[0].token_id, rvllm_core::TokenId(954));
        let after_decode_stats = backend.probe_perf_stats();
        let after_decode_arena = backend
            .probe_arena_stats()
            .expect("arena stats after real E2B decode");
        assert_eq!(
            after_decode_stats.library_compiles, after_prepare_stats.library_compiles,
            "real E2B decode must not compile a Metal library after prepare"
        );
        assert_eq!(
            after_decode_stats.pipeline_state_compiles, after_prepare_stats.pipeline_state_compiles,
            "real E2B decode must not compile PSOs after prepare"
        );
        assert_eq!(
            after_decode_arena, after_prepare_arena,
            "real E2B decode must reuse the prepared Metal arena"
        );
        Ok(())
    })();

    if let Some(previous) = previous_large_probe_opt_in {
        std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", previous);
    } else {
        std::env::remove_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE");
    }

    result.expect("real Gemma4 E2B hot-path arena and pipeline invariant test should run");
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_in_flight_ring_holds_three_distinct_tickets() {
    let mut ring = ModelInFlightRing::default();
    let mut tickets = Vec::new();
    for step_id in 0..MODEL_METAL_IN_FLIGHT_SLOTS as u64 {
        let ticket = ring
            .reserve(step_id, AppleLaunchKind::Rollout, None)
            .expect("three fixed slots must be admissible");
        assert_eq!(
            ring.execution_slot(ticket).expect("ticket owns a slot"),
            step_id as usize,
            "first three tickets must deterministically own distinct slots"
        );
        ring.complete(
            ticket,
            Ok(vec![StepToken {
                req_id: rvllm_core::ReqId(step_id + 10),
                token_id: TokenId(step_id as u32 + 100),
                finished: false,
            }]),
        )
        .expect("reserved slot must accept its completion");
        tickets.push(ticket);
    }

    assert_eq!(ring.len(), MODEL_METAL_IN_FLIGHT_SLOTS);
    for (step_id, ticket) in tickets.into_iter().enumerate() {
        let output = ring
            .collect(ticket)
            .expect("ticket must retain its own output");
        assert_eq!(output[0].token_id, TokenId(step_id as u32 + 100));
    }
    assert_eq!(ring.len(), 0);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_in_flight_ring_full_fails_closed_without_overwrite() {
    let mut ring = ModelInFlightRing::default();
    let tickets: Vec<_> = (0..MODEL_METAL_IN_FLIGHT_SLOTS as u64)
        .map(|step_id| {
            let ticket = ring
                .reserve(step_id, AppleLaunchKind::Prefill, None)
                .expect("fixed slot must be available");
            ring.complete(ticket, Ok(Vec::new()))
                .expect("completion must bind to reserved ticket");
            ticket
        })
        .collect();

    let error = ring
        .reserve(99, AppleLaunchKind::Rollout, None)
        .expect_err("a fourth uncollected step must be rejected");
    assert!(format!("{error}").contains("in_flight_ring_full"));
    assert_eq!(ring.len(), MODEL_METAL_IN_FLIGHT_SLOTS);
    for ticket in tickets {
        assert!(
            ring.collect(ticket).is_ok(),
            "ring-full must not overwrite live steps"
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_in_flight_ring_collects_out_of_order_and_rejects_stale_tickets() {
    let mut ring = ModelInFlightRing::default();
    let tickets: Vec<_> = (0..3)
        .map(|step_id| {
            let ticket = ring
                .reserve(step_id, AppleLaunchKind::Rollout, None)
                .expect("slot must be available");
            ring.complete(
                ticket,
                Ok(vec![StepToken {
                    req_id: rvllm_core::ReqId(step_id),
                    token_id: TokenId(step_id as u32),
                    finished: false,
                }]),
            )
            .expect("completion must bind");
            ticket
        })
        .collect();

    assert_eq!(
        ring.collect(tickets[1]).expect("middle ticket")[0].token_id,
        TokenId(1)
    );
    let stale = ring
        .collect(tickets[1])
        .expect_err("a completion may be collected only once");
    assert!(format!("{stale}").contains("collect_stale_ticket"));
    assert_eq!(
        ring.len(),
        2,
        "stale collection must not reclaim another step"
    );
    assert_eq!(
        ring.collect(tickets[2]).expect("last ticket")[0].token_id,
        TokenId(2)
    );
    assert_eq!(
        ring.collect(tickets[0]).expect("first ticket")[0].token_id,
        TokenId(0)
    );
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_in_flight_error_collection_reclaims_the_slot() {
    let mut ring = ModelInFlightRing::default();
    let failed = ring
        .reserve(7, AppleLaunchKind::Rollout, None)
        .expect("slot must be available");
    ring.complete(
        failed,
        Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "injected completion failure",
            },
            model_ctx("test_completion"),
        )),
    )
    .expect("error completion must bind to its ticket");

    let error = ring
        .collect(failed)
        .expect_err("completion failure must propagate through collect");
    assert!(format!("{error}").contains("injected completion failure"));
    assert_eq!(ring.len(), 0, "failed completion must reclaim its slot");
    let replacement = ring
        .reserve(8, AppleLaunchKind::Prefill, None)
        .expect("reclaimed slot must admit subsequent work");
    assert_eq!(ring.execution_slot(replacement).unwrap(), 0);
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[test]
fn model_metal_command_buffer_error_collection_reclaims_the_slot() {
    let mut ring = ModelInFlightRing::default();
    let ticket = ring
        .reserve(11, AppleLaunchKind::Rollout, None)
        .expect("slot must be available");
    let command_error = metal_command_buffer_completion_error(MTLCommandBufferStatus::Error)
        .expect("Metal error status must map to a backend error");
    ring.complete(ticket, Err(command_error))
        .expect("GPU completion error must bind to its ticket");

    let error = ring
        .collect(ticket)
        .expect_err("Metal completion error must propagate");
    assert!(format!("{error}").contains("metal_command_buffer_failed"));
    assert_eq!(ring.len(), 0, "error collection must reclaim the slot");
    let replacement = ring
        .reserve(12, AppleLaunchKind::Prefill, None)
        .expect("reclaimed GPU-error slot must admit later work");
    assert_eq!(ring.execution_slot(replacement).unwrap(), 0);
}
