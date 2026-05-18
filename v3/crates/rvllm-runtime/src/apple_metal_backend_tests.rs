use super::*;
#[cfg(target_os = "macos")]
use rvllm_apple_metal::weight_loader::scan_safetensor_tensors;
use serde_json::{Map, Value};
use std::fs::{self, File};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<half::f16>());
    for value in values {
        let bits = half::f16::from_f32(*value).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
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
    let update_scale = if apply_layer_scalar { 6.0f32 } else { 1.0f32 };

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
        residual[d] += projected_attn[d] * update_scale;
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
        residual[d] += mlp_out[d] * update_scale;
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
    let layer_scalar = 3.0f32;
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
    for dim in 0..hidden {
        residual[dim] += attn_residual[dim] * layer_scalar;
    }
    residual = cpu_full_nonzero_rms_norm(&residual, &post_attn_norm, eps);

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
    for dim in 0..hidden {
        residual[dim] += mlp_out[dim] * layer_scalar;
    }
    residual = cpu_full_nonzero_rms_norm(&residual, &post_ff_norm, eps);

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

        let projected = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
        for dim in 0..hidden {
            residual[dim] += projected[dim] * layer_scalar[dim];
        }
        residual = cpu_full_nonzero_rms_norm(&residual, &norm, eps);

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for dim in 0..hidden {
            residual[dim] += mlp_out[dim] * layer_scalar[dim];
        }
        residual = cpu_full_nonzero_rms_norm(&residual, &norm, eps);

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
fn read_e2b_hf_reference_selected_logits(path: &std::path::Path) -> (Vec<u32>, Vec<f32>, u32) {
    let raw = fs::read_to_string(path).expect("read HF reference logits artifact");
    let value: Value = serde_json::from_str(&raw).expect("parse HF reference logits artifact");

    assert_eq!(
        value["schema"].as_str(),
        Some("rvllm.gemma4_hf_reference_logits.v1")
    );
    assert_eq!(
        value["prompt_token_ids"]
            .as_array()
            .expect("prompt ids")
            .as_slice(),
        &[Value::from(2), Value::from(4)]
    );
    assert_eq!(value["decode_steps"].as_u64(), Some(1));

    let step = &value["steps"].as_array().expect("steps")[0];
    let next_token = step["next_token"].as_u64().expect("next token") as u32;
    let selected = step["selected_logits"].as_array().expect("selected logits");
    let mut token_ids = Vec::with_capacity(selected.len());
    let mut logits = Vec::with_capacity(selected.len());
    for item in selected {
        token_ids.push(item["token_id"].as_u64().expect("selected token id") as u32);
        logits.push(item["logit"].as_f64().expect("selected logit") as f32);
    }
    (token_ids, logits, next_token)
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
        [0.0f32, 0.0, 0.094_23, 0.0, 0.0, 24.338_19, 0.0, 0.0],
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
    let layer_scalar = vec![6.0f32];
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
        &[1],
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

    let layer_scalar = vec![3.0f32];
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
        &[1],
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
    let hf_trace_path =
        std::path::PathBuf::from(format!("/tmp/gemma4-e2b-hf-layer{trace_layer}-trace.json"));
    if !hf_trace_path.exists() {
        eprintln!(
            "skipping: HF layer trace artifact is missing at {}",
            hf_trace_path.display()
        );
        return;
    }
    let metal_trace_path = std::path::PathBuf::from(format!(
        "/tmp/gemma4-e2b-metal-layer{trace_layer}-trace.json"
    ));
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
        vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
        vec![0, 2],
        vec![1],
        vec![2],
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
    assert_eq!(
        hf["prompt_token_ids"]
            .as_array()
            .expect("prompt ids")
            .as_slice(),
        &[Value::from(2), Value::from(4)]
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
        "after_ffn_branch",
        "after_post_feedforward_layernorm",
        "per_layer_input",
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
        "final_residual_after_layer",
    ] {
        compare_trace_summary_stats(&hf, &metal, name);
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
