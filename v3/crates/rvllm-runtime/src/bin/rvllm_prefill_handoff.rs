//! Verify real Metal prefill -> packed ANE input conversion without invoking
//! any ANE framework. Emits cache summaries and reviewable MIL input artifacts.
#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
use rvllm_apple::ane_attention_layout::PackedAttentionLayout;
#[cfg(target_os = "macos")]
use rvllm_apple::{AppleBackend, AppleRuntimePlan, HandoffKind};
#[cfg(target_os = "macos")]
use rvllm_core::{ReqId, TokenId};
#[cfg(target_os = "macos")]
use rvllm_runtime::apple_bridge::{
    handoff_from_decode_plan_with_paged_kv, handoff_from_prefill_plan_with_paged_kv,
};
#[cfg(target_os = "macos")]
use rvllm_runtime::apple_metal_backend::ModelMetalBackend;
#[cfg(target_os = "macos")]
use rvllm_runtime::{BatchPlan, PagedKvConfig, PagedKvPool};
#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};
#[cfg(target_os = "macos")]
use std::io::Write;
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::time::Instant;

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("prefill handoff: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    Err("prefill handoff requires macOS with Apple Metal".into())
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir = None;
    let mut output_dir = None;
    let mut reference_path = None;
    let mut prompt = Vec::new();
    let mut capacity = 64_usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--hf-reference" => reference_path = Some(PathBuf::from(value)),
            "--prompt-token-ids" => {
                prompt = value
                    .split(',')
                    .map(str::parse::<u32>)
                    .collect::<Result<_, _>>()?
            }
            "--max-total-tokens" => capacity = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let model_dir = model_dir.ok_or("--model-dir is required")?;
    let output_dir = output_dir.ok_or("--output-dir is required")?;
    let prompt_len = u32::try_from(prompt.len())?;
    if prompt.is_empty() || prompt.len() > capacity || capacity % 32 != 0 {
        return Err(
            "provide nonempty --prompt-token-ids within a 32-aligned --max-total-tokens".into(),
        );
    }
    let reference_tokens = reference_path
        .as_ref()
        .map(|path| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let reference_prompt: Vec<u32> =
                serde_json::from_value(reference["prompt_token_ids"].clone())?;
            let generated: Vec<u32> =
                serde_json::from_value(reference["generated_tokens"].clone())?;
            if reference_prompt != prompt {
                return Err("reference prompt IDs do not match --prompt-token-ids".into());
            }
            if generated.is_empty() {
                return Err("reference has no generated token".into());
            }
            if prompt
                .len()
                .checked_add(generated.len() - 1)
                .filter(|&n| n <= capacity)
                .is_none()
            {
                return Err("reference continuation exceeds --max-total-tokens".into());
            }
            Ok(generated)
        })
        .transpose()?;
    let expected_first_token = reference_tokens.as_ref().map(|tokens| tokens[0]);
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)?;
    if prompt.iter().any(|&id| id as usize >= arch.vocab_size) {
        return Err("prompt token is outside the checkpoint vocabulary".into());
    }
    let eos = rvllm_loader::generation::load_eos_token_ids(&model_dir)?;
    if let Some(reference) = &reference_tokens {
        if reference.iter().any(|&id| id as usize >= arch.vocab_size)
            || reference[..reference.len() - 1]
                .iter()
                .any(|id| eos.contains(id))
        {
            return Err("reference contains an invalid token or continues after model EOS".into());
        }
    }
    // Reject unsupported packed shapes before loading weights or launching GPU
    // work. These same layouts are checked against the exported cache below.
    let layouts: Vec<_> = arch
        .layer_types
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            let heads = arch.num_attention_heads;
            let kv = arch.num_kv_heads_for_layer(index);
            let dim = arch.head_dim_for_layer(index);
            match kind {
                rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention => {
                    PackedAttentionLayout::sliding(heads, kv, dim, arch.sliding_window_size)
                }
                rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention => {
                    PackedAttentionLayout::new(heads, kv, dim, capacity)
                }
            }
        })
        .collect::<Result<_, _>>()?;
    if layouts.len() != arch.num_hidden_layers {
        return Err("checkpoint layer types disagree with layer count".into());
    }
    // Before backend initialization or worker creation: this executable owns
    // its environment. These are the existing development Metal controls.
    std::env::set_var("RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE", "1");
    std::env::set_var("RVLLM_METAL_MAX_TOTAL_TOKENS", capacity.to_string());
    std::env::set_var("RVLLM_METAL_MAX_BATCH_TOKENS", prompt.len().to_string());
    std::env::set_var("RVLLM_METAL_MAX_BATCH_SEQUENCES", "1");
    let config_bytes = std::fs::read(model_dir.join("config.json"))?;
    let layout_hash: [u8; 32] = Sha256::digest(&config_bytes).into();
    std::fs::create_dir(&output_dir)?;
    let plan = AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple Silicon", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: 1,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: arch.hidden_size,
        ane_intermediate_size: arch.intermediate_size,
        ane_num_layers: arch.num_hidden_layers,
        model_layout_hash: layout_hash,
        weights_path: Some(model_dir.clone()),
    };
    let mut backend = ModelMetalBackend::new(model_dir.clone());
    let prepare_start = Instant::now();
    backend.prepare(&plan)?;
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
    let limits = backend
        .model_capacity()
        .ok_or("prepared model has no capacity")?;
    let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(limits.physical_kv_pages, 1))?;
    let owner = ReqId(1);
    let chain = pool.allocate_chain(owner, prompt_len)?;
    let pages = pool.view(chain)?.pages.to_vec();
    let batch = BatchPlan::Prefill {
        req_ids: vec![owner],
        prompt_tokens_flat: prompt.iter().copied().map(TokenId).collect(),
        cu_seqlens_q: vec![0, prompt_len],
        query_start_positions: vec![0],
        context_lens: vec![prompt_len],
        kv_chains: vec![Some(chain)],
    };
    let handoff = handoff_from_prefill_plan_with_paged_kv(
        &batch,
        HandoffKind::MetalPrefillToMetalDecode,
        None,
        &pool,
        layout_hash,
    )?;
    let prefill_start = Instant::now();
    let start = backend.prefill_for_ane(&handoff)?;
    let prefill_sample_capture_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let perf = backend.probe_perf_stats();
    if perf.prefill_steps != 1 || perf.decode_steps != 0 || perf.command_buffers != 1 {
        return Err(
            "handoff did not use exactly one prefill command buffer with no decode replay".into(),
        );
    }
    if start.req_id != owner || start.next_position() != prompt.len() {
        return Err("ANE decode start does not match the prompt request/position".into());
    }
    let first_token = start.first_token.raw();
    if expected_first_token.is_some_and(|expected| expected != first_token) {
        return Err(format!(
            "first prefill token {first_token} differs from reference {expected_first_token:?}"
        )
        .into());
    }
    let snapshot = start.cache;
    if snapshot.layers.len() != arch.num_hidden_layers {
        return Err("captured layer count disagrees with checkpoint".into());
    }
    let pack_start = Instant::now();
    let mut layers = Vec::with_capacity(snapshot.layers.len());
    let mut emitted_shapes = std::collections::HashSet::new();
    for (index, layer) in snapshot.layers.iter().enumerate() {
        let shape = layer.shape;
        let layout = layouts[index];
        if (
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.sliding_window,
        ) != (
            layout.query_heads(),
            layout.kv_heads(),
            layout.head_dim(),
            layout.window(),
        ) {
            return Err(format!("layer {index} cache shape disagrees with checkpoint").into());
        }
        let packed = layout.import_cache(&layer.keys, &layer.values, snapshot.tokens)?;
        let key_nonzero = layer.keys.iter().filter(|v| **v != half::f16::ZERO).count();
        let value_nonzero = layer
            .values
            .iter()
            .filter(|v| **v != half::f16::ZERO)
            .count();
        if key_nonzero == 0 || value_nonzero == 0 {
            return Err(format!("layer {index} exported an empty key or value tensor").into());
        }
        let key_file = format!("layer-{index:02}-keys.fp16");
        let value_file = format!("layer-{index:02}-values.fp16");
        let key_sha256 = write_f16_tensor(&output_dir.join(&key_file), &layer.keys)?;
        let value_sha256 = write_f16_tensor(&output_dir.join(&value_file), &layer.values)?;
        if emitted_shapes.insert((
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            layout.capacity(),
        )) {
            let stem = format!(
                "attention-h{}-kv{}-d{}-c{}",
                shape.query_heads,
                shape.kv_heads,
                shape.head_dim,
                layout.capacity()
            );
            std::fs::write(output_dir.join(format!("{stem}.mil")), layout.mil())?;
            std::fs::write(output_dir.join(format!("{stem}-prefill.fp16")), &packed)?;
        }
        layers.push(serde_json::json!({
            "layer": index, "query_heads": shape.query_heads, "kv_heads": shape.kv_heads,
            "head_dim": shape.head_dim, "sliding_window": shape.sliding_window,
            "packed_capacity": layout.capacity(), "retained_tokens": layout.retained_tokens(snapshot.tokens)?,
            "kv_elements_each": layer.keys.len(), "key_nonzero": key_nonzero, "value_nonzero": value_nonzero,
            "key_file": key_file, "value_file": value_file,
            "key_sha256": key_sha256, "value_sha256": value_sha256,
            "key_max_abs": layer.keys.iter().map(|v| v.to_f32().abs()).fold(0.0_f32, f32::max),
            "value_max_abs": layer.values.iter().map(|v| v.to_f32().abs()).fold(0.0_f32, f32::max),
            "packed_input_bytes": packed.len(), "packed_input_sha256": format!("{:x}", Sha256::digest(&packed)),
        }));
    }
    let pack_and_artifact_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
    // This is a verifier for the first-token/position contract. It consumes
    // the actual sampled token and original BF16/F16 Metal cache; it neither
    // consumes the converted snapshot nor substitutes for ANE qualification.
    let continuation = if let Some(reference) = &reference_tokens {
        let verification_start = Instant::now();
        let mut generated = vec![first_token];
        for &expected in &reference[1..] {
            let position = u32::try_from(prompt.len() + generated.len() - 1)?;
            pool.append_tokens(chain, 1)?;
            let decode = BatchPlan::Decode {
                req_ids: vec![owner],
                bucket: 1,
                last_tokens: vec![TokenId(*generated.last().ok_or("missing decode seed")?)],
                positions: vec![position],
                context_lens: vec![position + 1],
                kv_chains: vec![Some(chain)],
            };
            let capsule = handoff_from_decode_plan_with_paged_kv(
                &decode,
                HandoffKind::MetalPrefillToMetalDecode,
                None,
                &pool,
                layout_hash,
            )?;
            let ticket = backend.launch_rollout(&capsule, None)?;
            let outputs = backend.collect(ticket)?;
            let [token] = outputs.as_slice() else {
                return Err("Metal continuation did not return one token".into());
            };
            if token.req_id != owner || token.token_id.raw() != expected {
                return Err(format!("Metal continuation token {} at position {position} differs from reference {expected}", token.token_id.raw()).into());
            }
            generated.push(token.token_id.raw());
        }
        serde_json::json!({
            "backend": "Metal", "matched": true, "generated_tokens": generated,
            "first_decode_position": (reference.len() > 1).then_some(snapshot.tokens),
            "decode_steps": reference.len() - 1,
            "elapsed_ms": verification_start.elapsed().as_secs_f64() * 1000.0,
            "consumed_converted_ane_cache": false,
        })
    } else {
        serde_json::Value::Null
    };
    let report = serde_json::json!({
        "schema": "rvllm.metal_ane_prefill_handoff.v2", "model_dir": model_dir,
        "prompt_token_ids": prompt, "context_capacity": capacity, "layers": layers,
        "metal_numeric_abi": snapshot.metal_numeric_abi, "source_dtype": format!("{:?}", snapshot.source_dtype),
        "destination_dtype": "float16", "physical_pages": pages.iter().map(|p| p.0).collect::<Vec<_>>(),
        "exported_tensor_order": "token, kv_head, head_dim; little-endian IEEE FP16",
        "prepare_ms": prepare_ms, "prefill_sample_capture_ms": prefill_sample_capture_ms,
        "prefill_cpu_wall_ns": perf.last_step_cpu_wall_ns,
        "first_generated_token": first_token, "ane_next_position": snapshot.tokens,
        "first_token_is_eos": eos.contains(&first_token),
        "expected_first_token": expected_first_token, "hf_reference": reference_path,
        "prefill_steps": perf.prefill_steps, "decode_steps": perf.decode_steps,
        "command_buffers": perf.command_buffers,
        "pack_and_artifact_ms": pack_and_artifact_ms,
        "continuation_verification": continuation,
        "cache_capture_after_collection": true, "ane_execution_verified": false,
        "claim": "One Metal prefill sampled the first output from its final row and captured prompt KV after collection. ANE compile/decode was not attempted."
    });
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(output_dir.join("report.json"), &json)?;
    println!("{json}");
    Ok(())
}

#[cfg(target_os = "macos")]
fn write_f16_tensor(path: &Path, values: &[half::f16]) -> Result<String, std::io::Error> {
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    let mut output = std::io::BufWriter::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 8192];
    for values in values.chunks(buffer.len() / 2) {
        let bytes = &mut buffer[..values.len() * 2];
        for (value, bytes) in values.iter().zip(bytes.chunks_exact_mut(2)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(&*bytes);
        output.write_all(bytes)?;
    }
    output.flush()?;
    Ok(format!("{:x}", digest.finalize()))
}
