//! Native Gemma 4 E2B greedy generation on Apple Metal.
//!
//! This executor is intentionally narrow: batch 1, cached BF16/FP16 HF
//! safetensors, token-by-token prompt replay, greedy argmax.  It owns its
//! Metal buffers directly instead of reusing CUDA arena abstractions.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

use half::f16;
use metal_rs::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use rvllm_core::{AppleCtx, AppleError, DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};

const BACKEND: &str = "native-apple-gemma4-metal";
const METAL_SOURCE: &str = include_str!("../../rvllm-apple/metal/prefill.metal");
const PLE_DIM: usize = 256;
const NUM_KV_SHARED_LAYERS: usize = 20;

pub struct Gemma4AppleEngine {
    arch: Gemma4Arch,
    device: Device,
    queue: CommandQueue,
    kernels: Gemma4Pipelines,
    embedding: Buffer,
    embed_tokens_per_layer: Buffer,
    per_layer_model_projection: Buffer,
    per_layer_projection_norm: Buffer,
    lm_head: Buffer,
    final_norm: Buffer,
    layers: Vec<LayerWeights>,
    scratch: Scratch,
}

struct LayerWeights {
    intermediate: usize,
    is_kv_shared_layer: bool,
    qkv: Buffer,
    o_proj: Buffer,
    gate_up: Buffer,
    down: Buffer,
    input_layernorm: Buffer,
    post_attention_layernorm: Buffer,
    pre_feedforward_layernorm: Buffer,
    post_feedforward_layernorm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    per_layer_input_gate: Buffer,
    per_layer_projection: Buffer,
    post_per_layer_input_norm: Buffer,
    layer_scalar: Buffer,
}

struct Scratch {
    residual: Buffer,
    normed: Buffer,
    qkv_out: Buffer,
    q_normed: Buffer,
    k_normed: Buffer,
    v_normed: Buffer,
    attn_out: Buffer,
    gemm_tmp: Buffer,
    gate_up_out: Buffer,
    mlp_out: Buffer,
    logits: Buffer,
    next_token: Buffer,
    per_layer_context: Buffer,
    per_layer_inputs: Buffer,
    ple_gate: Buffer,
    ple_proj: Buffer,
}

struct KvCache {
    layers: Vec<LayerKvCache>,
    shared_sliding_layer_idx: usize,
    shared_global_layer_idx: usize,
}

struct LayerKvCache {
    key: Buffer,
    value: Buffer,
}

struct Gemma4Pipelines {
    embedding: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    rmsnorm_heads: ComputePipelineState,
    rmsnorm_heads_no_gamma: ComputePipelineState,
    matvec: ComputePipelineState,
    matvec_logits: ComputePipelineState,
    rope_cache: ComputePipelineState,
    rope_query: ComputePipelineState,
    attention: ComputePipelineState,
    norm_add_residual: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    argmax: ComputePipelineState,
    ple_combine: ComputePipelineState,
    ple_gate_mul: ComputePipelineState,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct EmbeddingParams {
    token_id: u32,
    hidden: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RmsNormParams {
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct HeadNormParams {
    num_heads: u32,
    head_dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct MatVecParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LogitsParams {
    vocab: u32,
    hidden: u32,
    softcap: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RopeCacheParams {
    position: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct AttentionParams {
    position: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    is_sliding: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct NormAddParams {
    hidden: u32,
    eps: f32,
    apply_scalar: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GeluParams {
    intermediate: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ArgmaxParams {
    vocab: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PleCombineParams {
    token_id: u32,
    num_layers: u32,
    ple_dim: u32,
    hidden_scale: f32,
    combine_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PleGateParams {
    layer_idx: u32,
    ple_dim: u32,
}

impl Gemma4AppleEngine {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let arch = Gemma4Arch::from_dir(model_dir)?;
        validate_e2b_arch(&arch)?;

        let device = Device::system_default()
            .ok_or_else(|| apple_err(AppleError::MetalUnavailable, "load"))?;
        let queue = device.new_command_queue();
        let kernels = Gemma4Pipelines::new(&device)?;
        let shards = ShardSet::open(model_dir)?;

        let prefix = &arch.weight_prefix;
        let embed_name = format!("{prefix}.embed_tokens.weight");
        let embed_entry = shards.must_tensor(&embed_name, model_dir)?;
        expect_shape(
            model_dir,
            &embed_entry,
            &[arch.vocab_size, arch.hidden_size],
        )?;

        let embed_scale = (arch.hidden_size as f32).sqrt();
        let mut embed_bytes = shards.tensor_to_f16_bytes(&embed_entry, model_dir)?;
        scale_f16_bytes(&mut embed_bytes, embed_scale);
        let embedding = new_buffer_from_bytes(&device, &embed_bytes);

        let ple_total = arch.num_hidden_layers * PLE_DIM;
        let ple_embed_name = format!("{prefix}.embed_tokens_per_layer.weight");
        let ple_embed_entry = shards.must_tensor(&ple_embed_name, model_dir)?;
        expect_shape(model_dir, &ple_embed_entry, &[arch.vocab_size, ple_total])?;
        let mut ple_embed_bytes = shards.tensor_to_f16_bytes(&ple_embed_entry, model_dir)?;
        scale_f16_bytes(&mut ple_embed_bytes, (PLE_DIM as f32).sqrt());
        let embed_tokens_per_layer = new_buffer_from_bytes(&device, &ple_embed_bytes);

        let ple_model_projection_name = format!("{prefix}.per_layer_model_projection.weight");
        let ple_model_projection_entry =
            shards.must_tensor(&ple_model_projection_name, model_dir)?;
        expect_shape(
            model_dir,
            &ple_model_projection_entry,
            &[ple_total, arch.hidden_size],
        )?;
        let per_layer_model_projection = new_buffer_from_bytes(
            &device,
            &shards.tensor_to_f16_bytes(&ple_model_projection_entry, model_dir)?,
        );

        let ple_norm_name = format!("{prefix}.per_layer_projection_norm.weight");
        let ple_norm_entry = shards.must_tensor(&ple_norm_name, model_dir)?;
        expect_shape(model_dir, &ple_norm_entry, &[PLE_DIM])?;
        let per_layer_projection_norm = new_buffer_from_bytes(
            &device,
            &shards.tensor_to_f16_bytes(&ple_norm_entry, model_dir)?,
        );

        let lm_head = if let Some(entry) = shards.tensor("lm_head.weight") {
            expect_shape(model_dir, &entry, &[arch.vocab_size, arch.hidden_size])?;
            new_buffer_from_bytes(&device, &shards.tensor_to_f16_bytes(&entry, model_dir)?)
        } else {
            new_buffer_from_bytes(
                &device,
                &shards.tensor_to_f16_bytes(&embed_entry, model_dir)?,
            )
        };

        let final_norm_name = format!("{prefix}.norm.weight");
        let final_norm_entry = shards.must_tensor(&final_norm_name, model_dir)?;
        expect_shape(model_dir, &final_norm_entry, &[arch.hidden_size])?;
        let final_norm = new_buffer_from_bytes(
            &device,
            &shards.tensor_to_f16_bytes(&final_norm_entry, model_dir)?,
        );

        let mut layers = Vec::with_capacity(arch.num_hidden_layers);
        for layer_idx in 0..arch.num_hidden_layers {
            layers.push(load_layer(&device, &shards, model_dir, &arch, layer_idx)?);
        }

        let max_intermediate = layers
            .iter()
            .map(|layer| layer.intermediate)
            .max()
            .unwrap_or(arch.intermediate_size);
        let scratch = Scratch::new(&device, &arch, max_intermediate);

        Ok(Self {
            arch,
            device,
            queue,
            kernels,
            embedding,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            lm_head,
            final_norm,
            layers,
            scratch,
        })
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(apple_err(
                AppleError::LayerShapeInvalid {
                    reason: "prompt must contain at least one token",
                },
                "generate",
            ));
        }
        for &token in prompt_ids {
            if token as usize >= self.arch.vocab_size {
                return Err(apple_err(
                    AppleError::LayerShapeInvalid {
                        reason: "prompt token id exceeds E2B vocabulary",
                    },
                    "generate",
                ));
            }
        }

        let max_seq = prompt_ids.len().checked_add(max_new).ok_or_else(|| {
            apple_err(
                AppleError::LayerShapeInvalid {
                    reason: "prompt plus generation length overflows",
                },
                "generate",
            )
        })?;
        if max_seq > self.arch.max_position_embeddings {
            return Err(apple_err(
                AppleError::LayerShapeInvalid {
                    reason: "prompt plus generation exceeds model context",
                },
                "generate",
            ));
        }

        let kv_cache = KvCache::new(&self.device, &self.arch, max_seq);
        let mut next = 0u32;
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            next = self.forward_token(token_id, position as u32, &kv_cache)?;
        }

        let mut out = Vec::with_capacity(max_new);
        for step in 0..max_new {
            out.push(next);
            if eos_ids.contains(&next) {
                break;
            }
            if step + 1 == max_new {
                break;
            }
            let position = prompt_ids.len() + step;
            next = self.forward_token(next, position as u32, &kv_cache)?;
        }
        Ok(out)
    }

    fn forward_token(&self, token_id: u32, position: u32, kv_cache: &KvCache) -> Result<u32> {
        self.dispatch_embedding(token_id)?;
        self.dispatch_per_layer_inputs(token_id)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.dispatch_layer(layer_idx, layer, kv_cache, position)?;
        }
        self.dispatch_logits()
    }

    fn dispatch_embedding(&self, token_id: u32) -> Result<()> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let params = EmbeddingParams {
            token_id,
            hidden: self.arch.hidden_size as u32,
            scale: 1.0,
        };
        set_pipeline(&encoder, &self.kernels.embedding);
        encoder.set_buffer(0, Some(&self.embedding), 0);
        encoder.set_buffer(1, Some(&self.scratch.residual), 0);
        set_bytes(&encoder, 2, &params);
        dispatch_1d(&encoder, self.arch.hidden_size as u64, 256);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if debug_enabled() {
            debug_f32_stats(
                "embedding residual",
                &self.scratch.residual,
                self.arch.hidden_size,
            );
        }
        Ok(())
    }

    fn dispatch_per_layer_inputs(&self, token_id: u32) -> Result<()> {
        let total = self.arch.num_hidden_layers * PLE_DIM;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.encode_matvec(
            &encoder,
            &self.per_layer_model_projection,
            &self.scratch.residual,
            &self.scratch.per_layer_context,
            total,
            self.arch.hidden_size,
        );

        let params = PleCombineParams {
            token_id,
            num_layers: self.arch.num_hidden_layers as u32,
            ple_dim: PLE_DIM as u32,
            hidden_scale: 1.0 / (self.arch.hidden_size as f32).sqrt(),
            combine_scale: std::f32::consts::FRAC_1_SQRT_2,
        };
        set_pipeline(&encoder, &self.kernels.ple_combine);
        encoder.set_buffer(0, Some(&self.scratch.per_layer_context), 0);
        encoder.set_buffer(1, Some(&self.embed_tokens_per_layer), 0);
        encoder.set_buffer(2, Some(&self.per_layer_projection_norm), 0);
        encoder.set_buffer(3, Some(&self.scratch.per_layer_inputs), 0);
        set_bytes(&encoder, 4, &params);
        dispatch_1d(&encoder, total as u64, 256);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if debug_enabled() {
            debug_f32_stats(
                "per-layer input[0]",
                &self.scratch.per_layer_inputs,
                PLE_DIM,
            );
        }
        Ok(())
    }

    fn dispatch_layer(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        kv_cache: &KvCache,
        position: u32,
    ) -> Result<()> {
        let head_dim = self.arch.head_dim_for_layer(layer_idx);
        let num_heads = self.arch.num_attention_heads;
        let num_kv_heads = self.arch.num_kv_heads_for_layer(layer_idx);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_rows = if layer.is_kv_shared_layer {
            q_dim
        } else {
            q_dim + 2 * kv_dim
        };
        let rotary_dim = self.arch.rotary_dim_for_layer(layer_idx);
        let is_sliding = self.arch.layer_types[layer_idx] == Gemma4LayerType::SlidingAttention;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.encode_rmsnorm(
            &encoder,
            &self.scratch.residual,
            &layer.input_layernorm,
            &self.scratch.normed,
            self.arch.hidden_size,
        );
        self.encode_matvec(
            &encoder,
            &layer.qkv,
            &self.scratch.normed,
            &self.scratch.qkv_out,
            qkv_rows,
            self.arch.hidden_size,
        );

        self.encode_head_rmsnorm(
            &encoder,
            &self.scratch.qkv_out,
            0,
            &layer.q_norm,
            &self.scratch.q_normed,
            0,
            num_heads,
            head_dim,
        );
        let rope = RopeCacheParams {
            position,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            rotary_dim: rotary_dim as u32,
            theta: self.arch.rope_theta_for_layer(layer_idx),
        };
        if layer.is_kv_shared_layer {
            set_pipeline(&encoder, &self.kernels.rope_query);
            encoder.set_buffer(0, Some(&self.scratch.q_normed), 0);
            encoder.set_buffer(1, Some(&self.scratch.q_normed), 0);
            set_bytes(&encoder, 2, &rope);
            dispatch_2d(&encoder, (head_dim / 2) as u64, num_heads as u64, 64, 1);
        } else {
            self.encode_head_rmsnorm(
                &encoder,
                &self.scratch.qkv_out,
                q_dim * std::mem::size_of::<f32>(),
                &layer.k_norm,
                &self.scratch.k_normed,
                0,
                num_kv_heads,
                head_dim,
            );
            self.encode_head_rmsnorm_no_gamma(
                &encoder,
                &self.scratch.qkv_out,
                (q_dim + kv_dim) * std::mem::size_of::<f32>(),
                &self.scratch.v_normed,
                0,
                num_kv_heads,
                head_dim,
            );
            let write_cache = &kv_cache.layers[layer_idx];
            set_pipeline(&encoder, &self.kernels.rope_cache);
            encoder.set_buffer(0, Some(&self.scratch.q_normed), 0);
            encoder.set_buffer(1, Some(&self.scratch.k_normed), 0);
            encoder.set_buffer(2, Some(&self.scratch.v_normed), 0);
            encoder.set_buffer(3, Some(&self.scratch.q_normed), 0);
            encoder.set_buffer(4, Some(&write_cache.key), 0);
            encoder.set_buffer(5, Some(&write_cache.value), 0);
            set_bytes(&encoder, 6, &rope);
            dispatch_2d(
                &encoder,
                (head_dim / 2) as u64,
                num_heads.max(num_kv_heads) as u64,
                64,
                1,
            );
        }

        let attention = AttentionParams {
            position,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            sliding_window: self.arch.sliding_window_size as u32,
            is_sliding: u32::from(is_sliding),
            scale: 1.0,
        };
        set_pipeline(&encoder, &self.kernels.attention);
        let attention_cache = kv_cache.attention_cache(layer_idx, is_sliding);
        encoder.set_buffer(0, Some(&self.scratch.q_normed), 0);
        encoder.set_buffer(1, Some(&attention_cache.key), 0);
        encoder.set_buffer(2, Some(&attention_cache.value), 0);
        encoder.set_buffer(3, Some(&self.scratch.attn_out), 0);
        set_bytes(&encoder, 4, &attention);
        dispatch_1d(&encoder, q_dim as u64, 256);

        self.encode_matvec(
            &encoder,
            &layer.o_proj,
            &self.scratch.attn_out,
            &self.scratch.gemm_tmp,
            self.arch.hidden_size,
            q_dim,
        );
        self.encode_norm_add_residual(
            &encoder,
            &self.scratch.gemm_tmp,
            &layer.post_attention_layernorm,
            &self.scratch.residual,
            self.arch.hidden_size,
            false,
            &layer.layer_scalar,
        );

        self.encode_rmsnorm(
            &encoder,
            &self.scratch.residual,
            &layer.pre_feedforward_layernorm,
            &self.scratch.normed,
            self.arch.hidden_size,
        );
        self.encode_matvec(
            &encoder,
            &layer.gate_up,
            &self.scratch.normed,
            &self.scratch.gate_up_out,
            2 * layer.intermediate,
            self.arch.hidden_size,
        );

        let gelu = GeluParams {
            intermediate: layer.intermediate as u32,
        };
        set_pipeline(&encoder, &self.kernels.gelu_mul);
        encoder.set_buffer(0, Some(&self.scratch.gate_up_out), 0);
        encoder.set_buffer(1, Some(&self.scratch.mlp_out), 0);
        set_bytes(&encoder, 2, &gelu);
        dispatch_1d(&encoder, layer.intermediate as u64, 256);

        self.encode_matvec(
            &encoder,
            &layer.down,
            &self.scratch.mlp_out,
            &self.scratch.gemm_tmp,
            self.arch.hidden_size,
            layer.intermediate,
        );
        self.encode_norm_add_residual(
            &encoder,
            &self.scratch.gemm_tmp,
            &layer.post_feedforward_layernorm,
            &self.scratch.residual,
            self.arch.hidden_size,
            false,
            &layer.layer_scalar,
        );

        self.encode_matvec(
            &encoder,
            &layer.per_layer_input_gate,
            &self.scratch.residual,
            &self.scratch.ple_gate,
            PLE_DIM,
            self.arch.hidden_size,
        );
        let ple_gate = PleGateParams {
            layer_idx: layer_idx as u32,
            ple_dim: PLE_DIM as u32,
        };
        set_pipeline(&encoder, &self.kernels.ple_gate_mul);
        encoder.set_buffer(0, Some(&self.scratch.ple_gate), 0);
        encoder.set_buffer(1, Some(&self.scratch.per_layer_inputs), 0);
        set_bytes(&encoder, 2, &ple_gate);
        dispatch_1d(&encoder, PLE_DIM as u64, 256);

        self.encode_matvec(
            &encoder,
            &layer.per_layer_projection,
            &self.scratch.ple_gate,
            &self.scratch.ple_proj,
            self.arch.hidden_size,
            PLE_DIM,
        );
        self.encode_norm_add_residual(
            &encoder,
            &self.scratch.ple_proj,
            &layer.post_per_layer_input_norm,
            &self.scratch.residual,
            self.arch.hidden_size,
            true,
            &layer.layer_scalar,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if debug_enabled() {
            debug_f32_stats(
                &format!("layer {layer_idx} residual"),
                &self.scratch.residual,
                self.arch.hidden_size,
            );
            if layer_idx <= 4 {
                debug_f32_stats(
                    &format!("layer {layer_idx} qkv_out"),
                    &self.scratch.qkv_out,
                    qkv_rows,
                );
                debug_f32_stats(
                    &format!("layer {layer_idx} q_normed"),
                    &self.scratch.q_normed,
                    q_dim,
                );
                if !layer.is_kv_shared_layer {
                    debug_f32_stats(
                        &format!("layer {layer_idx} k_normed"),
                        &self.scratch.k_normed,
                        kv_dim,
                    );
                    debug_f32_stats(
                        &format!("layer {layer_idx} v_normed"),
                        &self.scratch.v_normed,
                        kv_dim,
                    );
                }
                debug_f32_stats(
                    &format!("layer {layer_idx} attn_out"),
                    &self.scratch.attn_out,
                    q_dim,
                );
                debug_f32_stats(
                    &format!("layer {layer_idx} gate_up_out"),
                    &self.scratch.gate_up_out,
                    2 * layer.intermediate,
                );
                debug_f32_stats(
                    &format!("layer {layer_idx} mlp_out"),
                    &self.scratch.mlp_out,
                    layer.intermediate,
                );
                debug_f32_stats(
                    &format!("layer {layer_idx} ple_gate"),
                    &self.scratch.ple_gate,
                    PLE_DIM,
                );
                debug_f32_stats(
                    &format!("layer {layer_idx} ple_proj"),
                    &self.scratch.ple_proj,
                    self.arch.hidden_size,
                );
            }
        }
        Ok(())
    }

    fn dispatch_logits(&self) -> Result<u32> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_rmsnorm(
            &encoder,
            &self.scratch.residual,
            &self.final_norm,
            &self.scratch.normed,
            self.arch.hidden_size,
        );

        let logits = LogitsParams {
            vocab: self.arch.vocab_size as u32,
            hidden: self.arch.hidden_size as u32,
            softcap: self.arch.logit_softcap,
        };
        set_pipeline(&encoder, &self.kernels.matvec_logits);
        encoder.set_buffer(0, Some(&self.lm_head), 0);
        encoder.set_buffer(1, Some(&self.scratch.normed), 0);
        encoder.set_buffer(2, Some(&self.scratch.logits), 0);
        set_bytes(&encoder, 3, &logits);
        dispatch_1d(&encoder, self.arch.vocab_size as u64, 256);

        let argmax = ArgmaxParams {
            vocab: self.arch.vocab_size as u32,
        };
        set_pipeline(&encoder, &self.kernels.argmax);
        encoder.set_buffer(0, Some(&self.scratch.logits), 0);
        encoder.set_buffer(1, Some(&self.scratch.next_token), 0);
        set_bytes(&encoder, 2, &argmax);
        dispatch_1d(&encoder, 1, 1);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if debug_enabled() {
            debug_f32_stats(
                "final residual",
                &self.scratch.residual,
                self.arch.hidden_size,
            );
            debug_f32_stats("final normed", &self.scratch.normed, self.arch.hidden_size);
            debug_top_logits(&self.scratch.logits, self.arch.vocab_size, 10);
        }

        let ptr = self.scratch.next_token.contents() as *const u32;
        Ok(unsafe { *ptr })
    }

    fn encode_rmsnorm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        gamma: &Buffer,
        output: &Buffer,
        dim: usize,
    ) {
        let params = RmsNormParams {
            dim: dim as u32,
            eps: self.arch.rms_norm_eps,
        };
        set_pipeline(encoder, &self.kernels.rmsnorm);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(gamma), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_bytes(encoder, 3, &params);
        dispatch_1d(encoder, dim as u64, 256);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_head_rmsnorm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        input_offset: usize,
        gamma: &Buffer,
        output: &Buffer,
        output_offset: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let params = HeadNormParams {
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            eps: self.arch.rms_norm_eps,
        };
        set_pipeline(encoder, &self.kernels.rmsnorm_heads);
        encoder.set_buffer(0, Some(input), input_offset as u64);
        encoder.set_buffer(1, Some(gamma), 0);
        encoder.set_buffer(2, Some(output), output_offset as u64);
        set_bytes(encoder, 3, &params);
        dispatch_1d(encoder, (num_heads * head_dim) as u64, 256);
    }

    fn encode_head_rmsnorm_no_gamma(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        input_offset: usize,
        output: &Buffer,
        output_offset: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let params = HeadNormParams {
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            eps: self.arch.rms_norm_eps,
        };
        set_pipeline(encoder, &self.kernels.rmsnorm_heads_no_gamma);
        encoder.set_buffer(0, Some(input), input_offset as u64);
        encoder.set_buffer(1, Some(output), output_offset as u64);
        set_bytes(encoder, 2, &params);
        dispatch_1d(encoder, (num_heads * head_dim) as u64, 256);
    }

    fn encode_matvec(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        cols: usize,
    ) {
        let params = MatVecParams {
            rows: rows as u32,
            cols: cols as u32,
        };
        set_pipeline(encoder, &self.kernels.matvec);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_bytes(encoder, 3, &params);
        dispatch_1d(encoder, rows as u64, 256);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_norm_add_residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        gamma: &Buffer,
        residual: &Buffer,
        hidden: usize,
        apply_scalar: bool,
        layer_scalar: &Buffer,
    ) {
        let params = NormAddParams {
            hidden: hidden as u32,
            eps: self.arch.rms_norm_eps,
            apply_scalar: u32::from(apply_scalar),
        };
        set_pipeline(encoder, &self.kernels.norm_add_residual);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(gamma), 0);
        encoder.set_buffer(2, Some(residual), 0);
        encoder.set_buffer(3, Some(layer_scalar), 0);
        set_bytes(encoder, 4, &params);
        dispatch_1d(encoder, hidden as u64, 256);
    }
}

impl Gemma4Pipelines {
    fn new(device: &Device) -> Result<Self> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(METAL_SOURCE, &options)
            .map_err(|_| {
                apple_err(
                    AppleError::InvalidMetalRecipe {
                        reason: "Gemma4 Metal source did not compile",
                    },
                    "compile_kernels",
                )
            })?;
        Ok(Self {
            embedding: pipeline(device, &library, "rvllm_g4_embedding")?,
            rmsnorm: pipeline(device, &library, "rvllm_g4_rmsnorm")?,
            rmsnorm_heads: pipeline(device, &library, "rvllm_g4_rmsnorm_heads")?,
            rmsnorm_heads_no_gamma: pipeline(device, &library, "rvllm_g4_rmsnorm_heads_no_gamma")?,
            matvec: pipeline(device, &library, "rvllm_g4_matvec_half")?,
            matvec_logits: pipeline(device, &library, "rvllm_g4_matvec_logits")?,
            rope_cache: pipeline(device, &library, "rvllm_g4_rope_cache")?,
            rope_query: pipeline(device, &library, "rvllm_g4_rope_query")?,
            attention: pipeline(device, &library, "rvllm_g4_attention")?,
            norm_add_residual: pipeline(device, &library, "rvllm_g4_norm_add_residual")?,
            gelu_mul: pipeline(device, &library, "rvllm_g4_gelu_mul")?,
            argmax: pipeline(device, &library, "rvllm_g4_argmax")?,
            ple_combine: pipeline(device, &library, "rvllm_g4_ple_combine")?,
            ple_gate_mul: pipeline(device, &library, "rvllm_g4_ple_gate_mul")?,
        })
    }
}

impl Scratch {
    fn new(device: &Device, arch: &Gemma4Arch, max_intermediate: usize) -> Self {
        let max_q_dim = arch.max_q_dim();
        let max_kv_dim = arch.max_kv_heads() * arch.max_head_dim();
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let max_gemm = (2 * max_intermediate)
            .max(max_qkv_rows)
            .max(arch.hidden_size);
        Self {
            residual: new_f32_buffer(device, arch.hidden_size),
            normed: new_f32_buffer(device, arch.hidden_size),
            qkv_out: new_f32_buffer(device, max_qkv_rows),
            q_normed: new_f32_buffer(device, max_q_dim),
            k_normed: new_f32_buffer(device, max_kv_dim),
            v_normed: new_f32_buffer(device, max_kv_dim),
            attn_out: new_f32_buffer(device, max_q_dim),
            gemm_tmp: new_f32_buffer(device, max_gemm),
            gate_up_out: new_f32_buffer(device, 2 * max_intermediate),
            mlp_out: new_f32_buffer(device, max_intermediate),
            logits: new_f32_buffer(device, arch.vocab_size),
            next_token: device.new_buffer(
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            per_layer_context: new_f32_buffer(device, arch.num_hidden_layers * PLE_DIM),
            per_layer_inputs: new_f32_buffer(device, arch.num_hidden_layers * PLE_DIM),
            ple_gate: new_f32_buffer(device, PLE_DIM),
            ple_proj: new_f32_buffer(device, arch.hidden_size),
        }
    }
}

impl KvCache {
    fn new(device: &Device, arch: &Gemma4Arch, max_seq: usize) -> Self {
        let mut layers = Vec::with_capacity(arch.num_hidden_layers);
        for layer_idx in 0..arch.num_hidden_layers {
            let kv_dim = arch.kv_dim_for_layer(layer_idx);
            layers.push(LayerKvCache {
                key: new_f32_buffer(device, max_seq * kv_dim),
                value: new_f32_buffer(device, max_seq * kv_dim),
            });
        }
        let first_shared = first_kv_shared_layer_idx(arch);
        let mut shared_sliding_layer_idx = 0usize;
        let mut shared_global_layer_idx = 0usize;
        for idx in 0..first_shared {
            match arch.layer_types[idx] {
                Gemma4LayerType::SlidingAttention => shared_sliding_layer_idx = idx,
                Gemma4LayerType::GlobalAttention => shared_global_layer_idx = idx,
            }
        }
        Self {
            layers,
            shared_sliding_layer_idx,
            shared_global_layer_idx,
        }
    }

    fn attention_cache(&self, layer_idx: usize, is_sliding: bool) -> &LayerKvCache {
        if layer_idx >= self.layers.len() - NUM_KV_SHARED_LAYERS {
            if is_sliding {
                &self.layers[self.shared_sliding_layer_idx]
            } else {
                &self.layers[self.shared_global_layer_idx]
            }
        } else {
            &self.layers[layer_idx]
        }
    }
}

struct ShardMap {
    mmap: memmap2::Mmap,
    header: ShardHeader,
}

struct ShardSet {
    shards: Vec<ShardMap>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
}

impl ShardSet {
    fn open(model_dir: &Path) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut shards = Vec::with_capacity(idx.shards.len());
        for path in &idx.shards {
            let file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: path.clone(),
                source,
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: path.clone(),
                source,
            })?;
            let header = ShardHeader::parse(path, &mmap)?;
            shards.push(ShardMap { mmap, header });
        }

        let mut tensors = BTreeMap::new();
        for (shard_idx, shard) in shards.iter().enumerate() {
            for (name, entry) in &shard.header.tensors {
                tensors.insert(name.clone(), (shard_idx, entry.clone()));
            }
        }
        Ok(Self { shards, tensors })
    }

    fn tensor(&self, name: &str) -> Option<TensorEntry> {
        self.tensors.get(name).map(|(_, entry)| entry.clone())
    }

    fn must_tensor(&self, name: &str, model_dir: &Path) -> Result<TensorEntry> {
        self.tensor(name).ok_or_else(|| {
            loader_err(
                model_dir,
                Some(name.to_string()),
                LoaderError::MissingTensor {
                    name: name.to_string(),
                },
            )
        })
    }

    fn raw_bytes(&self, entry: &TensorEntry) -> &[u8] {
        let (shard_idx, _) = &self.tensors[&entry.name];
        let shard = &self.shards[*shard_idx];
        let start = entry.file_offset as usize;
        &shard.mmap[start..start + entry.nbytes as usize]
    }

    fn tensor_to_f16_bytes(&self, entry: &TensorEntry, model_dir: &Path) -> Result<Vec<u8>> {
        let raw = self.raw_bytes(entry);
        match entry.dtype {
            DType::F16 => Ok(raw.to_vec()),
            DType::Bf16 => Ok(bf16_to_f16(raw)),
            DType::F32 => Ok(f32_to_f16(raw)),
            _ => Err(loader_err(
                model_dir,
                Some(entry.name.clone()),
                LoaderError::DtypeMismatch {
                    tensor: entry.name.clone(),
                    expected: DType::F16,
                    got: entry.dtype,
                },
            )),
        }
    }
}

fn load_layer(
    device: &Device,
    shards: &ShardSet,
    model_dir: &Path,
    arch: &Gemma4Arch,
    layer_idx: usize,
) -> Result<LayerWeights> {
    let prefix = format!("{}.layers.{layer_idx}", arch.weight_prefix);
    let ln = |suffix: &str| format!("{prefix}.{suffix}");
    let head_dim = arch.head_dim_for_layer(layer_idx);
    let q_dim = arch.num_attention_heads * head_dim;
    let kv_dim = arch.num_kv_heads_for_layer(layer_idx) * head_dim;
    let is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx(arch);
    let qkv_rows = if is_kv_shared_layer {
        q_dim
    } else {
        q_dim + 2 * kv_dim
    };

    let q = shards.must_tensor(&ln("self_attn.q_proj.weight"), model_dir)?;
    expect_shape(model_dir, &q, &[q_dim, arch.hidden_size])?;
    let qkv = if is_kv_shared_layer {
        concat_f16_tensors(shards, &[q], model_dir)?
    } else {
        let k = shards.must_tensor(&ln("self_attn.k_proj.weight"), model_dir)?;
        let v = shards.must_tensor(&ln("self_attn.v_proj.weight"), model_dir)?;
        expect_shape(model_dir, &k, &[kv_dim, arch.hidden_size])?;
        expect_shape(model_dir, &v, &[kv_dim, arch.hidden_size])?;
        concat_f16_tensors(shards, &[q, k, v], model_dir)?
    };

    let o = shards.must_tensor(&ln("self_attn.o_proj.weight"), model_dir)?;
    expect_shape(model_dir, &o, &[arch.hidden_size, q_dim])?;

    let gate = shards.must_tensor(&ln("mlp.gate_proj.weight"), model_dir)?;
    let up = shards.must_tensor(&ln("mlp.up_proj.weight"), model_dir)?;
    let intermediate = *gate.shape.first().unwrap_or(&0);
    expect_shape(model_dir, &gate, &[intermediate, arch.hidden_size])?;
    expect_shape(model_dir, &up, &[intermediate, arch.hidden_size])?;
    let gate_up = concat_f16_tensors(shards, &[gate, up], model_dir)?;

    let down = shards.must_tensor(&ln("mlp.down_proj.weight"), model_dir)?;
    expect_shape(model_dir, &down, &[arch.hidden_size, intermediate])?;

    let input_ln = load_norm(
        device,
        shards,
        model_dir,
        &ln("input_layernorm.weight"),
        arch.hidden_size,
    )?;
    let post_attn_ln = load_norm(
        device,
        shards,
        model_dir,
        &ln("post_attention_layernorm.weight"),
        arch.hidden_size,
    )?;
    let pre_ff_ln = load_norm(
        device,
        shards,
        model_dir,
        &ln("pre_feedforward_layernorm.weight"),
        arch.hidden_size,
    )?;
    let post_ff_ln = load_norm(
        device,
        shards,
        model_dir,
        &ln("post_feedforward_layernorm.weight"),
        arch.hidden_size,
    )?;
    let q_norm = load_norm(
        device,
        shards,
        model_dir,
        &ln("self_attn.q_norm.weight"),
        head_dim,
    )?;
    let k_norm = if is_kv_shared_layer {
        new_buffer_from_bytes(device, &vec![0u8; head_dim * std::mem::size_of::<u16>()])
    } else {
        load_norm(
            device,
            shards,
            model_dir,
            &ln("self_attn.k_norm.weight"),
            head_dim,
        )?
    };

    let per_layer_input_gate = shards.must_tensor(&ln("per_layer_input_gate.weight"), model_dir)?;
    expect_shape(
        model_dir,
        &per_layer_input_gate,
        &[PLE_DIM, arch.hidden_size],
    )?;
    let per_layer_projection = shards.must_tensor(&ln("per_layer_projection.weight"), model_dir)?;
    expect_shape(
        model_dir,
        &per_layer_projection,
        &[arch.hidden_size, PLE_DIM],
    )?;
    let post_per_layer_input_norm =
        shards.must_tensor(&ln("post_per_layer_input_norm.weight"), model_dir)?;
    expect_shape(model_dir, &post_per_layer_input_norm, &[arch.hidden_size])?;

    let layer_scalar = shards.must_tensor(&ln("layer_scalar"), model_dir)?;
    expect_shape(model_dir, &layer_scalar, &[1])?;

    if qkv.len() != qkv_rows * arch.hidden_size * std::mem::size_of::<u16>() {
        return Err(loader_err(
            model_dir,
            Some(format!("{prefix}.self_attn.qkv")),
            LoaderError::Corrupt {
                detail: "concatenated qkv byte length mismatch".into(),
            },
        ));
    }

    Ok(LayerWeights {
        intermediate,
        is_kv_shared_layer,
        qkv: new_buffer_from_bytes(device, &qkv),
        o_proj: new_buffer_from_bytes(device, &shards.tensor_to_f16_bytes(&o, model_dir)?),
        gate_up: new_buffer_from_bytes(device, &gate_up),
        down: new_buffer_from_bytes(device, &shards.tensor_to_f16_bytes(&down, model_dir)?),
        input_layernorm: input_ln,
        post_attention_layernorm: post_attn_ln,
        pre_feedforward_layernorm: pre_ff_ln,
        post_feedforward_layernorm: post_ff_ln,
        q_norm,
        k_norm,
        per_layer_input_gate: new_buffer_from_bytes(
            device,
            &shards.tensor_to_f16_bytes(&per_layer_input_gate, model_dir)?,
        ),
        per_layer_projection: new_buffer_from_bytes(
            device,
            &shards.tensor_to_f16_bytes(&per_layer_projection, model_dir)?,
        ),
        post_per_layer_input_norm: new_buffer_from_bytes(
            device,
            &shards.tensor_to_f16_bytes(&post_per_layer_input_norm, model_dir)?,
        ),
        layer_scalar: new_buffer_from_bytes(
            device,
            &shards.tensor_to_f16_bytes(&layer_scalar, model_dir)?,
        ),
    })
}

fn load_norm(
    device: &Device,
    shards: &ShardSet,
    model_dir: &Path,
    name: &str,
    dim: usize,
) -> Result<Buffer> {
    let entry = shards.must_tensor(name, model_dir)?;
    expect_shape(model_dir, &entry, &[dim])?;
    Ok(new_buffer_from_bytes(
        device,
        &shards.tensor_to_f16_bytes(&entry, model_dir)?,
    ))
}

fn validate_e2b_arch(arch: &Gemma4Arch) -> Result<()> {
    let ok = arch.hidden_size == 1536
        && arch.num_hidden_layers == 35
        && arch.num_attention_heads == 8
        && arch.head_dim_sliding == 256
        && arch.head_dim_global == 512
        && arch.num_kv_heads_sliding == 1
        && arch.num_kv_heads_global == 1
        && arch.intermediate_size == 6144
        && arch.vocab_size == 262144
        && arch.sliding_window_size == 512
        && arch.tie_word_embeddings;
    if ok {
        Ok(())
    } else {
        Err(apple_err(
            AppleError::LayerShapeInvalid {
                reason: "only google/gemma-4-E2B batch-1 shape is implemented",
            },
            "validate_e2b_arch",
        ))
    }
}

fn first_kv_shared_layer_idx(arch: &Gemma4Arch) -> usize {
    arch.num_hidden_layers.saturating_sub(NUM_KV_SHARED_LAYERS)
}

fn expect_shape(model_dir: &Path, entry: &TensorEntry, expected: &[usize]) -> Result<()> {
    if entry.shape == expected {
        Ok(())
    } else {
        Err(loader_err(
            model_dir,
            Some(entry.name.clone()),
            LoaderError::ShapeMismatch {
                tensor: entry.name.clone(),
                expected: expected.to_vec(),
                got: entry.shape.clone(),
            },
        ))
    }
}

fn concat_f16_tensors(
    shards: &ShardSet,
    entries: &[TensorEntry],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for entry in entries {
        out.extend_from_slice(&shards.tensor_to_f16_bytes(entry, model_dir)?);
    }
    Ok(out)
}

fn new_buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn new_f32_buffer(device: &Device, elements: usize) -> Buffer {
    device.new_buffer(
        (elements * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn pipeline(
    device: &Device,
    library: &metal_rs::Library,
    name: &'static str,
) -> Result<ComputePipelineState> {
    let function = library.get_function(name, None).map_err(|_| {
        apple_err(
            AppleError::InvalidMetalRecipe {
                reason: "Gemma4 Metal kernel symbol is missing",
            },
            name,
        )
    })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|_| {
            apple_err(
                AppleError::InvalidMetalRecipe {
                    reason: "Gemma4 Metal pipeline creation failed",
                },
                name,
            )
        })
}

fn set_pipeline(encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState) {
    encoder.set_compute_pipeline_state(pipeline);
}

fn set_bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}

fn dispatch_1d(encoder: &ComputeCommandEncoderRef, total: u64, threads: u64) {
    let groups = total.div_ceil(threads);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

fn dispatch_2d(
    encoder: &ComputeCommandEncoderRef,
    width: u64,
    height: u64,
    threads_x: u64,
    threads_y: u64,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: width.div_ceil(threads_x),
            height: height.div_ceil(threads_y),
            depth: 1,
        },
        MTLSize {
            width: threads_x,
            height: threads_y,
            depth: 1,
        },
    );
}

fn bf16_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for chunk in raw.chunks_exact(2) {
        let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
        let value = f32::from_bits((bf16 as u32) << 16);
        out.extend_from_slice(&f16::from_f32(value).to_le_bytes());
    }
    out
}

fn f32_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.extend_from_slice(&f16::from_f32(value).to_le_bytes());
    }
    out
}

fn scale_f16_bytes(bytes: &mut [u8], scale: f32) {
    for chunk in bytes.chunks_exact_mut(2) {
        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32() * scale;
        chunk.copy_from_slice(&f16::from_f32(value).to_le_bytes());
    }
}

fn debug_enabled() -> bool {
    std::env::var_os("RVLLM_APPLE_DEBUG").is_some()
}

fn debug_f32_stats(label: &str, buffer: &Buffer, elements: usize) {
    let ptr = buffer.contents() as *const f32;
    let values = unsafe { std::slice::from_raw_parts(ptr, elements) };
    let mut amax = 0.0f32;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
        } else {
            amax = amax.max(value.abs());
        }
    }
    let first: Vec<f32> = values.iter().copied().take(4).collect();
    eprintln!(
        "[apple-metal-debug] {label}: first4={first:.6?} amax={amax:.6e} nan={nan_count} inf={inf_count}"
    );
}

fn debug_top_logits(buffer: &Buffer, vocab: usize, k: usize) {
    let ptr = buffer.contents() as *const f32;
    let logits = unsafe { std::slice::from_raw_parts(ptr, vocab) };
    let mut top: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .take(k)
        .collect();
    for (idx, value) in logits.iter().copied().enumerate().skip(k) {
        if !value.is_finite() {
            continue;
        }
        if let Some((min_pos, _)) = top
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
        {
            if value > top[min_pos].1 {
                top[min_pos] = (idx, value);
            }
        }
    }
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("[apple-metal-debug] top logits: {top:?}");
}

fn loader_err(model_dir: &Path, tensor: Option<String>, err: LoaderError) -> RvllmError {
    RvllmError::Loader {
        err,
        ctx: LoaderCtx {
            path: PathBuf::from(model_dir),
            tensor,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn apple_err(err: AppleError, op: &'static str) -> RvllmError {
    RvllmError::Apple {
        err,
        ctx: AppleCtx {
            backend: BACKEND,
            op,
            device: "system-default-metal",
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}
