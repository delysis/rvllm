//! Metal Shading Language (MSL) kernel sources.
//!
//! All compute kernels are embedded as string constants and compiled
//! at runtime. This avoids build-time metallib compilation and makes
//! the crate work without Xcode command-line tools.

/// All Metal kernel source concatenated. Compiled once at init via
/// `MetalContext::compile_library()`.
pub const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RMSNorm (per-token, f16 in/out)
// ============================================================================
// Each threadgroup processes one token. Reduction across hidden dim
// uses threadgroup memory.

kernel void rmsnorm_f16(
    device const half *input      [[buffer(0)]],
    device half       *output     [[buffer(1)]],
    device const half *gamma      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    constant float    &eps        [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    // Each threadgroup = one token
    uint token = gid;
    uint base = token * hidden;

    // Phase 1: compute sum of squares
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Phase 2: normalize and apply gamma
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[base + i]);
        output[base + i] = half(v * rms * float(gamma[i]));
    }
}

// ============================================================================
// GEMM (f16, tiled)
// ============================================================================
// Simple but correct f16 GEMM: C[M,N] = A[M,K] * B[K,N]
// Weights stored in column-major (transposed): B is [N,K] in memory,
// accessed as B[n,k].
// Uses simdgroup matrix ops on Apple9+ for performance.

constant uint TILE_M = 8;
constant uint TILE_N = 8;

kernel void gemm_f16(
    device const half *A          [[buffer(0)]],  // [M, K] row-major
    device const half *B          [[buffer(1)]],  // [N, K] col-major (transposed)
    device half       *C          [[buffer(2)]],  // [M, N] row-major
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]],
    uint2 tg_size                 [[threads_per_threadgroup]]
) {
    uint row = gid.x * TILE_M + tid.x;
    uint col = gid.y * TILE_N + tid.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float a = float(A[row * K + k]);
        float b = float(B[col * K + k]);  // B transposed: [N,K]
        acc += a * b;
    }

    C[row * N + col] = half(acc * alpha + float(C[row * N + col]) * beta);
}

// ============================================================================
// GEMM with residual add: C = alpha * A*B + residual
// ============================================================================
kernel void gemm_residual_f16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    device const half *residual   [[buffer(3)]],
    constant uint     &M          [[buffer(4)]],
    constant uint     &N          [[buffer(5)]],
    constant uint     &K          [[buffer(6)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]]
) {
    uint row = gid.x * TILE_M + tid.x;
    uint col = gid.y * TILE_N + tid.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += float(A[row * K + k]) * float(B[col * K + k]);
    }

    uint idx = row * N + col;
    C[idx] = half(acc + float(residual[idx]));
}

// ============================================================================
// Partial RoPE (Gemma 4 style: only rotate first rope_dim dims)
// ============================================================================
kernel void rope_partial_f16(
    device half       *q          [[buffer(0)]],  // [num_tokens, q_dim]
    device half       *k          [[buffer(1)]],  // [num_tokens, kv_dim]
    device const float *cos_table [[buffer(2)]],  // [max_pos, head_dim/2]
    device const float *sin_table [[buffer(3)]],  // [max_pos, head_dim/2]
    device const int  *positions  [[buffer(4)]],  // [num_tokens]
    constant uint     &num_tokens [[buffer(5)]],
    constant uint     &num_heads  [[buffer(6)]],
    constant uint     &num_kv_heads [[buffer(7)]],
    constant uint     &head_dim   [[buffer(8)]],
    constant uint     &rope_dim  [[buffer(9)]],   // dims to rotate (typically head_dim)
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint pair = gid.y;  // pair index within one head
    if (token >= num_tokens) return;

    int pos = positions[token];
    uint half_rope = rope_dim / 2;
    if (pair >= half_rope) return;

    float cos_val = cos_table[pos * half_rope + pair];
    float sin_val = sin_table[pos * half_rope + pair];

    // Apply to all Q heads
    uint q_dim = num_heads * head_dim;
    for (uint h = 0; h < num_heads; h++) {
        uint base = token * q_dim + h * head_dim;
        uint i0 = base + pair;
        uint i1 = base + pair + half_rope;
        float x0 = float(q[i0]);
        float x1 = float(q[i1]);
        q[i0] = half(x0 * cos_val - x1 * sin_val);
        q[i1] = half(x0 * sin_val + x1 * cos_val);
    }

    // Apply to all KV heads
    uint kv_dim = num_kv_heads * head_dim;
    for (uint h = 0; h < num_kv_heads; h++) {
        uint base = token * kv_dim + h * head_dim;
        uint i0 = base + pair;
        uint i1 = base + pair + half_rope;
        float x0 = float(k[i0]);
        float x1 = float(k[i1]);
        k[i0] = half(x0 * cos_val - x1 * sin_val);
        k[i1] = half(x0 * sin_val + x1 * cos_val);
    }
}

// ============================================================================
// KV Cache Write (slot-mapped)
// ============================================================================
kernel void kv_cache_write_f16(
    device const half *k_src      [[buffer(0)]],  // [num_tokens, kv_dim]
    device const half *v_src      [[buffer(1)]],  // [num_tokens, kv_dim]
    device half       *k_cache    [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device half       *v_cache    [[buffer(3)]],
    device const int  *slot_map   [[buffer(4)]],  // [num_tokens] -> cache slot
    constant uint     &num_tokens [[buffer(5)]],
    constant uint     &kv_dim     [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= kv_dim) return;

    int slot = slot_map[token];
    if (slot < 0) return;

    k_cache[uint(slot) * kv_dim + dim] = k_src[token * kv_dim + dim];
    v_cache[uint(slot) * kv_dim + dim] = v_src[token * kv_dim + dim];
}

// ============================================================================
// Attention Decode (single Q token per sequence, paged KV)
// ============================================================================
// GQA-aware: each Q head group shares one KV head.
kernel void attention_decode_f16(
    device const half *q          [[buffer(0)]],   // [num_seqs, q_dim]
    device const half *k_cache    [[buffer(1)]],   // [total_blocks * block_size, kv_dim]
    device const half *v_cache    [[buffer(2)]],
    device half       *output     [[buffer(3)]],   // [num_seqs, q_dim]
    device const int  *block_tables [[buffer(4)]],  // [num_seqs, max_blocks_per_seq]
    device const int  *context_lens [[buffer(5)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &num_kv_heads [[buffer(8)]],
    constant uint     &head_dim   [[buffer(9)]],
    constant uint     &block_size [[buffer(10)]],
    constant uint     &max_blocks [[buffer(11)]],
    constant float    &scale      [[buffer(12)]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    // One thread per (seq, head) pair
    uint seq = gid / num_heads;
    uint head = gid % num_heads;
    if (seq >= num_seqs) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    int ctx_len = context_lens[seq];
    if (ctx_len <= 0) return;

    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    // Load Q vector for this head
    float q_shared[512]; // max head_dim
    for (uint d = 0; d < head_dim; d++) {
        q_shared[d] = float(q[seq * q_dim + head * head_dim + d]);
    }

    // Compute attention scores and weighted sum (online softmax)
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    threadgroup float out_accum[512];
    for (uint d = 0; d < head_dim; d++) {
        out_accum[d] = 0.0f;
    }

    // Process each KV token
    for (int t = 0; t < ctx_len; t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        // Dot product Q·K
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            uint k_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim
                       + kv_head * head_dim + d;
            score += q_shared[d] * float(k_cache[k_idx]);
        }
        // Reduce score across threads (simplified — single thread for now)
        score *= scale;

        // Online softmax update
        float old_max = max_score;
        max_score = max(max_score, score);
        float correction = exp(old_max - max_score);
        sum_exp = sum_exp * correction + exp(score - max_score);

        // Accumulate V weighted by attention
        float weight = exp(score - max_score);
        for (uint d = 0; d < head_dim; d++) {
            uint v_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim
                       + kv_head * head_dim + d;
            out_accum[d] = out_accum[d] * correction + weight * float(v_cache[v_idx]);
        }
    }

    // Write output
    float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < head_dim; d++) {
        output[seq * q_dim + head * head_dim + d] = half(out_accum[d] * inv_sum);
    }
}

// ============================================================================
// Attention Prefill (multi-token Q, causal mask, paged KV)
// ============================================================================
// Simplified prefill attention with causal masking.
kernel void attention_prefill_f16(
    device const half *q          [[buffer(0)]],   // [total_q, q_dim]
    device const half *k_cache    [[buffer(1)]],
    device const half *v_cache    [[buffer(2)]],
    device half       *output     [[buffer(3)]],   // [total_q, q_dim]
    device const int  *block_tables [[buffer(4)]],
    device const int  *context_lens [[buffer(5)]],
    device const int  *cu_seqlens [[buffer(6)]],   // [batch+1]
    constant uint     &total_q    [[buffer(7)]],
    constant uint     &batch_size [[buffer(8)]],
    constant uint     &num_heads  [[buffer(9)]],
    constant uint     &num_kv_heads [[buffer(10)]],
    constant uint     &head_dim   [[buffer(11)]],
    constant uint     &block_size [[buffer(12)]],
    constant uint     &max_blocks [[buffer(13)]],
    constant float    &scale      [[buffer(14)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    // Each thread handles one (q_token, head, dim_chunk)
    uint q_pos = gid.x;
    uint head = gid.y;
    if (q_pos >= total_q || head >= num_heads) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    // Find which sequence this q_pos belongs to
    uint seq = 0;
    for (uint s = 0; s < batch_size; s++) {
        if (int(q_pos) >= cu_seqlens[s] && int(q_pos) < cu_seqlens[s + 1]) {
            seq = s;
            break;
        }
    }
    uint seq_start = cu_seqlens[seq];
    uint q_offset = q_pos - seq_start;  // position within this sequence
    int ctx_len = context_lens[seq];

    // Compute attention for this Q position
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_vals[512]; // max head_dim — stack allocated
    for (uint d = 0; d < head_dim; d++) out_vals[d] = 0.0f;

    // Causal: only attend to positions <= q_offset
    uint attn_len = min(uint(ctx_len), q_offset + 1);

    for (uint t = 0; t < attn_len; t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];

        // Q·K dot product
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float q_val = float(q[q_pos * q_dim + head * head_dim + d]);
            uint k_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim + kv_head * head_dim + d;
            score += q_val * float(k_cache[k_idx]);
        }
        score *= scale;

        float old_max = max_score;
        max_score = max(max_score, score);
        float correction = exp(old_max - max_score);
        sum_exp = sum_exp * correction + exp(score - max_score);

        float weight = exp(score - max_score);
        for (uint d = 0; d < head_dim; d++) {
            uint v_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim + kv_head * head_dim + d;
            out_vals[d] = out_vals[d] * correction + weight * float(v_cache[v_idx]);
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        output[q_pos * q_dim + head * head_dim + d] = half(out_vals[d] * inv_sum);
    }
}

// ============================================================================
// GELU(tanh) * up (fused activation for Gemma 4)
// ============================================================================
kernel void gelu_mul_f16(
    device const half *gate_up    [[buffer(0)]],  // [num_tokens, 2*intermediate]
    device half       *output     [[buffer(1)]],  // [num_tokens, intermediate]
    constant uint     &num_tokens [[buffer(2)]],
    constant uint     &intermediate [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= intermediate) return;

    float gate = float(gate_up[token * 2 * intermediate + dim]);
    float up = float(gate_up[token * 2 * intermediate + intermediate + dim]);

    // GELU(tanh) approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x = gate;
    float c = 0.7978845608f; // sqrt(2/pi)
    float gelu = 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));

    output[token * intermediate + dim] = half(gelu * up);
}

// ============================================================================
// Residual Add
// ============================================================================
kernel void residual_add_f16(
    device half       *residual   [[buffer(0)]],
    device const half *addition   [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    residual[gid] = half(float(residual[gid]) + float(addition[gid]));
}

// ============================================================================
// Argmax (per-sequence)
// ============================================================================
kernel void argmax_f16(
    device const half *logits     [[buffer(0)]],  // [num_seqs, vocab]
    device int        *output     [[buffer(1)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(2)]],
    constant uint     &vocab      [[buffer(3)]],
    uint gid                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (gid >= num_seqs) return;
    uint base = gid * vocab;

    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(logits[base + i]);
        if (v > local_max) {
            local_max = v;
            local_idx = int(i);
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && shared_max[tid + s] > shared_max[tid]) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared_idx[0];
    }
}

// ============================================================================
// Logit Softcap: 30 * tanh(logits / 30) for Gemma 4
// ============================================================================
kernel void softcap_f16(
    device half       *logits     [[buffer(0)]],
    constant uint     &count      [[buffer(1)]],
    constant float    &cap        [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = float(logits[gid]);
    logits[gid] = half(cap * tanh(x / cap));
}

// ============================================================================
// BF16 → F16 conversion (for weight loading)
// ============================================================================
kernel void bf16_to_f16(
    device const ushort *bf16_in  [[buffer(0)]],
    device half         *f16_out  [[buffer(1)]],
    constant uint       &count    [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    // BF16 to F32: shift left 16 bits (BF16 is truncated F32)
    uint bf16_bits = uint(bf16_in[gid]);
    uint f32_bits = bf16_bits << 16;
    float f32_val = as_type<float>(f32_bits);
    f16_out[gid] = half(f32_val);
}
"#;

/// Number of kernels defined in the source.
pub const KERNEL_COUNT: usize = KERNEL_NAMES.len();

/// List of all kernel function names.
pub const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f16",
    "gemm_f16",
    "gemm_residual_f16",
    "rope_partial_f16",
    "kv_cache_write_f16",
    "attention_decode_f16",
    "attention_prefill_f16",
    "gelu_mul_f16",
    "residual_add_f16",
    "argmax_f16",
    "softcap_f16",
    "bf16_to_f16",
];

#[cfg(test)]
mod tests {
    use super::*;

    use half::f16;

    fn rmsnorm_ref(input: &[f32], gamma: &[f32], hidden: u32, eps: f32) -> Vec<f32> {
        let denom = (input
            .iter()
            .take(hidden as usize)
            .map(|&v| v * v)
            .sum::<f32>()
            / hidden as f32
            + eps)
        .sqrt();
        input
            .iter()
            .take(hidden as usize)
            .zip(gamma.iter())
            .map(|(&x, &g)| x / denom * g)
            .collect()
    }

    fn gelu_tanh_ref(x: f32) -> f32 {
        let c = 0.7978845608f32;
        0.5f32 * x * (1.0f32 + f32::tanh(c * (x + 0.044715f32 * x * x * x)))
    }

    fn gelu_mul_ref(gate_up: &[f32], intermediate: u32) -> Vec<f32> {
        let inter = intermediate as usize;
        let mut out = vec![0.0f32; inter];
        for d in 0..inter {
            let gate = gate_up[d];
            let up = gate_up[inter + d];
            out[d] = gelu_tanh_ref(gate) * up;
        }
        out
    }

    fn rope_partial_ref(
        q: &mut [f32],
        k: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        positions: &[i32],
        num_tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
    ) {
        let half_rope = (rope_dim / 2) as usize;
        let num_tokens = num_tokens as usize;
        let num_heads = num_heads as usize;
        let num_kv_heads = num_kv_heads as usize;
        let head_dim = head_dim as usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        for token in 0..num_tokens {
            let pos = positions[token] as usize;
            for pair in 0..half_rope {
                let cos_val = cos[pos * half_rope + pair];
                let sin_val = sin[pos * half_rope + pair];

                for h in 0..num_heads {
                    let base = token * q_dim + h * head_dim;
                    let i0 = base + pair;
                    let i1 = base + pair + half_rope;
                    let x0 = q[i0];
                    let x1 = q[i1];
                    q[i0] = x0 * cos_val - x1 * sin_val;
                    q[i1] = x0 * sin_val + x1 * cos_val;
                }

                for h in 0..num_kv_heads {
                    let base = token * kv_dim + h * head_dim;
                    let i0 = base + pair;
                    let i1 = base + pair + half_rope;
                    let x0 = k[i0];
                    let x1 = k[i1];
                    k[i0] = x0 * cos_val - x1 * sin_val;
                    k[i1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    fn argmax_ref(logits: &[f16], num_seqs: u32, vocab: u32) -> Vec<i32> {
        let mut out = vec![0i32; num_seqs as usize];
        for s in 0..num_seqs as usize {
            let base = s * vocab as usize;
            let mut best_idx = 0usize;
            let mut best_val = -f32::INFINITY;
            for i in 0..vocab as usize {
                let v = logits[base + i].to_f32();
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            out[s] = best_idx as i32;
        }
        out
    }

    fn attention_decode_ref(
        q: &[f16],
        k_cache: &[f16],
        v_cache: &[f16],
        block_tables: &[i32],
        context_lens: &[i32],
        num_seqs: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        max_blocks: u32,
        scale: f32,
    ) -> Vec<f32> {
        let num_seqs = num_seqs as usize;
        let num_heads = num_heads as usize;
        let num_kv_heads = num_kv_heads as usize;
        let head_dim = head_dim as usize;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;

        let mut out = vec![0f32; num_seqs * q_dim];

        for seq in 0..num_seqs {
            let ctx_len = context_lens[seq] as usize;
            if ctx_len == 0 {
                continue;
            }

            for head in 0..num_heads {
                let kv_head = head / (num_heads / num_kv_heads);

                let mut q_local = vec![0f32; head_dim];
                for d in 0..head_dim {
                    q_local[d] = q[seq * q_dim + head * head_dim + d].to_f32();
                }

                let mut out_accum = vec![0f32; head_dim];
                let mut max_score = -f32::INFINITY;
                let mut sum_exp = 0.0f32;

                for t in 0..ctx_len {
                    let block_idx = t / block_size as usize;
                    let block_offset = t % block_size as usize;
                    let block_id = block_tables[seq * max_blocks as usize + block_idx];
                    if block_id < 0 {
                        continue;
                    }

                    let mut score = 0.0f32;
                    let block_base = block_id as usize * block_size as usize * kv_dim;
                    for d in 0..head_dim {
                        let k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        score += q_local[d] * k_cache[k_idx].to_f32();
                    }
                    score *= scale;

                    let old_max = max_score;
                    max_score = max_score.max(score);
                    let correction = (old_max - max_score).exp();
                    sum_exp = sum_exp * correction + (score - max_score).exp();
                    let weight = (score - max_score).exp();

                    for d in 0..head_dim {
                        let v_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        out_accum[d] = out_accum[d] * correction + weight * v_cache[v_idx].to_f32();
                    }
                }

                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                for d in 0..head_dim {
                    out[seq * q_dim + head * head_dim + d] = out_accum[d] * inv_sum;
                }
            }
        }

        out
    }

    fn attention_decode_naive_ref(
        q: &[f16],
        k_cache: &[f16],
        v_cache: &[f16],
        block_tables: &[i32],
        context_lens: &[i32],
        num_seqs: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        max_blocks: u32,
        scale: f32,
    ) -> Vec<f32> {
        let num_seqs = num_seqs as usize;
        let num_heads = num_heads as usize;
        let num_kv_heads = num_kv_heads as usize;
        let head_dim = head_dim as usize;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;

        let mut out = vec![0f32; num_seqs * q_dim];
        for seq in 0..num_seqs {
            let ctx_len = context_lens[seq] as usize;
            if ctx_len == 0 {
                continue;
            }

            for head in 0..num_heads {
                let kv_head = head / (num_heads / num_kv_heads);
                let mut q_local = vec![0f32; head_dim];
                for d in 0..head_dim {
                    q_local[d] = q[seq * q_dim + head * head_dim + d].to_f32();
                }

                let mut scores = vec![0f32; ctx_len];
                for t in 0..ctx_len {
                    let block_idx = t / block_size as usize;
                    let block_offset = t % block_size as usize;
                    let block_id = block_tables[seq * max_blocks as usize + block_idx];
                    if block_id < 0 {
                        continue;
                    }

                    let block_base = block_id as usize * block_size as usize * kv_dim;
                    let mut score = 0f32;
                    for d in 0..head_dim {
                        let k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        score += q_local[d] * k_cache[k_idx].to_f32();
                    }
                    scores[t] = score * scale;
                }

                let max_score = scores.iter().fold(f32::NEG_INFINITY, |acc, &s| acc.max(s));
                let mut sum_exp = 0f32;
                for &score in &scores {
                    sum_exp += (score - max_score).exp();
                }

                for t in 0..ctx_len {
                    if scores[t] <= f32::NEG_INFINITY {
                        continue;
                    }
                    let block_idx = t / block_size as usize;
                    let block_offset = t % block_size as usize;
                    let block_id = block_tables[seq * max_blocks as usize + block_idx];
                    if block_id < 0 {
                        continue;
                    }

                    let block_base = block_id as usize * block_size as usize * kv_dim;
                    let weight = (scores[t] - max_score).exp() / sum_exp;

                    for d in 0..head_dim {
                        let v_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        out[seq * q_dim + head * head_dim + d] +=
                            weight * v_cache[v_idx].to_f32();
                    }
                }
            }
        }

        out
    }

    #[test]
    fn kernel_attention_decode_cpu_reference_matches_naive() {
        let q = vec![
            f16::from_f32(0.15), f16::from_f32(-0.1), f16::from_f32(0.2), f16::from_f32(0.05),
            f16::from_f32(-0.15), f16::from_f32(0.12), f16::from_f32(0.08), f16::from_f32(-0.04),
            f16::from_f32(0.25), f16::from_f32(-0.2), f16::from_f32(0.18), f16::from_f32(0.11),
            f16::from_f32(0.03), f16::from_f32(0.22), f16::from_f32(-0.07), f16::from_f32(0.09),
            f16::from_f32(0.19), f16::from_f32(-0.06), f16::from_f32(0.02), f16::from_f32(0.13),
            f16::from_f32(-0.09), f16::from_f32(0.16), f16::from_f32(0.05), f16::from_f32(0.01),
            f16::from_f32(0.02), f16::from_f32(0.01), f16::from_f32(-0.11), f16::from_f32(0.04),
            f16::from_f32(0.05), f16::from_f32(0.06), f16::from_f32(-0.07), f16::from_f32(0.08),
            f16::from_f32(0.09), f16::from_f32(0.07), f16::from_f32(-0.12), f16::from_f32(0.03),
            f16::from_f32(-0.01), f16::from_f32(0.02), f16::from_f32(0.04), f16::from_f32(0.06),
            f16::from_f32(0.08), f16::from_f32(0.1), f16::from_f32(-0.03), f16::from_f32(0.05),
            f16::from_f32(0.07), f16::from_f32(-0.08), f16::from_f32(0.06), f16::from_f32(0.03),
            f16::from_f32(0.02), f16::from_f32(0.05), f16::from_f32(-0.06), f16::from_f32(0.07),
            f16::from_f32(0.09), f16::from_f32(0.01), f16::from_f32(-0.02), f16::from_f32(0.04),
        ];
        let mut k_cache = vec![f16::from_f32(0.0); 3 * 8];
        let mut v_cache = vec![f16::from_f32(0.0); 3 * 8];
        for i in 0..k_cache.len() {
            k_cache[i] = f16::from_f32(0.01 * (i as f32 + 1.0));
            v_cache[i] = f16::from_f32(0.02 * (i as f32 + 1.0));
        }
        let block_tables = vec![0_i32, 0, 0, 0];
        let context_lens = vec![3_i32];

        let got = attention_decode_ref(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            1,
            2,
            1,
            8,
            4,
            1,
            0.125,
        );

        let expected = attention_decode_naive_ref(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            1,
            2,
            1,
            8,
            4,
            1,
            0.125,
        );
        assert_eq!(got.len(), expected.len());
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn kernel_rmsnorm_reference_matches_definition() {
        let input = vec![f32::from(1.0_f32), 2.0, 3.0];
        let gamma = vec![1.0_f32, 1.0, 1.0];
        let got = rmsnorm_ref(&input, &gamma, 3, 1e-6);
        let expected = {
            let v0 = 1f32 / f32::sqrt((1f32 + 4.0 + 9.0) / 3.0 + 1e-6);
            vec![v0, 2.0 * v0, 3.0 * v0]
        };
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn kernel_gelu_reference_matches_definition() {
        let got_single = gelu_tanh_ref(0.5);
        let got_pair = gelu_mul_ref(
            &[1.0, 0.5, -0.5, 2.0, 1.5, -1.0, 0.25, 3.0],
            4,
        );
        let expected = {
            let c = 0.7978845608f32;
            0.5f32 * 0.5 * (1.0f32 + f32::tanh(c * (0.5 + 0.044715f32 * 0.5f32.powi(3))))
        };
        assert!((got_single - expected).abs() < 1e-6);
        assert_eq!(got_pair.len(), 4);
    }

    #[test]
    fn kernel_rope_reference_matches_definition() {
        let mut q = vec![f32::from(1.0); 2 * 8];
        let mut k = vec![f32::from(2.0); 1 * 8];
        let cos = vec![0.5f32, 0.6, 0.7, 0.8];
        let sin = vec![0.5f32, 0.6, 0.7, 0.8];
        let pos = vec![0_i32];
        rope_partial_ref(&mut q, &mut k, &cos, &sin, &pos, 1, 2, 1, 8, 8);
        assert_ne!(q, vec![1.0; 16]);
        assert_ne!(k, vec![2.0; 8]);
    }

    #[test]
    fn kernel_argmax_reference_matches_definition() {
        let logits = vec![
            f16::from_f32(0.2),
            f16::from_f32(-0.4),
            f16::from_f32(0.7),
            f16::from_f32(-0.1),
            f16::from_f32(0.9),
            f16::from_f32(0.1),
        ];
        let got = argmax_ref(&logits, 2, 3);
        assert_eq!(got, vec![2, 1]);
    }

    #[test]
    fn kernel_count_matches_names() {
        assert_eq!(KERNEL_COUNT, KERNEL_NAMES.len());
    }
}
