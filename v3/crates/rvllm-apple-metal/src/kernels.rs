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

static inline half f16_sat(float x) {
    return half(clamp(x, -65504.0f, 65504.0f));
}

static inline float gelu_tanh(float x) {
    if (x >= 5.0f) {
        return x;
    }
    if (x <= -5.0f) {
        return 0.0f;
    }
    float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

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
        output[base + i] = f16_sat(v * rms * float(gamma[i]));
    }
}

kernel void rmsnorm_unit_f16(
    device const half *input      [[buffer(0)]],
    device half       *output     [[buffer(1)]],
    constant uint     &hidden     [[buffer(2)]],
    constant float    &eps        [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid;
    uint base = token * hidden;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(hidden) + eps);
    for (uint i = tid; i < hidden; i += tg_size) {
        output[base + i] = f16_sat(float(input[base + i]) * rms);
    }
}

kernel void rmsnorm_headwise_f16(
    device const half *input      [[buffer(0)]],
    device half       *output     [[buffer(1)]],
    device const half *gamma      [[buffer(2)]],
    constant uint     &head_dim   [[buffer(3)]],
    constant float    &eps        [[buffer(4)]],
    constant uint     &num_heads  [[buffer(5)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    uint hidden = num_heads * head_dim;
    uint base = token * hidden + head * head_dim;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        output[base + i] = f16_sat(v * rms * float(gamma[i]));
    }
}

kernel void rmsnorm_headwise_unit_f16(
    device const half *input      [[buffer(0)]],
    device half       *output     [[buffer(1)]],
    constant uint     &head_dim   [[buffer(2)]],
    constant float    &eps        [[buffer(3)]],
    constant uint     &num_heads  [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    uint hidden = num_heads * head_dim;
    uint base = token * hidden + head * head_dim;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        output[base + i] = f16_sat(float(input[base + i]) * rms);
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
constant uint TILE16 = 16;

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

    uint idx = row * N + col;
    float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
    C[idx] = f16_sat(acc * alpha + prior);
}

// Small/probe matrix tiled GEMM: C[M,N] = A[M,K] * B[K,N].
// B is stored transposed as [N,K]. This keeps the general kernel as the
// fallback and only gives dispatch a bounded aligned tile option.
kernel void gemm_f16_tiled16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]]
) {
    uint row = gid.x * TILE16 + tid.x;
    uint col = gid.y * TILE16 + tid.y;

    threadgroup half tile_a[16][16];
    threadgroup half tile_b[16][16];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE16) {
        uint a_col = k0 + tid.y;
        uint b_col = k0 + tid.x;
        tile_a[tid.x][tid.y] = (row < M && a_col < K) ? A[row * K + a_col] : half(0.0);
        tile_b[tid.x][tid.y] = (col < N && b_col < K) ? B[col * K + b_col] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE16; kk++) {
            acc += float(tile_a[tid.x][kk]) * float(tile_b[kk][tid.y]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        uint idx = row * N + col;
        float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
        C[idx] = f16_sat(acc * alpha + prior);
    }
}

kernel void gemm_rmsnorm_f16(
    device const half *A          [[buffer(0)]],  // [M, K] row-major
    device const half *B          [[buffer(1)]],  // [N, K] col-major (transposed)
    device const half *gamma      [[buffer(2)]],  // [N]
    device half       *C          [[buffer(3)]],  // [M, N] row-major
    constant uint     &M          [[buffer(4)]],
    constant uint     &N          [[buffer(5)]],
    constant uint     &K          [[buffer(6)]],
    constant float    &eps        [[buffer(7)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint row                      [[threadgroup_position_in_grid]]
) {
    if (row >= M) return;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint col = tid; col < N; col += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]);
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(N) + eps);
    for (uint col = tid; col < N; col += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]);
        }
        C[row * N + col] = f16_sat(acc * rms * float(gamma[col]));
    }
}

kernel void gemm_headwise_rmsnorm_f16(
    device const half *A          [[buffer(0)]],  // [M, K] row-major
    device const half *B          [[buffer(1)]],  // [total_rows, K] col-major (transposed)
    device const half *gamma      [[buffer(2)]],  // [head_dim]
    device half       *C          [[buffer(3)]],  // [M, num_heads * head_dim]
    constant uint     &M          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant uint     &head_dim   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &b_row_offset [[buffer(8)]],
    constant float    &eps        [[buffer(9)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    if (token >= M) return;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        C[c_head_base + d] = f16_sat(acc * rms * float(gamma[d]));
    }
}

kernel void gemm_headwise_rmsnorm_unit_f16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &K          [[buffer(4)]],
    constant uint     &head_dim   [[buffer(5)]],
    constant uint     &num_heads  [[buffer(6)]],
    constant uint     &b_row_offset [[buffer(7)]],
    constant float    &eps        [[buffer(8)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    if (token >= M) return;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        C[c_head_base + d] = f16_sat(acc * rms);
    }
}

kernel void copy_f16(
    device const half *src [[buffer(0)]],
    device half       *dst [[buffer(1)]],
    constant uint     &len [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    dst[gid] = src[gid];
}

kernel void copy_ple_layer_f16(
    device const half *packed_ple [[buffer(0)]],
    device half       *dst        [[buffer(1)]],
    constant uint     &num_tokens [[buffer(2)]],
    constant uint     &num_layers [[buffer(3)]],
    constant uint     &layer_idx  [[buffer(4)]],
    constant uint     &ple_dim    [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= ple_dim || layer_idx >= num_layers) return;
    uint src_idx = token * num_layers * ple_dim + layer_idx * ple_dim + dim;
    dst[token * ple_dim + dim] = packed_ple[src_idx];
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
    device const half *layer_scale [[buffer(7)]],
    constant uint     &layer_scale_dim [[buffer(8)]],
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
    float scale = 1.0f;
    if (layer_scale_dim == 1) {
        scale = float(layer_scale[0]);
    } else if (layer_scale_dim == N) {
        scale = float(layer_scale[col]);
    }
    C[idx] = f16_sat(acc * scale + float(residual[idx]));
}

// ============================================================================
// Split fused QKV (interleaved) into planar Q/K/V.
// qkv stores [token][q_dim + 2*kv_dim] while kernels downstream expect
// contiguous planar Q, K, V regions.
// ============================================================================
kernel void split_qkv_f16(
    device const half *qkv   [[buffer(0)]],  // [num_tokens, q_dim + 2*kv_dim]
    device half *q           [[buffer(1)]],  // [num_tokens, q_dim]
    device half *k           [[buffer(2)]],  // [num_tokens, kv_dim]
    device half *v           [[buffer(3)]],  // [num_tokens, kv_dim]
    constant uint &num_tokens [[buffer(4)]],
    constant uint &q_dim      [[buffer(5)]],
    constant uint &kv_dim     [[buffer(6)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens) return;

    uint qkv_dim = q_dim + 2u * kv_dim;
    if (dim < q_dim) {
        q[token * q_dim + dim] = qkv[token * qkv_dim + dim];
    } else if (dim < q_dim + kv_dim) {
        uint kd = dim - q_dim;
        k[token * kv_dim + kd] = qkv[token * qkv_dim + q_dim + kd];
    } else if (dim < q_dim + 2u * kv_dim) {
        uint vd = dim - q_dim - kv_dim;
        v[token * kv_dim + vd] = qkv[token * qkv_dim + q_dim + kv_dim + vd];
    }
}

// ============================================================================
// Embedding gather (row lookup) f16
// ==========================================================================
kernel void embedding_gather_f16(
    device const half *embedding   [[buffer(0)]],
    device const uint *token_ids    [[buffer(1)]],
    device half       *out         [[buffer(2)]],
    constant uint     &num_tokens  [[buffer(3)]],
    constant uint     &hidden      [[buffer(4)]],
    constant uint     &vocab       [[buffer(5)]],
    constant float    &scale       [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= hidden) return;

    uint tok = token_ids[token];
    if (tok >= vocab) {
        out[token * hidden + dim] = half(0.0);
        return;
    }

    out[token * hidden + dim] = f16_sat(float(embedding[tok * hidden + dim]) * scale);
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
        uint i1 = base + pair + head_dim / 2;
        float x0 = float(q[i0]);
        float x1 = float(q[i1]);
        q[i0] = f16_sat(x0 * cos_val - x1 * sin_val);
        q[i1] = f16_sat(x0 * sin_val + x1 * cos_val);
    }

    // Apply to all KV heads
    uint kv_dim = num_kv_heads * head_dim;
    for (uint h = 0; h < num_kv_heads; h++) {
        uint base = token * kv_dim + h * head_dim;
        uint i0 = base + pair;
        uint i1 = base + pair + head_dim / 2;
        float x0 = float(k[i0]);
        float x1 = float(k[i1]);
        k[i0] = f16_sat(x0 * cos_val - x1 * sin_val);
        k[i1] = f16_sat(x0 * sin_val + x1 * cos_val);
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

// Experimental KV cache compression utilities. These kernels are not used by
// the default F16 probe path; tests opt in explicitly and dequantize back to
// F16 before comparison. Quantization is symmetric per cache row.
kernel void experimental_kv_quantize_int8_f16(
    device const half *src        [[buffer(0)]],  // [num_rows, kv_dim]
    device char       *dst        [[buffer(1)]],  // [num_rows, kv_dim]
    device float      *scales     [[buffer(2)]],  // [num_rows]
    constant uint     &num_rows   [[buffer(3)]],
    constant uint     &kv_dim     [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint row                      [[threadgroup_position_in_grid]]
) {
    if (row >= num_rows) return;

    threadgroup float shared_max[256];
    float local_max = 0.0f;
    uint base = row * kv_dim;
    for (uint dim = tid; dim < kv_dim; dim += tg_size) {
        local_max = fmax(local_max, fabs(float(src[base + dim])));
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = shared_max[0] > 0.0f ? shared_max[0] / 127.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < kv_dim; dim += tg_size) {
        float q = round(float(src[base + dim]) / scale);
        q = clamp(q, -127.0f, 127.0f);
        dst[base + dim] = char(q);
    }
}

kernel void experimental_kv_dequantize_int8_f16(
    device const char  *src       [[buffer(0)]],  // [num_rows, kv_dim]
    device const float *scales    [[buffer(1)]],  // [num_rows]
    device half        *dst       [[buffer(2)]],  // [num_rows, kv_dim]
    constant uint      &num_rows  [[buffer(3)]],
    constant uint      &kv_dim    [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint total = num_rows * kv_dim;
    if (gid >= total) return;
    uint row = gid / kv_dim;
    dst[gid] = f16_sat(float(src[gid]) * scales[row]);
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
        output[seq * q_dim + head * head_dim + d] = f16_sat(out_accum[d] * inv_sum);
    }
}

// Parallel decode attention for bounded probe shapes. One threadgroup handles
// one (sequence, Q head), parallelizing max/sum reductions across context
// tokens. Dispatch keeps attention_decode_f16 as the correctness fallback.
kernel void attention_decode_reduction_f16(
    device const half *q          [[buffer(0)]],
    device const half *k_cache    [[buffer(1)]],
    device const half *v_cache    [[buffer(2)]],
    device half       *output     [[buffer(3)]],
    device const int  *block_tables [[buffer(4)]],
    device const int  *context_lens [[buffer(5)]],
    constant uint     &num_seqs   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &num_kv_heads [[buffer(8)]],
    constant uint     &head_dim   [[buffer(9)]],
    constant uint     &block_size [[buffer(10)]],
    constant uint     &max_blocks [[buffer(11)]],
    constant float    &scale      [[buffer(12)]],
    uint gid                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    uint seq = gid / num_heads;
    uint head = gid % num_heads;
    if (seq >= num_seqs || head_dim > 256) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    int ctx_len_i = context_lens[seq];
    if (ctx_len_i <= 0) return;
    uint ctx_len = uint(ctx_len_i);

    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    threadgroup float shared[256];

    float local_max = -INFINITY;
    for (uint t = tid; t < ctx_len; t += tg_size) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        float score = 0.0f;
        uint block_base = uint(block_id) * block_size * kv_dim;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = seq * q_dim + head * head_dim + d;
            uint k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
            score += float(q[q_idx]) * float(k_cache[k_idx]);
        }
        local_max = max(local_max, score * scale);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared[0];

    float local_sum = 0.0f;
    for (uint t = tid; t < ctx_len; t += tg_size) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        float score = 0.0f;
        uint block_base = uint(block_id) * block_size * kv_dim;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = seq * q_dim + head * head_dim + d;
            uint k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
            score += float(q[q_idx]) * float(k_cache[k_idx]);
        }
        local_sum += exp(score * scale - max_score);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < ctx_len; t++) {
            uint block_idx = t / block_size;
            uint block_offset = t % block_size;
            int block_id = block_tables[seq * max_blocks + block_idx];
            if (block_id < 0) continue;

            float score = 0.0f;
            uint block_base = uint(block_id) * block_size * kv_dim;
            for (uint kd = 0; kd < head_dim; kd++) {
                uint q_idx = seq * q_dim + head * head_dim + kd;
                uint k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + kd;
                score += float(q[q_idx]) * float(k_cache[k_idx]);
            }
            float weight = exp(score * scale - max_score) * inv_sum;
            uint v_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
            acc += weight * float(v_cache[v_idx]);
        }
        output[seq * q_dim + head * head_dim + d] = f16_sat(acc);
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
        output[q_pos * q_dim + head * head_dim + d] = f16_sat(out_vals[d] * inv_sum);
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

    output[token * intermediate + dim] = f16_sat(gelu_tanh(gate) * up);
}

kernel void ple_combine_f16(
    device half       *token_ple   [[buffer(0)]],
    device const half *context_ple [[buffer(1)]],
    constant uint     &num_tokens  [[buffer(2)]],
    constant uint     &stride      [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= stride) return;
    uint idx = token * stride + dim;
    token_ple[idx] = f16_sat((float(token_ple[idx]) + float(context_ple[idx])) * 0.70710678118f);
}

kernel void ple_gelu_mul_f16(
    device half       *gate        [[buffer(0)]],
    device const half *packed_ple  [[buffer(1)]],
    constant uint     &num_tokens  [[buffer(2)]],
    constant uint     &num_layers  [[buffer(3)]],
    constant uint     &layer_idx   [[buffer(4)]],
    constant uint     &ple_dim     [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= ple_dim || layer_idx >= num_layers) return;
    uint gate_idx = token * ple_dim + dim;
    uint ple_idx = token * num_layers * ple_dim + layer_idx * ple_dim + dim;
    gate[gate_idx] = f16_sat(gelu_tanh(float(gate[gate_idx])) * float(packed_ple[ple_idx]));
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
    residual[gid] = f16_sat(float(residual[gid]) + float(addition[gid]));
}

kernel void residual_add_rmsnorm_f16(
    device half       *residual   [[buffer(0)]],
    device const half *addition   [[buffer(1)]],
    device half       *output     [[buffer(2)]],
    device const half *gamma      [[buffer(3)]],
    constant uint     &hidden     [[buffer(4)]],
    constant float    &eps        [[buffer(5)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid;
    uint base = token * hidden;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = base + i;
        half updated = f16_sat(float(residual[idx]) + float(addition[idx]));
        residual[idx] = updated;
        float v = float(updated);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(hidden) + eps);
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = base + i;
        output[idx] = f16_sat(float(residual[idx]) * rms * float(gamma[i]));
    }
}

kernel void layer_scale_f16(
    device half       *x          [[buffer(0)]],
    device const half *scale      [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    constant uint     &scale_dim  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float s = 1.0f;
    if (scale_dim == 1) {
        s = float(scale[0]);
    } else if (scale_dim == hidden) {
        s = float(scale[gid % hidden]);
    }
    x[gid] = f16_sat(float(x[gid]) * s);
}

// ============================================================================
// Argmax (per-sequence)
// ============================================================================
static inline bool argmax_better(float candidate_val, int candidate_idx, float best_val, int best_idx) {
    return candidate_val > best_val || (candidate_val == best_val && candidate_idx < best_idx);
}

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

kernel void softcap_argmax_f16(
    device half       *logits     [[buffer(0)]],  // [num_seqs, vocab]
    device int        *output     [[buffer(1)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(2)]],
    constant uint     &vocab      [[buffer(3)]],
    constant float    &cap        [[buffer(4)]],
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
        half capped = f16_sat(cap * tanh(float(logits[base + i]) / cap));
        logits[base + i] = capped;
        float v = float(capped);
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
// Fused final RMSNorm + lm_head + optional softcap + argmax
// ============================================================================
// Bounded small-vocab probe path. The unfused RMSNorm/GEMM/softcap/argmax path
// remains the fallback for wider shapes.
kernel void final_norm_lm_head_argmax_small_f16(
    device const half *residual   [[buffer(0)]],
    device const half *gamma      [[buffer(1)]],
    device const half *lm_head    [[buffer(2)]],
    device half       *logits     [[buffer(3)]],
    device int        *output     [[buffer(4)]],
    constant uint     &hidden     [[buffer(5)]],
    constant uint     &vocab      [[buffer(6)]],
    constant float    &eps        [[buffer(7)]],
    constant float    &softcap    [[buffer(8)]],
    uint token                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    uint base = token * hidden;
    float local_sum = 0.0f;
    for (uint d = tid; d < hidden; d += tg_size) {
        float v = float(residual[base + d]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_sum[0] / float(hidden) + eps);

    float local_max = -INFINITY;
    int local_idx = 0;
    for (uint v = tid; v < vocab; v += tg_size) {
        float acc = 0.0f;
        for (uint d = 0; d < hidden; d++) {
            float normed = float(residual[base + d]) * inv_rms * float(gamma[d]);
            acc += normed * float(lm_head[v * hidden + d]);
        }
        if (softcap > 0.0f) {
            acc = softcap * tanh(acc / softcap);
        }
        logits[token * vocab + v] = f16_sat(acc);
        if (argmax_better(acc, int(v), local_max, local_idx)) {
            local_max = acc;
            local_idx = int(v);
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && argmax_better(shared_max[tid + s], shared_idx[tid + s], shared_max[tid], shared_idx[tid])) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[token] = shared_idx[0];
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
    logits[gid] = f16_sat(cap * tanh(x / cap));
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
    f16_out[gid] = f16_sat(f32_val);
}
"#;

/// Number of kernels defined in the source.
pub const KERNEL_COUNT: usize = KERNEL_NAMES.len();

/// List of all kernel function names.
pub const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f16",
    "rmsnorm_unit_f16",
    "rmsnorm_headwise_f16",
    "rmsnorm_headwise_unit_f16",
    "gemm_f16",
    "gemm_f16_tiled16",
    "gemm_rmsnorm_f16",
    "gemm_headwise_rmsnorm_f16",
    "gemm_headwise_rmsnorm_unit_f16",
    "copy_f16",
    "copy_ple_layer_f16",
    "gemm_residual_f16",
    "split_qkv_f16",
    "rope_partial_f16",
    "kv_cache_write_f16",
    "experimental_kv_quantize_int8_f16",
    "experimental_kv_dequantize_int8_f16",
    "attention_decode_f16",
    "attention_decode_reduction_f16",
    "attention_prefill_f16",
    "embedding_gather_f16",
    "gelu_mul_f16",
    "ple_combine_f16",
    "ple_gelu_mul_f16",
    "residual_add_f16",
    "residual_add_rmsnorm_f16",
    "layer_scale_f16",
    "argmax_f16",
    "softcap_argmax_f16",
    "final_norm_lm_head_argmax_small_f16",
    "softcap_f16",
    "bf16_to_f16",
];

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentalKvInt8Reference {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub kv_dim: usize,
}

/// CPU reference for experimental F16 KV-cache row compression.
///
/// This is intentionally separate from the production F16 cache path. It uses
/// one symmetric int8 scale per cache row and leaves zero rows with scale 1.0.
pub fn experimental_quantize_kv_f16_to_int8_reference(
    src: &[half::f16],
    kv_dim: usize,
) -> ExperimentalKvInt8Reference {
    assert!(kv_dim > 0, "kv_dim must be nonzero");
    assert_eq!(src.len() % kv_dim, 0, "src must contain whole KV rows");

    let rows = src.len() / kv_dim;
    let mut values = vec![0_i8; src.len()];
    let mut scales = vec![1.0_f32; rows];

    for row in 0..rows {
        let base = row * kv_dim;
        let mut max_abs = 0.0_f32;
        for dim in 0..kv_dim {
            max_abs = max_abs.max(src[base + dim].to_f32().abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales[row] = scale;
        for dim in 0..kv_dim {
            let q = (src[base + dim].to_f32() / scale)
                .round()
                .clamp(-127.0, 127.0);
            values[base + dim] = q as i8;
        }
    }

    ExperimentalKvInt8Reference {
        values,
        scales,
        kv_dim,
    }
}

/// CPU reference dequantization for experimental int8 KV rows.
pub fn experimental_dequantize_kv_int8_to_f16_reference(
    quantized: &ExperimentalKvInt8Reference,
) -> Vec<half::f16> {
    assert!(quantized.kv_dim > 0, "kv_dim must be nonzero");
    assert_eq!(
        quantized.values.len() % quantized.kv_dim,
        0,
        "values must contain whole KV rows"
    );
    assert_eq!(
        quantized.values.len() / quantized.kv_dim,
        quantized.scales.len(),
        "scale count must match KV rows"
    );

    let mut out = Vec::with_capacity(quantized.values.len());
    for row in 0..quantized.scales.len() {
        let base = row * quantized.kv_dim;
        let scale = quantized.scales[row];
        for dim in 0..quantized.kv_dim {
            out.push(half::f16::from_f32(
                quantized.values[base + dim] as f32 * scale,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    use crate::{arena::MetalBufferArena, context::MetalContext, pipeline::PipelineCache};
    use half::f16;
    #[cfg(target_os = "macos")]
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    #[cfg(target_os = "macos")]
    fn metal_test_context(
        arena_bytes: usize,
    ) -> rvllm_core::Result<(MetalContext, PipelineCache, MetalBufferArena)> {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(crate::kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let arena = MetalBufferArena::new(ctx.device(), arena_bytes)?;
        Ok((ctx, pipelines, arena))
    }

    #[cfg(target_os = "macos")]
    unsafe fn write_f16_region(
        arena: &MetalBufferArena,
        region: &crate::arena::MetalRegion,
        values: &[f16],
    ) {
        let ptr = arena.host_ptr(region) as *mut f16;
        for (idx, value) in values.iter().enumerate() {
            *ptr.add(idx) = *value;
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn read_f16_region(
        arena: &MetalBufferArena,
        region: &crate::arena::MetalRegion,
        len: usize,
    ) -> Vec<f32> {
        std::slice::from_raw_parts(arena.host_ptr(region) as *const f16, len)
            .iter()
            .map(|value| value.to_f32())
            .collect()
    }

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

    fn headwise_rmsnorm_ref(
        input: &[f32],
        gamma: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        assert_eq!(gamma.len(), head_dim);
        let hidden = num_heads * head_dim;
        let mut out = vec![0.0f32; num_tokens * hidden];
        for token in 0..num_tokens {
            for head in 0..num_heads {
                let base = token * hidden + head * head_dim;
                let mean_sq = input[base..base + head_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / head_dim as f32;
                let rms = (mean_sq + eps).sqrt();
                for dim in 0..head_dim {
                    out[base + dim] = input[base + dim] / rms * gamma[dim];
                }
            }
        }
        out
    }

    fn gemm_headwise_rmsnorm_ref(
        a: &[f32],
        b: &[f32],
        gamma: Option<&[f32]>,
        m: usize,
        k: usize,
        head_dim: usize,
        num_heads: usize,
        b_row_offset: usize,
        eps: f32,
    ) -> Vec<f32> {
        if let Some(gamma) = gamma {
            assert_eq!(gamma.len(), head_dim);
        }
        let mut out = vec![0.0f32; m * num_heads * head_dim];
        for token in 0..m {
            for head in 0..num_heads {
                let mut raw = vec![0.0f32; head_dim];
                for dim in 0..head_dim {
                    let row = b_row_offset + head * head_dim + dim;
                    for kk in 0..k {
                        raw[dim] += a[token * k + kk] * b[row * k + kk];
                    }
                }
                let mean_sq = raw.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
                let rms = (mean_sq + eps).sqrt();
                let base = token * num_heads * head_dim + head * head_dim;
                for dim in 0..head_dim {
                    let scale = gamma.map_or(1.0, |gamma| gamma[dim]);
                    out[base + dim] = raw[dim] / rms * scale;
                }
            }
        }
        out
    }

    fn gemm_rmsnorm_ref(
        a: &[f32],
        b: &[f32],
        gamma: &[f32],
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Vec<f32> {
        assert_eq!(gamma.len(), n);
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            let mut raw = vec![0.0f32; n];
            for col in 0..n {
                for kk in 0..k {
                    raw[col] += a[row * k + kk] * b[col * k + kk];
                }
            }
            let mean_sq = raw.iter().map(|value| value * value).sum::<f32>() / n as f32;
            let rms = (mean_sq + eps).sqrt();
            for col in 0..n {
                out[row * n + col] = raw[col] / rms * gamma[col];
            }
        }
        out
    }

    fn gemm_ref_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[row * k + kk] * b[col * k + kk];
                }
                out[row * n + col] = acc;
            }
        }
        out
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
                    let i1 = base + pair + head_dim / 2;
                    let x0 = q[i0];
                    let x1 = q[i1];
                    q[i0] = x0 * cos_val - x1 * sin_val;
                    q[i1] = x0 * sin_val + x1 * cos_val;
                }

                for h in 0..num_kv_heads {
                    let base = token * kv_dim + h * head_dim;
                    let i0 = base + pair;
                    let i1 = base + pair + head_dim / 2;
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
                        out[seq * q_dim + head * head_dim + d] += weight * v_cache[v_idx].to_f32();
                    }
                }
            }
        }

        out
    }

    #[test]
    fn kernel_attention_decode_cpu_reference_matches_naive() {
        let q = vec![
            f16::from_f32(0.15),
            f16::from_f32(-0.1),
            f16::from_f32(0.2),
            f16::from_f32(0.05),
            f16::from_f32(-0.15),
            f16::from_f32(0.12),
            f16::from_f32(0.08),
            f16::from_f32(-0.04),
            f16::from_f32(0.25),
            f16::from_f32(-0.2),
            f16::from_f32(0.18),
            f16::from_f32(0.11),
            f16::from_f32(0.03),
            f16::from_f32(0.22),
            f16::from_f32(-0.07),
            f16::from_f32(0.09),
            f16::from_f32(0.19),
            f16::from_f32(-0.06),
            f16::from_f32(0.02),
            f16::from_f32(0.13),
            f16::from_f32(-0.09),
            f16::from_f32(0.16),
            f16::from_f32(0.05),
            f16::from_f32(0.01),
            f16::from_f32(0.02),
            f16::from_f32(0.01),
            f16::from_f32(-0.11),
            f16::from_f32(0.04),
            f16::from_f32(0.05),
            f16::from_f32(0.06),
            f16::from_f32(-0.07),
            f16::from_f32(0.08),
            f16::from_f32(0.09),
            f16::from_f32(0.07),
            f16::from_f32(-0.12),
            f16::from_f32(0.03),
            f16::from_f32(-0.01),
            f16::from_f32(0.02),
            f16::from_f32(0.04),
            f16::from_f32(0.06),
            f16::from_f32(0.08),
            f16::from_f32(0.1),
            f16::from_f32(-0.03),
            f16::from_f32(0.05),
            f16::from_f32(0.07),
            f16::from_f32(-0.08),
            f16::from_f32(0.06),
            f16::from_f32(0.03),
            f16::from_f32(0.02),
            f16::from_f32(0.05),
            f16::from_f32(-0.06),
            f16::from_f32(0.07),
            f16::from_f32(0.09),
            f16::from_f32(0.01),
            f16::from_f32(-0.02),
            f16::from_f32(0.04),
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
        let got_pair = gelu_mul_ref(&[1.0, 0.5, -0.5, 2.0, 1.5, -1.0, 0.25, 3.0], 4);
        let expected = {
            let c = 0.7978845608f32;
            0.5f32 * 0.5 * (1.0f32 + f32::tanh(c * (0.5 + 0.044715f32 * 0.5f32.powi(3))))
        };
        assert!((got_single - expected).abs() < 1e-6);
        assert_eq!(got_pair.len(), 4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gelu_mul_large_intermediate_matches_cpu_and_overwrites_output() -> rvllm_core::Result<()> {
        const NUM_TOKENS: u32 = 2;
        const INTERMEDIATE: u32 = 6144;
        let half_bytes = std::mem::size_of::<f16>();
        let gate_up_len = (NUM_TOKENS * 2 * INTERMEDIATE) as usize;
        let output_len = (NUM_TOKENS * INTERMEDIATE) as usize;
        let (ctx, pipelines, mut arena) =
            metal_test_context((gate_up_len + output_len + 4096) * half_bytes)?;
        let gate_up_region = arena.region("gelu_gate_up", gate_up_len * half_bytes, 16)?;
        let output_region = arena.region("gelu_output", output_len * half_bytes, 16)?;

        let mut gate_up = vec![f16::ZERO; gate_up_len];
        let mut expected = vec![0.0f32; output_len];
        for token in 0..NUM_TOKENS as usize {
            let gate_base = token * 2 * INTERMEDIATE as usize;
            let up_base = gate_base + INTERMEDIATE as usize;
            let out_base = token * INTERMEDIATE as usize;
            for dim in 0..INTERMEDIATE as usize {
                let gate = ((dim % 97) as f32 - 48.0) / 3.0;
                let up = (((dim * 7 + token * 13) % 113) as f32 - 56.0) / 4.0;
                gate_up[gate_base + dim] = f16::from_f32(gate);
                gate_up[up_base + dim] = f16::from_f32(up);
                expected[out_base + dim] = f16::from_f32(gelu_tanh_ref(gate) * up).to_f32();
            }
        }
        unsafe {
            write_f16_region(&arena, &gate_up_region, &gate_up);
            write_f16_region(&arena, &output_region, &vec![f16::NAN; output_len]);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "gelu_mul_large_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "gelu_mul_large_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encoder.setComputePipelineState(pipelines.get("gelu_mul_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), gate_up_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), output_region.offset, 1);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_TOKENS as *const _ as *mut _),
                4,
                2,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&INTERMEDIATE as *const _ as *mut _),
                4,
                3,
            );
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: NUM_TOKENS as usize,
                    height: INTERMEDIATE as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got = unsafe { read_f16_region(&arena, &output_region, output_len) };
        let mut max_delta = 0.0f32;
        let mut max_abs = 0.0f32;
        for (idx, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(actual.is_finite(), "gelu output[{idx}] should be finite");
            max_abs = max_abs.max(actual.abs());
            max_delta = max_delta.max((actual - want).abs());
            if actual.abs() >= 1024.0 {
                let token = idx / INTERMEDIATE as usize;
                let dim = idx % INTERMEDIATE as usize;
                let gate_base = token * 2 * INTERMEDIATE as usize;
                let up_base = gate_base + INTERMEDIATE as usize;
                eprintln!(
                    "large gelu output idx={idx} token={token} dim={dim} actual={actual} want={want} gate={} up={}",
                    gate_up[gate_base + dim].to_f32(),
                    gate_up[up_base + dim].to_f32()
                );
                break;
            }
        }
        assert!(
            max_abs < 256.0,
            "gelu large-intermediate output unexpectedly large: {max_abs}"
        );
        assert!(
            max_delta <= 0.25,
            "gelu large-intermediate max delta {max_delta}"
        );
        Ok(())
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
    fn kernel_rope_reference_partial_pairs_across_full_split_half() {
        let mut q = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let mut k = (10..18).map(|v| v as f32).collect::<Vec<_>>();
        let cos = vec![0.0f32];
        let sin = vec![1.0f32];
        let pos = vec![0_i32];

        rope_partial_ref(&mut q, &mut k, &cos, &sin, &pos, 1, 1, 1, 8, 2);

        assert_eq!(q[0], -4.0);
        assert_eq!(q[4], 0.0);
        assert_eq!(q[1], 1.0);
        assert_eq!(k[0], -14.0);
        assert_eq!(k[4], 10.0);
        assert_eq!(k[1], 11.0);
    }

    #[test]
    fn kernel_rmsnorm_headwise_reference_uses_head_dim_gamma_per_head() {
        let input = vec![3.0f32, 4.0, 30.0, 40.0];
        let gamma = vec![1.0f32, 2.0];
        let got = headwise_rmsnorm_ref(&input, &gamma, 1, 2, 2, 1e-6);
        let flat = rmsnorm_ref(&input, &[1.0, 2.0, 1.0, 2.0], 4, 1e-6);

        assert!((got[0] - 3.0 / 12.5f32.sqrt()).abs() < 1e-6);
        assert!((got[1] - 8.0 / 12.5f32.sqrt()).abs() < 1e-6);
        assert!((got[2] - 30.0 / 1250.0f32.sqrt()).abs() < 1e-6);
        assert!((got[3] - 80.0 / 1250.0f32.sqrt()).abs() < 1e-6);
        assert!(
            (got[2] - flat[2]).abs() > 0.1,
            "headwise reduction must not match flattened q_dim RMSNorm"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_metal_matches_reference_out_of_place_and_in_place_alias() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const NUM_TOKENS: u32 = 2;
        const HIDDEN: u32 = 4;
        let eps = 1e-6f32;
        let input = [1.0f32, -2.0, 3.0, -4.0, 10.0, -20.0, 30.0, -40.0];
        let gamma = [1.0f32, 0.5, 2.0, -1.0];
        let input_f16 = input.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let gamma_f16 = gamma.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let input_region = arena.region("rmsnorm_input", input_f16.len() * half_bytes, 2)?;
        let output_region = arena.region("rmsnorm_output", input_f16.len() * half_bytes, 2)?;
        let alias_region = arena.region("rmsnorm_alias", input_f16.len() * half_bytes, 2)?;
        let gamma_region = arena.region("rmsnorm_gamma", gamma_f16.len() * half_bytes, 2)?;
        unsafe {
            write_f16_region(&arena, &input_region, &input_f16);
            write_f16_region(&arena, &output_region, &vec![f16::NAN; input_f16.len()]);
            write_f16_region(&arena, &alias_region, &input_f16);
            write_f16_region(&arena, &gamma_region, &gamma_f16);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "rmsnorm_alias_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        for (source_offset, dest_offset, op) in [
            (
                input_region.offset,
                output_region.offset,
                "rmsnorm_out_of_place",
            ),
            (alias_region.offset, alias_region.offset, "rmsnorm_in_place"),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op,
                        device: "apple-silicon",
                    },
                )
            })?;
            unsafe {
                encoder.setComputePipelineState(pipelines.get("rmsnorm_f16")?);
                encoder.setBuffer_offset_atIndex(Some(buf), source_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), dest_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), gamma_region.offset, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                    4,
                    4,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: NUM_TOKENS as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.endEncoding();
            }
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = input
            .chunks(HIDDEN as usize)
            .flat_map(|chunk| rmsnorm_ref(chunk, &gamma, HIDDEN, eps))
            .collect::<Vec<_>>();
        let out_of_place = unsafe { read_f16_region(&arena, &output_region, input.len()) };
        let in_place = unsafe { read_f16_region(&arena, &alias_region, input.len()) };
        for (name, got) in [("out_of_place", out_of_place), ("in_place", in_place)] {
            for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 0.003,
                    "{name} idx={idx} got={got} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gemm_beta_zero_ignores_nan_c_for_general_and_tiled_macos() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const M: u32 = 2;
        const N: u32 = 16;
        const K: u32 = 4;
        let a = (0..(M * K) as usize)
            .map(|idx| (idx as f32 - 3.0) * 0.25)
            .collect::<Vec<_>>();
        let b = (0..(N * K) as usize)
            .map(|idx| (idx as f32 % 9.0 - 4.0) * 0.125)
            .collect::<Vec<_>>();
        let a_f16 = a.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let b_f16 = b.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let nan_c = vec![f16::NAN; (M * N) as usize];
        let half_bytes = std::mem::size_of::<f16>();
        let a_region = arena.region("gemm_beta0_a", a_f16.len() * half_bytes, 2)?;
        let b_region = arena.region("gemm_beta0_b", b_f16.len() * half_bytes, 2)?;
        let general_region = arena.region("gemm_beta0_general", nan_c.len() * half_bytes, 2)?;
        let tiled_region = arena.region("gemm_beta0_tiled", nan_c.len() * half_bytes, 2)?;

        unsafe {
            write_f16_region(&arena, &a_region, &a_f16);
            write_f16_region(&arena, &b_region, &b_f16);
            write_f16_region(&arena, &general_region, &nan_c);
            write_f16_region(&arena, &tiled_region, &nan_c);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "gemm_beta0_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        for (kernel, output_offset, tile, op) in [
            (
                "gemm_f16",
                general_region.offset,
                8usize,
                "gemm_beta0_general",
            ),
            (
                "gemm_f16_tiled16",
                tiled_region.offset,
                16usize,
                "gemm_beta0_tiled",
            ),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op,
                        device: "apple-silicon",
                    },
                )
            })?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            unsafe {
                encoder.setComputePipelineState(pipelines.get(kernel)?);
                encoder.setBuffer_offset_atIndex(Some(buf), a_region.offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), b_region.offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&M as *const _ as *mut _),
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&N as *const _ as *mut _),
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&K as *const _ as *mut _),
                    4,
                    5,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&alpha as *const _ as *mut _),
                    4,
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&beta as *const _ as *mut _),
                    4,
                    7,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (M as usize).div_ceil(tile),
                        height: (N as usize).div_ceil(tile),
                        depth: 1,
                    },
                    MTLSize {
                        width: tile,
                        height: tile,
                        depth: 1,
                    },
                );
                encoder.endEncoding();
            }
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = gemm_ref_f32(&a, &b, M as usize, N as usize, K as usize);
        for (name, region) in [
            ("gemm_f16", &general_region),
            ("gemm_f16_tiled16", &tiled_region),
        ] {
            let got = unsafe { read_f16_region(&arena, region, expected.len()) };
            for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(got.is_finite(), "{name} output {idx} propagated NaN from C");
                assert!(
                    (got - expected).abs() < 0.02,
                    "{name} output {idx}: got={got} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn headwise_qkv_projection_norm_matches_cpu_for_multihead_fixture() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(64 * 1024)?;
        const M: u32 = 2;
        const K: u32 = 5;
        const HEAD_DIM: u32 = 4;
        const Q_HEADS: u32 = 3;
        const KV_HEADS: u32 = 2;
        const Q_DIM: u32 = Q_HEADS * HEAD_DIM;
        const KV_DIM: u32 = KV_HEADS * HEAD_DIM;
        const QKV_ROWS: u32 = Q_DIM + 2 * KV_DIM;
        let eps = 1e-6f32;

        let a = (0..(M * K) as usize)
            .map(|idx| ((idx as f32 % 7.0) - 3.0) * 0.25)
            .collect::<Vec<_>>();
        let b = (0..(QKV_ROWS * K) as usize)
            .map(|idx| ((idx as f32 % 11.0) - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let q_gamma = [0.75f32, 1.0, 1.25, 1.5];
        let k_gamma = [1.5f32, 1.25, 1.0, 0.75];
        let a_f16 = a.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let b_f16 = b.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let q_gamma_f16 = q_gamma
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let k_gamma_f16 = k_gamma
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let a_region = arena.region("qkv_headwise_a", a_f16.len() * half_bytes, 2)?;
        let b_region = arena.region("qkv_headwise_b", b_f16.len() * half_bytes, 2)?;
        let q_gamma_region =
            arena.region("qkv_headwise_q_gamma", q_gamma_f16.len() * half_bytes, 2)?;
        let k_gamma_region =
            arena.region("qkv_headwise_k_gamma", k_gamma_f16.len() * half_bytes, 2)?;
        let q_region = arena.region("qkv_headwise_q", (M * Q_DIM) as usize * half_bytes, 2)?;
        let k_region = arena.region("qkv_headwise_k", (M * KV_DIM) as usize * half_bytes, 2)?;
        let v_region =
            arena.region("qkv_headwise_v_unit", (M * KV_DIM) as usize * half_bytes, 2)?;
        unsafe {
            write_f16_region(&arena, &a_region, &a_f16);
            write_f16_region(&arena, &b_region, &b_f16);
            write_f16_region(&arena, &q_gamma_region, &q_gamma_f16);
            write_f16_region(&arena, &k_gamma_region, &k_gamma_f16);
            write_f16_region(&arena, &q_region, &vec![f16::NAN; (M * Q_DIM) as usize]);
            write_f16_region(&arena, &k_region, &vec![f16::NAN; (M * KV_DIM) as usize]);
            write_f16_region(&arena, &v_region, &vec![f16::NAN; (M * KV_DIM) as usize]);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "qkv_headwise_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        for (kernel, gamma_offset, output_offset, heads, b_row_offset, op) in [
            (
                "gemm_headwise_rmsnorm_f16",
                Some(q_gamma_region.offset),
                q_region.offset,
                Q_HEADS,
                0u32,
                "q_headwise_projection_norm",
            ),
            (
                "gemm_headwise_rmsnorm_f16",
                Some(k_gamma_region.offset),
                k_region.offset,
                KV_HEADS,
                Q_DIM,
                "k_headwise_projection_norm",
            ),
            (
                "gemm_headwise_rmsnorm_unit_f16",
                None,
                v_region.offset,
                KV_HEADS,
                Q_DIM + KV_DIM,
                "v_headwise_projection_unit_norm",
            ),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op,
                        device: "apple-silicon",
                    },
                )
            })?;
            unsafe {
                encoder.setComputePipelineState(pipelines.get(kernel)?);
                encoder.setBuffer_offset_atIndex(Some(buf), a_region.offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), b_region.offset, 1);
                let mut index = 2;
                if let Some(gamma_offset) = gamma_offset {
                    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, index);
                    index += 1;
                }
                encoder.setBuffer_offset_atIndex(Some(buf), output_offset, index);
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&M as *const _ as *mut _),
                    4,
                    index,
                );
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&K as *const _ as *mut _),
                    4,
                    index,
                );
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HEAD_DIM as *const _ as *mut _),
                    4,
                    index,
                );
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&heads as *const _ as *mut _),
                    4,
                    index,
                );
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&b_row_offset as *const _ as *mut _),
                    4,
                    index,
                );
                index += 1;
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                    4,
                    index,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (M * heads) as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.endEncoding();
            }
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let q_expected = gemm_headwise_rmsnorm_ref(
            &a,
            &b,
            Some(&q_gamma),
            M as usize,
            K as usize,
            HEAD_DIM as usize,
            Q_HEADS as usize,
            0,
            eps,
        );
        let k_expected = gemm_headwise_rmsnorm_ref(
            &a,
            &b,
            Some(&k_gamma),
            M as usize,
            K as usize,
            HEAD_DIM as usize,
            KV_HEADS as usize,
            Q_DIM as usize,
            eps,
        );
        let v_expected = gemm_headwise_rmsnorm_ref(
            &a,
            &b,
            None,
            M as usize,
            K as usize,
            HEAD_DIM as usize,
            KV_HEADS as usize,
            (Q_DIM + KV_DIM) as usize,
            eps,
        );
        for (name, region, expected) in [
            ("q", &q_region, q_expected),
            ("k", &k_region, k_expected),
            ("v_unit", &v_region, v_expected),
        ] {
            let got = unsafe { read_f16_region(&arena, region, expected.len()) };
            for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(got.is_finite(), "{name} output {idx} is non-finite");
                assert!(
                    (got - expected).abs() < 0.004,
                    "{name} output {idx}: got={got} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_projection_rmsnorm_handles_large_accumulation_without_nan() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(64 * 1024)?;
        const M: u32 = 2;
        const N: u32 = 8;
        const K: u32 = 8;
        let eps = 1e-6f32;
        let a = (0..(M * K) as usize)
            .map(|idx| if idx % 2 == 0 { 64.0 } else { -48.0 })
            .collect::<Vec<_>>();
        let b = (0..(N * K) as usize)
            .map(|idx| {
                let sign = if idx % 3 == 0 { -1.0 } else { 1.0 };
                sign * (32.0 + (idx % 5) as f32)
            })
            .collect::<Vec<_>>();
        let gamma = (0..N as usize)
            .map(|idx| 0.75 + idx as f32 * 0.05)
            .collect::<Vec<_>>();
        let a_f16 = a.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let b_f16 = b.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let gamma_f16 = gamma.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let a_region = arena.region("large_gemm_rmsnorm_a", a_f16.len() * half_bytes, 2)?;
        let b_region = arena.region("large_gemm_rmsnorm_b", b_f16.len() * half_bytes, 2)?;
        let gamma_region =
            arena.region("large_gemm_rmsnorm_gamma", gamma_f16.len() * half_bytes, 2)?;
        let c_region = arena.region("large_gemm_rmsnorm_c", (M * N) as usize * half_bytes, 2)?;
        unsafe {
            write_f16_region(&arena, &a_region, &a_f16);
            write_f16_region(&arena, &b_region, &b_f16);
            write_f16_region(&arena, &gamma_region, &gamma_f16);
            write_f16_region(&arena, &c_region, &vec![f16::NAN; (M * N) as usize]);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "large_gemm_rmsnorm_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "large_gemm_rmsnorm_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encoder.setComputePipelineState(pipelines.get("gemm_rmsnorm_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), a_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), b_region.offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), gamma_region.offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), c_region.offset, 3);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&M as *const _ as *mut _),
                4,
                4,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&N as *const _ as *mut _),
                4,
                5,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&K as *const _ as *mut _),
                4,
                6,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                4,
                7,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: M as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = gemm_rmsnorm_ref(&a, &b, &gamma, M as usize, N as usize, K as usize, eps);
        let got = unsafe { read_f16_region(&arena, &c_region, expected.len()) };
        for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                got.is_finite(),
                "fused projection RMSNorm output {idx} is non-finite"
            );
            assert!(
                (got - expected).abs() < 0.004,
                "fused projection RMSNorm output {idx}: got={got} expected={expected}"
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn residual_add_rmsnorm_matches_sequential_half_rounded_reference() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const NUM_TOKENS: u32 = 2;
        const HIDDEN: u32 = 8;
        let eps = 1e-5f32;
        let residual = (0..(NUM_TOKENS * HIDDEN) as usize)
            .map(|idx| (idx as f32 - 5.0) * 0.25)
            .collect::<Vec<_>>();
        let addition = (0..(NUM_TOKENS * HIDDEN) as usize)
            .map(|idx| {
                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                sign * (0.5 + (idx % 3) as f32 * 0.125)
            })
            .collect::<Vec<_>>();
        let gamma = (0..HIDDEN as usize)
            .map(|idx| 0.75 + idx as f32 * 0.0625)
            .collect::<Vec<_>>();

        let mut expected_residual = vec![0.0f32; residual.len()];
        let mut expected_normed = vec![0.0f32; residual.len()];
        for token in 0..NUM_TOKENS as usize {
            let base = token * HIDDEN as usize;
            let mut sum = 0.0f32;
            for dim in 0..HIDDEN as usize {
                let updated = f16::from_f32(residual[base + dim] + addition[base + dim]).to_f32();
                expected_residual[base + dim] = updated;
                sum += updated * updated;
            }
            let rms = (sum / HIDDEN as f32 + eps).sqrt();
            for dim in 0..HIDDEN as usize {
                expected_normed[base + dim] =
                    f16::from_f32(expected_residual[base + dim] / rms * gamma[dim]).to_f32();
            }
        }

        let residual_f16 = residual
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let addition_f16 = addition
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let gamma_f16 = gamma.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let residual_region = arena.region(
            "residual_add_rmsnorm_residual",
            residual_f16.len() * half_bytes,
            2,
        )?;
        let addition_region = arena.region(
            "residual_add_rmsnorm_addition",
            addition_f16.len() * half_bytes,
            2,
        )?;
        let gamma_region = arena.region(
            "residual_add_rmsnorm_gamma",
            gamma_f16.len() * half_bytes,
            2,
        )?;
        let output_region = arena.region(
            "residual_add_rmsnorm_output",
            residual_f16.len() * half_bytes,
            2,
        )?;
        unsafe {
            write_f16_region(&arena, &residual_region, &residual_f16);
            write_f16_region(&arena, &addition_region, &addition_f16);
            write_f16_region(&arena, &gamma_region, &gamma_f16);
            write_f16_region(&arena, &output_region, &vec![f16::NAN; residual_f16.len()]);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "residual_add_rmsnorm_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "residual_add_rmsnorm_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encoder.setComputePipelineState(pipelines.get("residual_add_rmsnorm_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), residual_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), addition_region.offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), output_region.offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), gamma_region.offset, 3);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                4,
                4,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                4,
                5,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: NUM_TOKENS as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got_residual = unsafe { read_f16_region(&arena, &residual_region, residual.len()) };
        let got_normed = unsafe { read_f16_region(&arena, &output_region, residual.len()) };
        for idx in 0..residual.len() {
            assert!(
                (got_residual[idx] - expected_residual[idx]).abs() < 0.0001,
                "residual[{idx}]: got={} expected={}",
                got_residual[idx],
                expected_residual[idx]
            );
            assert!(
                (got_normed[idx] - expected_normed[idx]).abs() < 0.0001,
                "normed[{idx}]: got={} expected={}",
                got_normed[idx],
                expected_normed[idx]
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_partial_metal_uses_head_dim_split_for_partial_global_rope() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        let mut q = (0..8).map(|v| f16::from_f32(v as f32)).collect::<Vec<_>>();
        let mut k = (10..18)
            .map(|v| f16::from_f32(v as f32))
            .collect::<Vec<_>>();
        let cos = [0.0f32];
        let sin = [1.0f32];
        let positions = [0_i32];
        let half_bytes = std::mem::size_of::<f16>();
        let q_region = arena.region("rope_partial_q", q.len() * half_bytes, 2)?;
        let k_region = arena.region("rope_partial_k", k.len() * half_bytes, 2)?;
        let cos_region = arena.region("rope_partial_cos", std::mem::size_of_val(&cos), 4)?;
        let sin_region = arena.region("rope_partial_sin", std::mem::size_of_val(&sin), 4)?;
        let pos_region = arena.region("rope_partial_pos", std::mem::size_of_val(&positions), 4)?;
        unsafe {
            write_f16_region(&arena, &q_region, &q);
            write_f16_region(&arena, &k_region, &k);
            std::ptr::copy_nonoverlapping(
                cos.as_ptr(),
                arena.host_ptr(&cos_region) as *mut f32,
                cos.len(),
            );
            std::ptr::copy_nonoverlapping(
                sin.as_ptr(),
                arena.host_ptr(&sin_region) as *mut f32,
                sin.len(),
            );
            std::ptr::copy_nonoverlapping(
                positions.as_ptr(),
                arena.host_ptr(&pos_region) as *mut i32,
                positions.len(),
            );
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "rope_partial_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "rope_partial_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let num_tokens = 1u32;
        let num_heads = 1u32;
        let num_kv_heads = 1u32;
        let head_dim = 8u32;
        let rope_dim = 2u32;
        unsafe {
            encoder.setComputePipelineState(pipelines.get("rope_partial_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), q_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), k_region.offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), cos_region.offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), sin_region.offset, 3);
            encoder.setBuffer_offset_atIndex(Some(buf), pos_region.offset, 4);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
                4,
                5,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&num_heads as *const _ as *mut _),
                4,
                6,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&num_kv_heads as *const _ as *mut _),
                4,
                7,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
                4,
                8,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&rope_dim as *const _ as *mut _),
                4,
                9,
            );
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        q = unsafe { read_f16_region(&arena, &q_region, q.len()) }
            .into_iter()
            .map(f16::from_f32)
            .collect();
        k = unsafe { read_f16_region(&arena, &k_region, k.len()) }
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(q[0].to_f32(), -4.0);
        assert_eq!(q[4].to_f32(), 0.0);
        assert_eq!(q[1].to_f32(), 1.0);
        assert_eq!(k[0].to_f32(), -14.0);
        assert_eq!(k[4].to_f32(), 10.0);
        assert_eq!(k[1].to_f32(), 11.0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rmsnorm_headwise_metal_uses_head_dim_gamma_per_head_and_aliases() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const NUM_TOKENS: u32 = 1;
        const NUM_HEADS: u32 = 2;
        const HEAD_DIM: u32 = 2;
        let eps = 1e-6f32;
        let input = [3.0f32, 4.0, 30.0, 40.0];
        let gamma = [1.0f32, 2.0];
        let input_f16 = input.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let gamma_f16 = gamma.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let input_region =
            arena.region("headwise_rmsnorm_input", input_f16.len() * half_bytes, 2)?;
        let output_region =
            arena.region("headwise_rmsnorm_output", input_f16.len() * half_bytes, 2)?;
        let alias_region =
            arena.region("headwise_rmsnorm_alias", input_f16.len() * half_bytes, 2)?;
        let gamma_region =
            arena.region("headwise_rmsnorm_gamma", gamma_f16.len() * half_bytes, 2)?;
        unsafe {
            write_f16_region(&arena, &input_region, &input_f16);
            write_f16_region(&arena, &output_region, &vec![f16::NAN; input_f16.len()]);
            write_f16_region(&arena, &alias_region, &input_f16);
            write_f16_region(&arena, &gamma_region, &gamma_f16);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "headwise_rmsnorm_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        for (source_offset, dest_offset, op) in [
            (
                input_region.offset,
                output_region.offset,
                "headwise_rmsnorm_out_of_place",
            ),
            (
                alias_region.offset,
                alias_region.offset,
                "headwise_rmsnorm_in_place",
            ),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op,
                        device: "apple-silicon",
                    },
                )
            })?;
            unsafe {
                encoder.setComputePipelineState(pipelines.get("rmsnorm_headwise_f16")?);
                encoder.setBuffer_offset_atIndex(Some(buf), source_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), dest_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), gamma_region.offset, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HEAD_DIM as *const _ as *mut _),
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&NUM_HEADS as *const _ as *mut _),
                    4,
                    5,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (NUM_TOKENS * NUM_HEADS) as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.endEncoding();
            }
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = headwise_rmsnorm_ref(
            &input,
            &gamma,
            NUM_TOKENS as usize,
            NUM_HEADS as usize,
            HEAD_DIM as usize,
            eps,
        );
        let out_of_place = unsafe { read_f16_region(&arena, &output_region, input.len()) };
        let in_place = unsafe { read_f16_region(&arena, &alias_region, input.len()) };
        for (name, got) in [("out_of_place", out_of_place), ("in_place", in_place)] {
            for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 0.003,
                    "{name} idx={idx} got={got} expected={expected}"
                );
            }
        }
        Ok(())
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

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn softcap_argmax_macos_smoke_matches_separate_softcap_then_argmax() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const NUM_SEQS: u32 = 2;
        const VOCAB: u32 = 6;
        const CAP: f32 = 3.0;
        let logits = vec![
            f16::from_f32(-9.0),
            f16::from_f32(0.5),
            f16::from_f32(4.0),
            f16::from_f32(1.25),
            f16::from_f32(3.5),
            f16::from_f32(-0.25),
            f16::from_f32(2.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.0),
            f16::from_f32(6.0),
            f16::from_f32(5.5),
            f16::from_f32(-7.0),
        ];
        let expected_logits = logits
            .iter()
            .map(|value| f16::from_f32(CAP * (value.to_f32() / CAP).tanh()))
            .collect::<Vec<_>>();
        let expected_tokens = argmax_ref(&expected_logits, NUM_SEQS, VOCAB);

        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let logits_region = arena.region("softcap_argmax_logits", logits.len() * half_bytes, 2)?;
        let output_region =
            arena.region("softcap_argmax_output", NUM_SEQS as usize * i32_bytes, 4)?;
        unsafe {
            write_f16_region(&arena, &logits_region, &logits);
        }

        let queue = ctx.queue_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "softcap_argmax_smoke_cmdbuf",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "softcap_argmax_smoke_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let buf = arena.buffer_retained();
        unsafe {
            encoder.setComputePipelineState(pipelines.get("softcap_argmax_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), logits_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), output_region.offset, 1);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_SEQS as *const _ as *mut _),
                4,
                2,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&VOCAB as *const _ as *mut _),
                4,
                3,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&CAP as *const _ as *mut _),
                4,
                4,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: NUM_SEQS as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got_logits = unsafe { read_f16_region(&arena, &logits_region, logits.len()) };
        for (idx, (got, expected)) in got_logits.iter().zip(expected_logits.iter()).enumerate() {
            assert!(
                (got - expected.to_f32()).abs() < 0.0001,
                "logit {idx}: got {got}, expected {}",
                expected.to_f32()
            );
        }
        let got_tokens = unsafe {
            std::slice::from_raw_parts(
                arena.host_ptr(&output_region) as *const i32,
                NUM_SEQS as usize,
            )
            .to_vec()
        };
        assert_eq!(got_tokens, expected_tokens);
        Ok(())
    }

    #[test]
    fn experimental_kv_int8_cpu_reference_roundtrips_small_vectors() {
        let src = [
            -0.50, -0.25, 0.0, 0.125, 0.25, 0.50, 0.75, -0.75, 0.0, 0.0, 0.0, 0.0,
        ]
        .into_iter()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
        let quantized = experimental_quantize_kv_f16_to_int8_reference(&src, 4);
        let got = experimental_dequantize_kv_int8_to_f16_reference(&quantized);

        assert_eq!(quantized.scales.len(), 3);
        assert_eq!(quantized.scales[2], 1.0);
        for row in 0..quantized.scales.len() {
            let tolerance = quantized.scales[row] * 0.55 + 0.001;
            for dim in 0..quantized.kv_dim {
                let idx = row * quantized.kv_dim + dim;
                assert!(
                    (got[idx].to_f32() - src[idx].to_f32()).abs() <= tolerance,
                    "idx={idx} src={} got={} tolerance={tolerance}",
                    src[idx].to_f32(),
                    got[idx].to_f32(),
                );
            }
        }
    }

    #[test]
    fn experimental_kv_int8_cpu_attention_decode_stays_close_to_f16_reference() {
        const NUM_SEQS: u32 = 1;
        const NUM_HEADS: u32 = 2;
        const NUM_KV_HEADS: u32 = 1;
        const HEAD_DIM: u32 = 4;
        const BLOCK_SIZE: u32 = 4;
        const MAX_BLOCKS: u32 = 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let kv_dim = (NUM_KV_HEADS * HEAD_DIM) as usize;
        let q = (0..(NUM_SEQS * NUM_HEADS * HEAD_DIM) as usize)
            .map(|i| f16::from_f32((i as f32 - 3.0) * 0.04))
            .collect::<Vec<_>>();
        let k_cache = (0..(BLOCK_SIZE as usize * kv_dim))
            .map(|i| f16::from_f32((i as f32 - 5.0) * 0.03))
            .collect::<Vec<_>>();
        let v_cache = (0..(BLOCK_SIZE as usize * kv_dim))
            .map(|i| f16::from_f32((i as f32 + 1.0) * 0.02))
            .collect::<Vec<_>>();
        let block_tables = [0_i32];
        let context_lens = [4_i32];

        let f16_out = attention_decode_ref(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            NUM_SEQS,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            MAX_BLOCKS,
            scale,
        );
        let k_int8 = experimental_quantize_kv_f16_to_int8_reference(&k_cache, kv_dim);
        let v_int8 = experimental_quantize_kv_f16_to_int8_reference(&v_cache, kv_dim);
        let k_deq = experimental_dequantize_kv_int8_to_f16_reference(&k_int8);
        let v_deq = experimental_dequantize_kv_int8_to_f16_reference(&v_int8);
        let int8_out = attention_decode_ref(
            &q,
            &k_deq,
            &v_deq,
            &block_tables,
            &context_lens,
            NUM_SEQS,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            MAX_BLOCKS,
            scale,
        );

        for (idx, (baseline, compressed)) in f16_out.iter().zip(int8_out.iter()).enumerate() {
            assert!(
                (baseline - compressed).abs() < 0.006,
                "idx={idx} baseline={baseline} compressed={compressed}",
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Apple Silicon Metal device; experimental compressed KV path"]
    fn experimental_kv_int8_metal_quantize_dequantize_smoke_matches_cpu_reference(
    ) -> rvllm_core::Result<()> {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(crate::kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let mut arena = MetalBufferArena::new(ctx.device(), 16 * 1024)?;

        const NUM_ROWS: u32 = 3;
        const KV_DIM: u32 = 8;
        let src = (0..(NUM_ROWS * KV_DIM) as usize)
            .map(|i| f16::from_f32((i as f32 - 7.0) * 0.03125))
            .collect::<Vec<_>>();
        let cpu_quant = experimental_quantize_kv_f16_to_int8_reference(&src, KV_DIM as usize);
        let cpu_deq = experimental_dequantize_kv_int8_to_f16_reference(&cpu_quant);

        let half_bytes = std::mem::size_of::<f16>();
        let src_region = arena.region("experimental_kv_int8_src", src.len() * half_bytes, 2)?;
        let q_region = arena.region("experimental_kv_int8_q", src.len(), 1)?;
        let scales_region = arena.region(
            "experimental_kv_int8_scales",
            NUM_ROWS as usize * std::mem::size_of::<f32>(),
            4,
        )?;
        let deq_region = arena.region("experimental_kv_int8_deq", src.len() * half_bytes, 2)?;

        unsafe {
            let src_ptr = arena.host_ptr(&src_region) as *mut f16;
            for (idx, value) in src.iter().enumerate() {
                *src_ptr.add(idx) = *value;
            }
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "experimental_kv_int8_smoke",
                    device: "apple-silicon",
                },
            )
        })?;

        let quant = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "experimental_kv_int8_quant_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            quant.setComputePipelineState(pipelines.get("experimental_kv_quantize_int8_f16")?);
            quant.setBuffer_offset_atIndex(Some(buf), src_region.offset, 0);
            quant.setBuffer_offset_atIndex(Some(buf), q_region.offset, 1);
            quant.setBuffer_offset_atIndex(Some(buf), scales_region.offset, 2);
            quant.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_ROWS as *const _ as *mut _),
                4,
                3,
            );
            quant.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&KV_DIM as *const _ as *mut _),
                4,
                4,
            );
            quant.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: NUM_ROWS as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            quant.endEncoding();
        }

        let dequant = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "experimental_kv_int8_dequant_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            dequant.setComputePipelineState(pipelines.get("experimental_kv_dequantize_int8_f16")?);
            dequant.setBuffer_offset_atIndex(Some(buf), q_region.offset, 0);
            dequant.setBuffer_offset_atIndex(Some(buf), scales_region.offset, 1);
            dequant.setBuffer_offset_atIndex(Some(buf), deq_region.offset, 2);
            dequant.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_ROWS as *const _ as *mut _),
                4,
                3,
            );
            dequant.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&KV_DIM as *const _ as *mut _),
                4,
                4,
            );
            dequant.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: src.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            dequant.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got_q = unsafe {
            std::slice::from_raw_parts(arena.host_ptr(&q_region) as *const i8, src.len())
        };
        let got_deq = unsafe {
            std::slice::from_raw_parts(arena.host_ptr(&deq_region) as *const f16, src.len())
        };
        assert_eq!(got_q, cpu_quant.values.as_slice());
        for (idx, (got, expected)) in got_deq.iter().zip(cpu_deq.iter()).enumerate() {
            assert!(
                (got.to_f32() - expected.to_f32()).abs() < 0.001,
                "idx={idx} got={} expected={}",
                got.to_f32(),
                expected.to_f32(),
            );
        }
        Ok(())
    }

    #[test]
    fn kernel_count_matches_names() {
        assert_eq!(KERNEL_COUNT, KERNEL_NAMES.len());
    }
}
