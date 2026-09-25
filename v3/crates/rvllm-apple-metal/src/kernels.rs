//! Metal Shading Language (MSL) kernel sources.
//!
//! All compute kernels are embedded as string constants and compiled
//! at runtime. This avoids build-time metallib compilation and makes
//! the crate work without Xcode command-line tools.

use std::borrow::Cow;

use crate::MetalFloatType;

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
    uint base = token * num_heads * head_dim + head * head_dim;
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
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

// ============================================================================
// Group-32 low-bit projection (W4A16 / W8A16)
// ============================================================================
// One simdgroup computes one output element. Weights are row-major [N,K]
// because each output channel is one independently-scaled row. Quantization
// groups never cross a row boundary and every group of at most 32 K-elements
// has one native FP16 scale. W4 stores the earlier K-element in the low
// nibble; both formats reserve their asymmetric minimum on the host.

kernel void projection_w4a16_f16(
    device const half  *A         [[buffer(0)]],  // [M,K]
    device const uchar *W         [[buffer(1)]],  // [N,ceil(K/2)]
    device const half  *scales    [[buffer(2)]],  // [N,ceil(K/32)]
    device half        *C         [[buffer(3)]],  // [M,N]
    constant uint      &M         [[buffer(4)]],
    constant uint      &N         [[buffer(5)]],
    constant uint      &K         [[buffer(6)]],
    constant uint      &C_stride  [[buffer(7)]],
    constant uint      &C_column  [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;

    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        uchar packed = W[n * packed_row_bytes + (k >> 1u)];
        int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
        q = q >= 8 ? q - 16 : q;
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[m * C_stride + C_column + n] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

kernel void projection_w8a16_f16(
    device const half  *A         [[buffer(0)]],  // [M,K]
    device const char  *W         [[buffer(1)]],  // [N,K]
    device const half  *scales    [[buffer(2)]],  // [N,ceil(K/32)]
    device half        *C         [[buffer(3)]],  // [M,N]
    constant uint      &M         [[buffer(4)]],
    constant uint      &N         [[buffer(5)]],
    constant uint      &K         [[buffer(6)]],
    constant uint      &C_stride  [[buffer(7)]],
    constant uint      &C_column  [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;

    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        int q = int(W[n * K + k]);
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[m * C_stride + C_column + n] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

// Experimental native-BF16 activation/output ABI.  The quantizer's group-32
// scales intentionally remain FP16: they are package metadata, not activation
// tensors, and retaining their established two-byte encoding permits an
// isolated A-dtype experiment without silently changing quantized weights.
// Accumulation and the SIMD reduction remain FP32; only the projection input
// and its single storage-rounding output boundary are BF16.
kernel void experimental_projection_w4abf16_bf16(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        uchar packed = W[n * packed_row_bytes + (k >> 1u)];
        int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
        q = q >= 8 ? q - 16 : q;
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) C[m * C_stride + C_column + n] = bfloat(total);
}

kernel void experimental_projection_w8abf16_bf16(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        int q = int(W[n * K + k]);
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) C[m * C_stride + C_column + n] = bfloat(total);
}

// Four-output native-BF16 research variants. One SIMD group owns four
// adjacent output channels and reuses every activation load across all four
// dot products. Output tails remain explicit; no padded weight row is read.
kernel void experimental_projection_w4abf16_bf16_n4(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 4u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * packed_row_bytes + (k >> 1u)];
                int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
                q = q >= 8 ? q - 16 : q;
                float scale = float(scales[n * groups_per_row + (k >> 5u)]);
                partial[column] += activation * (float(q) * scale);
            }
        }
    }
    float4 total = float4(
        simd_sum(partial.x), simd_sum(partial.y),
        simd_sum(partial.z), simd_sum(partial.w));
    if (lane == 0) {
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_n4(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 4u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                int q = int(W[n * K + k]);
                float scale = float(scales[n * groups_per_row + (k >> 5u)]);
                partial[column] += activation * (float(q) * scale);
            }
        }
    }
    float4 total = float4(
        simd_sum(partial.x), simd_sum(partial.y),
        simd_sum(partial.z), simd_sum(partial.w));
    if (lane == 0) {
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
        }
    }
}

// Eight-output variants trade twice the accumulator footprint for additional
// activation reuse. They are a separate research schedule, not an alias or an
// automatic replacement for N4.
kernel void experimental_projection_w4abf16_bf16_n8(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 8u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * packed_row_bytes + (k >> 1u)];
                int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
                q = q >= 8 ? q - 16 : q;
                float term = activation * (float(q) * float(scales[n * groups_per_row + (k >> 5u)]));
                if (column < 4u) partial_lo[column] += term;
                else partial_hi[column - 4u] += term;
            }
        }
    }
    float4 total_lo = float4(
        simd_sum(partial_lo.x), simd_sum(partial_lo.y),
        simd_sum(partial_lo.z), simd_sum(partial_lo.w));
    float4 total_hi = float4(
        simd_sum(partial_hi.x), simd_sum(partial_hi.y),
        simd_sum(partial_hi.z), simd_sum(partial_hi.w));
    if (lane == 0) {
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float value = column < 4u ? total_lo[column] : total_hi[column - 4u];
                C[m * C_stride + C_column + n] = bfloat(value);
            }
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_n8(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 8u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float term = activation * (float(W[n * K + k]) * float(scales[n * groups_per_row + (k >> 5u)]));
                if (column < 4u) partial_lo[column] += term;
                else partial_hi[column - 4u] += term;
            }
        }
    }
    float4 total_lo = float4(
        simd_sum(partial_lo.x), simd_sum(partial_lo.y),
        simd_sum(partial_lo.z), simd_sum(partial_lo.w));
    float4 total_hi = float4(
        simd_sum(partial_hi.x), simd_sum(partial_hi.y),
        simd_sum(partial_hi.z), simd_sum(partial_hi.w));
    if (lane == 0) {
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float value = column < 4u ? total_lo[column] : total_hi[column - 4u];
                C[m * C_stride + C_column + n] = bfloat(value);
            }
        }
    }
}

// W4 next-round schedule: each lane consumes both values in one packed byte.
kernel void experimental_projection_w4abf16_bf16_n4_packed2(
    device const bfloat *A [[buffer(0)]], device const uchar *W [[buffer(1)]],
    device const half *scales [[buffer(2)]], device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]], constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 output [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint n0 = output.x * 4u, m = output.y;
    if (m >= M || n0 >= N) return;
    uint row_bytes = (K + 1u) >> 1u, groups = (K + 31u) >> 5u;
    float4 even = 0.0f, odd = 0.0f;
    for (uint k0 = uint(lane) * 2u; k0 < K; k0 += 64u) {
        float a0 = float(A[m * K + k0]);
        bool has_k1 = k0 + 1u < K;
        float a1 = has_k1 ? float(A[m * K + k0 + 1u]) : 0.0f;
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * row_bytes + (k0 >> 1u)];
                int q0 = int(packed & 0x0fu); q0 = q0 >= 8 ? q0 - 16 : q0;
                int q1 = int(packed >> 4u); q1 = q1 >= 8 ? q1 - 16 : q1;
                float scale = float(scales[n * groups + (k0 >> 5u)]);
                even[column] += a0 * (float(q0) * scale);
                if (has_k1) odd[column] += a1 * (float(q1) * scale);
            }
        }
    }
    float4 total;
    for (uint column = 0u; column < 4u; ++column)
        total[column] = simd_sum(even[column]) + simd_sum(odd[column]);
    if (lane == 0) for (uint column = 0u; column < 4u; ++column) {
        uint n = n0 + column;
        if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
    }
}

// W8 next-round schedule: four adjacent K values per lane, with one activation
// vector reused across eight output rows. The scalar tail keeps arbitrary K legal.
kernel void experimental_projection_w8abf16_bf16_n8_k4(
    device const bfloat *A [[buffer(0)]], device const char *W [[buffer(1)]],
    device const half *scales [[buffer(2)]], device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]], constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 output [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint n0 = output.x * 8u, m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups = (K + 31u) >> 5u;
    float4 lo = 0.0f, hi = 0.0f;
    for (uint k0 = uint(lane) * 4u; k0 < K; k0 += 128u) {
        uint count = min(4u, K - k0);
        float4 av = 0.0f;
        for (uint j = 0u; j < count; ++j) av[j] = float(A[m * K + k0 + j]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float sum = 0.0f;
                for (uint j = 0u; j < count; ++j) {
                    float scale = float(scales[n * groups + ((k0 + j) >> 5u)]);
                    sum += av[j] * (float(W[n * K + k0 + j]) * scale);
                }
                if (column < 4u) lo[column] += sum; else hi[column - 4u] += sum;
            }
        }
    }
    float4 total_lo, total_hi;
    for (uint column = 0u; column < 4u; ++column) {
        total_lo[column] = simd_sum(lo[column]);
        total_hi[column] = simd_sum(hi[column]);
    }
    if (lane == 0) for (uint column = 0u; column < 8u; ++column) {
        uint n = n0 + column;
        if (n < N) C[m * C_stride + C_column + n] = bfloat(column < 4u ? total_lo[column] : total_hi[column - 4u]);
    }
}

constant uint TILE_M = 8;
constant uint TILE_N = 8;
constant uint TILE16 = 16;
constant uint GEMV_N = 8;
constant uint GEMV_TG = 256;
constant uint GEMM_RMSNORM_MAX_N = 6144;
constant uint HEADWISE_RMSNORM_MAX_D = 512;
constant uint FINAL_ARGMAX_TILE_N = 8;

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

// Decode/prompt microbatch GEMM: C[M,N] = A[M,K] * B[N,K]^T.
//
// The general/tiled kernels map one output element to one thread. That is
// correct but wastes most lanes for M<=16 and makes every thread serially walk
// all K. This path maps one row and eight output columns to a threadgroup, then
// reduces K cooperatively across 256 lanes.
kernel void gemm_f16_vec8(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]]
) {
    uint row = gid.x;
    uint col_base = gid.y * GEMV_N;
    if (row >= M) return;

    threadgroup float partial[8][256];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    for (uint k = tid; k < K; k += GEMV_TG) {
        float a = float(A[row * K + k]);
        uint col = col_base;
        if (col < N) acc0 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc1 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc2 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc3 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc4 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc5 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc6 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc7 += a * float(B[col * K + k]);
    }

    partial[0][tid] = acc0;
    partial[1][tid] = acc1;
    partial[2][tid] = acc2;
    partial[3][tid] = acc3;
    partial[4][tid] = acc4;
    partial[5][tid] = acc5;
    partial[6][tid] = acc6;
    partial[7][tid] = acc7;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GEMV_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
            partial[5][tid] += partial[5][tid + stride];
            partial[6][tid] += partial[6][tid + stride];
            partial[7][tid] += partial[7][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < GEMV_N) {
        uint col = col_base + tid;
        if (col < N) {
            uint idx = row * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
            C[idx] = f16_sat(partial[tid][0] * alpha + prior);
        }
    }
}

// Apple9/10 exact batch-eight projection GEMM.
//
// Eight simdgroups share the weight stream across all eight request rows.
// This is the promoted production path because its generated tokens are
// qualified against the independent Gemma reference.
kernel void gemm_f16_batch8(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col = gid.y * 8u + uint(simdgroup);

    float accumulators[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    if (col < N) {
        for (uint k = uint(lane); k < K; k += 32u) {
            float weight = float(B[col * K + k]);
            for (uint row = 0u; row < 8u && row_base + row < M; row++) {
                accumulators[row] += float(A[(row_base + row) * K + k]) * weight;
            }
        }
    }
    for (uint row = 0u; row < 8u; row++) {
        accumulators[row] = simd_sum(accumulators[row]);
    }
    if (lane == 0 && col < N) {
        for (uint row = 0u; row < 8u && row_base + row < M; row++) {
            uint output_index = (row_base + row) * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
            C[output_index] = f16_sat(accumulators[row] * alpha + prior);
        }
    }
}

// QKV prefill keeps projection sums in FP32 until head normalization.
kernel void qkv_project_f32_batch8(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device float      *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col = gid.y * 8u + uint(simdgroup);

    float accumulators[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    if (col < N) {
        for (uint k = uint(lane); k < K; k += 32u) {
            float weight = float(B[col * K + k]);
            for (uint row = 0u; row < 8u && row_base + row < M; row++) {
                accumulators[row] += float(A[(row_base + row) * K + k]) * weight;
            }
        }
    }
    for (uint row = 0u; row < 8u; row++) {
        accumulators[row] = simd_sum(accumulators[row]);
    }
    if (lane == 0 && col < N) {
        for (uint row = 0u; row < 8u && row_base + row < M; row++) {
            uint output_index = (row_base + row) * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
            C[output_index] = accumulators[row] * alpha + prior;
        }
    }
}

// Experimental Apple9/10 batch-eight projection GEMM.
//
// One SIMD group computes an 8x8 output tile with the public SIMD-group matrix
// instructions. A is already row-major [8,K]. Weights are stored as [N,K], so
// each tile is staged in threadgroup memory as the required [K,N] matrix before
// the SIMD-group load. This replaces eight scalar dot-product reductions per
// output column with the GPU's native 8x8 matrix primitive.
// It is compiled and hardware-tested, but not selected for production until
// the full-model numerical parity gate passes.
kernel void gemm_f16_simdgroup8x8(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col_base = gid.y * 8u;
    if (row_base + 8u > M || col_base + 8u > N) {
        return;
    }

    simdgroup_float8x8 mat_a;
    simdgroup_float8x8 mat_b;
    simdgroup_float8x8 mat_c(0.0f);
    threadgroup float activation_tile[64];
    threadgroup float weight_tile[64];
    for (uint k = 0u; k < K; k += 8u) {
        for (uint tile_index = uint(lane); tile_index < 64u; tile_index += 32u) {
            uint tile_row = tile_index / 8u;
            uint tile_col = tile_index % 8u;
            activation_tile[tile_index] =
                float(A[(row_base + tile_row) * K + k + tile_col]);
            weight_tile[tile_index] =
                float(B[(col_base + tile_col) * K + k + tile_row]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(mat_a, activation_tile, 8);
        simdgroup_load(mat_b, weight_tile, 8);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
    }

    threadgroup float output_tile[64];
    simdgroup_store(mat_c, output_tile, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile_index = uint(lane); tile_index < 64u; tile_index += 32u) {
        uint row = row_base + tile_index / 8u;
        uint col = col_base + tile_index % 8u;
        uint output_index = row * N + col;
        float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
        C[output_index] = f16_sat(output_tile[tile_index] * alpha + prior);
    }
}

// Four SIMD groups share 32x32 BF16/FP16 tiles; products accumulate in FP32.
// The BF16 runtime path is opt-in and shape bounded. Both output ABIs use
// the same computation, preserving QKV FP32 until head normalization.
inline void prefill_mma32_tile(device const half *A, device const half *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup half *at, threadgroup half *bt, threadgroup float *ct) {
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    const uint sm = uint(sg / 2u) * 16u;
    const uint sn = uint(sg % 2u) * 16u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u;
            uint k = kb + index % 32u;
            at[index] = mr + row < M && k < K ? A[(mr + row) * K + k] : half(0.0f);
            bt[index] = nc + row < N && k < K ? B[(nc + row) * K + k] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(b0, bt + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // All four groups must finish reading before any overwrite the tiles.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sm * 32u + sn, 32);
    simdgroup_store(c01, ct + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, ct + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, ct + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


kernel void qkv_project_f32_mma32(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[32 * 32];
    threadgroup half bt[32 * 32];
    threadgroup float ct[32 * 32];
    prefill_mma32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            uint output = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = alpha * ct[index] + prior;
        }
    }
}


kernel void gemm_f16_mma32(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[32 * 32];
    threadgroup half bt[32 * 32];
    threadgroup float ct[32 * 32];
    prefill_mma32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            uint output = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = f16_sat(alpha * ct[index] + prior);
        }
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
    threadgroup float projected[6144];
    float local_sum = 0.0f;
    for (uint col = tid; col < N; col += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]);
        }
        if (N <= GEMM_RMSNORM_MAX_N) {
            projected[col] = acc;
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
        if (N <= GEMM_RMSNORM_MAX_N) {
            acc = projected[col];
        } else {
            for (uint k = 0; k < K; k++) {
                acc += float(A[row * K + k]) * float(B[col * K + k]);
            }
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
    threadgroup float projected[512];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
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
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < K; k++) {
                acc += float(A[token * K + k]) * float(B[row * K + k]);
            }
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
    threadgroup float projected[512];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
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
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < K; k++) {
                acc += float(A[token * K + k]) * float(B[row * K + k]);
            }
        }
        C[c_head_base + d] = f16_sat(acc * rms);
    }
}

kernel void qkv_headwise_rmsnorm_f16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device const half *q_gamma    [[buffer(2)]],
    device const half *k_gamma    [[buffer(3)]],
    device const half *v_gamma    [[buffer(4)]],
    device half       *Q          [[buffer(5)]],
    device half       *K_out      [[buffer(6)]],
    device half       *V_out      [[buffer(7)]],
    constant uint     &M          [[buffer(8)]],
    constant uint     &hidden_k   [[buffer(9)]],
    constant uint     &head_dim   [[buffer(10)]],
    constant uint     &num_q_heads [[buffer(11)]],
    constant uint     &num_kv_heads [[buffer(12)]],
    constant uint     &q_row_offset [[buffer(13)]],
    constant uint     &k_row_offset [[buffer(14)]],
    constant uint     &v_row_offset [[buffer(15)]],
    constant float    &eps        [[buffer(16)]],
    constant uint     &v_has_gamma [[buffer(17)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint c_head_base = token * num_q_heads * head_dim + head * head_dim;
    device half *out = Q;
    device const half *gamma = q_gamma;
    bool use_gamma = true;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            c_head_base = token * num_kv_heads * head_dim + head * head_dim;
            out = K_out;
            gamma = k_gamma;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            c_head_base = token * num_kv_heads * head_dim + head * head_dim;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < hidden_k; k++) {
            acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
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
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < hidden_k; k++) {
                acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
            }
        }
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        out[c_head_base + d] = f16_sat(acc * rms * scale);
    }
}

kernel void qkv_headwise_rmsnorm_rope_cache_f16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device const half *q_gamma    [[buffer(2)]],
    device const half *k_gamma    [[buffer(3)]],
    device const half *v_gamma    [[buffer(4)]],
    device half       *Q          [[buffer(5)]],
    device half       *K_out      [[buffer(6)]],
    device half       *V_out      [[buffer(7)]],
    device const float *cos_table [[buffer(8)]],
    device const float *sin_table [[buffer(9)]],
    device const int  *positions  [[buffer(10)]],
    device const int  *slot_map   [[buffer(11)]],
    device half       *k_cache    [[buffer(12)]],
    device half       *v_cache    [[buffer(13)]],
    constant uint     &M          [[buffer(14)]],
    constant uint     &hidden_k   [[buffer(15)]],
    constant uint     &head_dim   [[buffer(16)]],
    constant uint     &num_q_heads [[buffer(17)]],
    constant uint     &num_kv_heads [[buffer(18)]],
    constant uint     &q_row_offset [[buffer(19)]],
    constant uint     &k_row_offset [[buffer(20)]],
    constant uint     &v_row_offset [[buffer(21)]],
    constant float    &eps        [[buffer(22)]],
    constant uint     &v_has_gamma [[buffer(23)]],
    constant uint     &rope_dim   [[buffer(24)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M || head_dim > HEADWISE_RMSNORM_MAX_D) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint q_dim = num_q_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint row_head_base = head * head_dim;
    uint c_head_base = token * q_dim + row_head_base;
    device half *out = Q;
    device const half *gamma = q_gamma;
    bool use_gamma = true;
    bool is_k = false;
    bool is_v = false;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = K_out;
            gamma = k_gamma;
            is_k = true;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
            is_v = true;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    threadgroup float normalized[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < hidden_k; k++) {
            acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
        }
        projected[d] = acc;
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
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        normalized[d] = float(f16_sat(projected[d] * rms * scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = slot_map[token];
    uint half_rope = rope_dim / 2;
    int pos = positions[token];
    for (uint d = tid; d < head_dim; d += tg_size) {
        float value = normalized[d];
        if (!is_v) {
            if (d < half_rope) {
                uint pair = d;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * cos_val - x1 * sin_val;
            } else if (d >= head_dim / 2 && d < head_dim / 2 + half_rope) {
                uint pair = d - head_dim / 2;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * sin_val + x1 * cos_val;
            }
        }
        half out_value = f16_sat(value);
        out[c_head_base + d] = out_value;
        if (slot >= 0 && is_k) {
            k_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        } else if (slot >= 0 && is_v) {
            v_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        }
    }
}

// Same norm, dtype rounding, RoPE and cache stores as the fused path.
// Buffer A holds packed FP32 QKV; buffer B is unused to share the host ABI.
kernel void qkv_projected_rmsnorm_rope_cache_f16(
    device const float *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device const half *q_gamma    [[buffer(2)]],
    device const half *k_gamma    [[buffer(3)]],
    device const half *v_gamma    [[buffer(4)]],
    device half       *Q          [[buffer(5)]],
    device half       *K_out      [[buffer(6)]],
    device half       *V_out      [[buffer(7)]],
    device const float *cos_table [[buffer(8)]],
    device const float *sin_table [[buffer(9)]],
    device const int  *positions  [[buffer(10)]],
    device const int  *slot_map   [[buffer(11)]],
    device half       *k_cache    [[buffer(12)]],
    device half       *v_cache    [[buffer(13)]],
    constant uint     &M          [[buffer(14)]],
    constant uint     &hidden_k   [[buffer(15)]],
    constant uint     &head_dim   [[buffer(16)]],
    constant uint     &num_q_heads [[buffer(17)]],
    constant uint     &num_kv_heads [[buffer(18)]],
    constant uint     &q_row_offset [[buffer(19)]],
    constant uint     &k_row_offset [[buffer(20)]],
    constant uint     &v_row_offset [[buffer(21)]],
    constant float    &eps        [[buffer(22)]],
    constant uint     &v_has_gamma [[buffer(23)]],
    constant uint     &rope_dim   [[buffer(24)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M || head_dim > HEADWISE_RMSNORM_MAX_D) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint q_dim = num_q_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint row_head_base = head * head_dim;
    uint c_head_base = token * q_dim + row_head_base;
    device half *out = Q;
    device const half *gamma = q_gamma;
    bool use_gamma = true;
    bool is_k = false;
    bool is_v = false;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = K_out;
            gamma = k_gamma;
            is_k = true;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
            is_v = true;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    threadgroup float normalized[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        uint projection_width = (num_q_heads + 2u * num_kv_heads) * head_dim;
        float acc = A[token * projection_width + row];
        projected[d] = acc;
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
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        normalized[d] = float(f16_sat(projected[d] * rms * scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = slot_map[token];
    uint half_rope = rope_dim / 2;
    int pos = positions[token];
    for (uint d = tid; d < head_dim; d += tg_size) {
        float value = normalized[d];
        if (!is_v) {
            if (d < half_rope) {
                uint pair = d;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * cos_val - x1 * sin_val;
            } else if (d >= head_dim / 2 && d < head_dim / 2 + half_rope) {
                uint pair = d - head_dim / 2;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * sin_val + x1 * cos_val;
            }
        }
        half out_value = f16_sat(value);
        out[c_head_base + d] = out_value;
        if (slot >= 0 && is_k) {
            k_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        } else if (slot >= 0 && is_v) {
            v_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        }
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
// the normal checkpoint-native decode path; tests opt in explicitly and
// dequantize back to F16 before comparison. Quantization is symmetric per cache
// row.
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
    constant uint     &attention_window [[buffer(13)]], // 0 = full attention
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

    uint attn_start = attention_window == 0
        ? 0
        : uint(ctx_len) - min(uint(ctx_len), attention_window);

    // Process each KV token in this layer's exact causal extent.
    for (uint t = attn_start; t < uint(ctx_len); t++) {
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

// Single-pass decode attention for production decode shapes. One 32-lane
// simdgroup handles one (sequence, Q head). Each lane owns up to eight head
// dimensions, keeps Q and the online-softmax output accumulator in registers,
// and participates in exactly one Q.K reduction per KV token. The arbitrary
// physical block IDs in block_tables preserve request-owned/shared-prefix
// page semantics; the logical token coordinate still determines the exact
// sliding-window lower bound.
kernel void attention_decode_online_f16(
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
    constant uint     &attention_window [[buffer(13)]],
    uint gid                      [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
    uint seq = gid / num_heads;
    uint head = gid % num_heads;
    if (seq >= num_seqs || head_dim == 0 || head_dim > 256) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    int ctx_len_i = context_lens[seq];
    if (ctx_len_i <= 0) return;
    uint ctx_len = uint(ctx_len_i);
    uint attn_start = attention_window == 0
        ? 0
        : ctx_len - min(ctx_len, attention_window);

    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    float q_lane[8];
    float out_lane[8];
    for (uint slot = 0; slot < 8; slot++) {
        uint d = uint(lane) + slot * 32u;
        q_lane[slot] = d < head_dim
            ? float(q[seq * q_dim + head * head_dim + d])
            : 0.0f;
        out_lane[slot] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    for (uint t = attn_start; t < ctx_len; t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        uint block_base = uint(block_id) * block_size * kv_dim;
        uint kv_base = block_base + block_offset * kv_dim + kv_head * head_dim;
        float partial_score = 0.0f;
        for (uint slot = 0; slot < 8; slot++) {
            uint d = uint(lane) + slot * 32u;
            if (d < head_dim) {
                partial_score += q_lane[slot] * float(k_cache[kv_base + d]);
            }
        }
        float score = simd_sum(partial_score) * scale;

        float next_max = max(max_score, score);
        float correction = exp(max_score - next_max);
        float weight = exp(score - next_max);
        sum_exp = sum_exp * correction + weight;
        for (uint slot = 0; slot < 8; slot++) {
            uint d = uint(lane) + slot * 32u;
            if (d < head_dim) {
                out_lane[slot] = out_lane[slot] * correction
                    + weight * float(v_cache[kv_base + d]);
            }
        }
        max_score = next_max;
    }

    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (uint slot = 0; slot < 8; slot++) {
        uint d = uint(lane) + slot * 32u;
        if (d < head_dim) {
            output[seq * q_dim + head * head_dim + d] = f16_sat(out_lane[slot] * inv_sum);
        }
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
    device const int  *positions  [[buffer(7)]],   // absolute query positions
    constant uint     &total_q    [[buffer(8)]],
    constant uint     &batch_size [[buffer(9)]],
    constant uint     &num_heads  [[buffer(10)]],
    constant uint     &num_kv_heads [[buffer(11)]],
    constant uint     &head_dim   [[buffer(12)]],
    constant uint     &block_size [[buffer(13)]],
    constant uint     &max_blocks [[buffer(14)]],
    constant float    &scale      [[buffer(15)]],
    constant uint     &attention_window [[buffer(16)]], // 0 = full attention
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

    // Causal extent is absolute. q_offset is only chunk-local and would drop
    // an already-restored prefix during chunked/cache-assisted prefill.
    uint absolute_position = uint(max(positions[q_pos], 0));
    uint attn_len = min(uint(ctx_len), absolute_position + 1);
    uint attn_start = attention_window == 0
        ? 0
        : attn_len - min(attn_len, attention_window);

    for (uint t = attn_start; t < attn_len; t++) {
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

// One SIMD group owns a query/head; dimensions are distributed across lanes.

kernel void attention_prefill_simdgroup_f16(
    device const half *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]],
    device const int *context_lens [[buffer(5)]],
    device const int *cu_seqlens [[buffer(6)]],
    device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]],
    constant uint &batch_size [[buffer(9)]],
    constant uint &num_heads [[buffer(10)]],
    constant uint &num_kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]],
    constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]],
    constant float &scale [[buffer(15)]],
    constant uint &attention_window [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint q_pos = group.x;
    uint head = group.y;
    if (q_pos >= total_q || head >= num_heads || (head_dim != 256 && head_dim != 512)) return;
    uint seq = batch_size;
    for (uint s = 0; s < batch_size; s++) {
        if (int(q_pos) >= cu_seqlens[s] && int(q_pos) < cu_seqlens[s + 1]) { seq = s; break; }
    }
    if (seq == batch_size || context_lens[seq] <= 0) return;
    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint attn_len = min(uint(context_lens[seq]), uint(max(positions[q_pos], 0)) + 1u);
    uint attn_start = attention_window == 0 ? 0 : attn_len - min(attn_len, attention_window);
    uint slots = head_dim / 32u;
    float q_lane[16];
    float out_lane[16];
    for (uint slot = 0; slot < slots; slot++) {
        q_lane[slot] = float(q[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u]);
        out_lane[slot] = 0.0f;
    }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    for (uint t = attn_start; t < attn_len; t++) {
        int block_id = block_tables[seq * max_blocks + t / block_size];
        if (block_id < 0) continue;
        uint kv_base = uint(block_id) * block_size * kv_dim + (t % block_size) * kv_dim + kv_head * head_dim;
        float partial_score = 0.0f;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            partial_score += q_lane[slot] * float(k_cache[kv_base + d]);
        }
        float score = simd_sum(partial_score) * scale;
        float next_max = max(max_score, score);
        float correction = exp(max_score - next_max);
        float weight = exp(score - next_max);
        sum_exp = sum_exp * correction + weight;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            out_lane[slot] = out_lane[slot] * correction + weight * float(v_cache[kv_base + d]);
        }
        max_score = next_max;
    }
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (uint slot = 0; slot < slots; slot++) {
        output[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u] = f16_sat(out_lane[slot] * inv_sum);
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

constant uint MOE_MAX_EXPERTS = 256;
constant uint MOE_MAX_TOP_K = 16;

kernel void moe_router_topk_f16(
    device const half *hidden                 [[buffer(0)]],
    device const float *router_proj           [[buffer(1)]],
    device const float *router_scale          [[buffer(2)]],
    device const float *router_per_expert_scale [[buffer(3)]],
    device int        *topk_indices           [[buffer(4)]],
    device float      *topk_weights           [[buffer(5)]],
    constant uint     &num_tokens             [[buffer(6)]],
    constant uint     &hidden_size            [[buffer(7)]],
    constant uint     &num_experts            [[buffer(8)]],
    constant uint     &top_k                  [[buffer(9)]],
    constant float    &eps                    [[buffer(10)]],
    constant float    &scalar_root_size       [[buffer(11)]],
    uint tid                                  [[thread_index_in_threadgroup]],
    uint tg_size                              [[threads_per_threadgroup]],
    uint token                                [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sum[256];
    threadgroup float logits[MOE_MAX_EXPERTS];
    threadgroup float probs[MOE_MAX_EXPERTS];
    threadgroup int selected[MOE_MAX_TOP_K];
    threadgroup float selected_probs[MOE_MAX_TOP_K];

    if (token >= num_tokens || num_experts > MOE_MAX_EXPERTS || top_k > MOE_MAX_TOP_K) return;

    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float v = float(hidden[token * hidden_size + i]);
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
    float rms = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    if (tid < MOE_MAX_EXPERTS) {
        logits[tid] = -INFINITY;
        probs[tid] = 0.0f;
    }
    if (tid < MOE_MAX_TOP_K) {
        selected[tid] = -1;
        selected_probs[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_experts) {
        float acc = 0.0f;
        for (uint h = 0; h < hidden_size; h++) {
            float x = float(hidden[token * hidden_size + h]) * rms
                    * router_scale[h] * scalar_root_size;
            acc += x * router_proj[tid * hidden_size + h];
        }
        logits[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    if (tid < num_experts) {
        local_max = logits[tid];
    }
    shared_sum[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] = max(shared_sum[tid], shared_sum[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_logit = shared_sum[0];

    float local_exp = 0.0f;
    if (tid < num_experts) {
        float p = exp(logits[tid] - max_logit);
        probs[tid] = p;
        local_exp = p;
    }
    shared_sum[tid] = local_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = shared_sum[0] > 0.0f ? (1.0f / shared_sum[0]) : 0.0f;
    if (tid < num_experts) {
        probs[tid] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float selected_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float best = -1.0f;
            int best_idx = -1;
            for (uint e = 0; e < num_experts; e++) {
                bool used = false;
                for (uint prev = 0; prev < k; prev++) {
                    used = used || (selected[prev] == int(e));
                }
                float p = used ? -1.0f : probs[e];
                if (p > best || (p == best && int(e) < best_idx)) {
                    best = p;
                    best_idx = int(e);
                }
            }
            selected[k] = best_idx;
            selected_probs[k] = max(best, 0.0f);
            selected_sum += selected_probs[k];
        }
        float inv_selected = selected_sum > 0.0f ? (1.0f / selected_sum) : 0.0f;
        for (uint k = 0; k < top_k; k++) {
            int expert = selected[k];
            float w = selected_probs[k] * inv_selected;
            if (expert >= 0) {
                w *= router_per_expert_scale[uint(expert)];
            }
            topk_indices[token * top_k + k] = expert;
            topk_weights[token * top_k + k] = w;
        }
    }
}

kernel void moe_expert_gate_up_f16(
    device const half *hidden       [[buffer(0)]],
    device const half *expert_gate_up [[buffer(1)]],
    device const int  *topk_indices [[buffer(2)]],
    device const float *topk_weights [[buffer(3)]],
    device half       *activated    [[buffer(4)]],
    constant uint     &num_tokens   [[buffer(5)]],
    constant uint     &hidden_size  [[buffer(6)]],
    constant uint     &num_experts  [[buffer(7)]],
    constant uint     &top_k        [[buffer(8)]],
    constant uint     &intermediate [[buffer(9)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint route = gid.y;
    uint dim = gid.z;
    if (token >= num_tokens || route >= top_k || dim >= intermediate) return;
    int expert_i = topk_indices[token * top_k + route];
    uint out_idx = (token * top_k + route) * intermediate + dim;
    if (expert_i < 0 || uint(expert_i) >= num_experts) {
        activated[out_idx] = half(0.0h);
        return;
    }
    uint expert = uint(expert_i);
    uint two_intermediate = 2 * intermediate;
    uint gate_base = (expert * two_intermediate + dim) * hidden_size;
    uint up_base = (expert * two_intermediate + intermediate + dim) * hidden_size;
    float gate = 0.0f;
    float up = 0.0f;
    uint hidden_base = token * hidden_size;
    for (uint h = 0; h < hidden_size; h++) {
        float x = float(hidden[hidden_base + h]);
        gate += x * float(expert_gate_up[gate_base + h]);
        up += x * float(expert_gate_up[up_base + h]);
    }
    float weight = topk_weights[token * top_k + route];
    activated[out_idx] = f16_sat(gelu_tanh(gate) * up * weight);
}

kernel void moe_expert_down_f16(
    device const half *activated    [[buffer(0)]],
    device const half *expert_down  [[buffer(1)]],
    device const int  *topk_indices [[buffer(2)]],
    device half       *output       [[buffer(3)]],
    constant uint     &num_tokens   [[buffer(4)]],
    constant uint     &hidden_size  [[buffer(5)]],
    constant uint     &num_experts  [[buffer(6)]],
    constant uint     &top_k        [[buffer(7)]],
    constant uint     &intermediate [[buffer(8)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint hidden_dim = gid.y;
    if (token >= num_tokens || hidden_dim >= hidden_size) return;

    float acc = 0.0f;
    for (uint route = 0; route < top_k; route++) {
        int expert_i = topk_indices[token * top_k + route];
        if (expert_i < 0 || uint(expert_i) >= num_experts) {
            continue;
        }
        uint expert = uint(expert_i);
        uint act_base = (token * top_k + route) * intermediate;
        uint down_base = (expert * hidden_size + hidden_dim) * intermediate;
        for (uint j = 0; j < intermediate; j++) {
            acc += float(activated[act_base + j]) * float(expert_down[down_base + j]);
        }
    }
    output[token * hidden_size + hidden_dim] = f16_sat(acc);
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
    constant uint     &hidden     [[buffer(3)]],
    device const half *layer_scale [[buffer(4)]],
    constant uint     &layer_scale_dim [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float scale = 1.0f;
    if (layer_scale_dim == 1) {
        scale = float(layer_scale[0]);
    } else if (layer_scale_dim == hidden) {
        scale = float(layer_scale[gid % hidden]);
    }
    residual[gid] = f16_sat(float(residual[gid]) + float(addition[gid]) * scale);
}

kernel void residual_add_then_scale_f16(
    device half       *residual   [[buffer(0)]],
    device const half *addition   [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    device const half *layer_scale [[buffer(4)]],
    constant uint     &layer_scale_dim [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float scale = 1.0f;
    if (layer_scale_dim == 1) {
        scale = float(layer_scale[0]);
    } else if (layer_scale_dim == hidden) {
        scale = float(layer_scale[gid % hidden]);
    }
    half updated = f16_sat(float(residual[gid]) + float(addition[gid]));
    residual[gid] = f16_sat(float(updated) * scale);
}

kernel void residual_add_rmsnorm_f16(
    device half       *residual   [[buffer(0)]],
    device const half *addition   [[buffer(1)]],
    device half       *output     [[buffer(2)]],
    device const half *gamma      [[buffer(3)]],
    constant uint     &hidden     [[buffer(4)]],
    constant float    &eps        [[buffer(5)]],
    device const half *layer_scale [[buffer(6)]],
    constant uint     &layer_scale_dim [[buffer(7)]],
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
        float scale = 1.0f;
        if (layer_scale_dim == 1) {
            scale = float(layer_scale[0]);
        } else if (layer_scale_dim == hidden) {
            scale = float(layer_scale[i]);
        }
        half updated = f16_sat(float(residual[idx]) + float(addition[idx]) * scale);
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
        if (argmax_better(v, int(i), local_max, local_idx)) {
            local_max = v;
            local_idx = int(i);
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
        if (argmax_better(v, int(i), local_max, local_idx)) {
            local_max = v;
            local_idx = int(i);
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

kernel void final_lm_head_argmax_tiles_f16(
    device const half *normed_hidden [[buffer(0)]],
    device const half *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]]
) {
    uint token = gid.x;
    uint tile = gid.y;
    if (token >= num_tokens || tile >= tile_count) return;

    uint col_base = tile * FINAL_ARGMAX_TILE_N;
    threadgroup float partial[8][256];
    threadgroup float tile_vals[8];
    threadgroup int tile_indices[8];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    uint hidden_base = token * hidden;
    for (uint k = tid; k < hidden; k += GEMV_TG) {
        float a = float(normed_hidden[hidden_base + k]);
        uint col = col_base;
        if (col < vocab) acc0 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc1 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc2 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc3 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc4 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc5 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc6 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc7 += a * float(lm_head[col * hidden + k]);
    }

    partial[0][tid] = acc0;
    partial[1][tid] = acc1;
    partial[2][tid] = acc2;
    partial[3][tid] = acc3;
    partial[4][tid] = acc4;
    partial[5][tid] = acc5;
    partial[6][tid] = acc6;
    partial[7][tid] = acc7;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GEMV_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
            partial[5][tid] += partial[5][tid + stride];
            partial[6][tid] += partial[6][tid + stride];
            partial[7][tid] += partial[7][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < FINAL_ARGMAX_TILE_N) {
        uint col = col_base + tid;
        float score = -INFINITY;
        int idx = 0;
        if (col < vocab) {
            half quantized = f16_sat(partial[tid][0]);
            score = float(quantized);
            if (softcap > 0.0f) {
                score = float(f16_sat(softcap * tanh(score / softcap)));
            }
            idx = int(col);
        }
        tile_vals[tid] = score;
        tile_indices[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float best_val = tile_vals[0];
        int best_idx = tile_indices[0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[i], tile_indices[i], best_val, best_idx)) {
                best_val = tile_vals[i];
                best_idx = tile_indices[i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

// Apple9/10 LM-head projection for 2 <= num_tokens <= 8. One threadgroup
// covers eight vocabulary columns for the entire request microbatch, sharing
// each `[vocab,hidden]` weight read across all request rows. It emits the same
// per-token/per-tile partial argmax ABI consumed by final_argmax_reduce_f32.
kernel void final_lm_head_argmax_tiles_batch8_f16(
    device const half *normed_hidden [[buffer(0)]],
    device const half *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint tile                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    ushort lane                       [[thread_index_in_simdgroup]],
    ushort simdgroup                  [[simdgroup_index_in_threadgroup]]
) {
    if (tile >= tile_count) return;
    uint col = tile * FINAL_ARGMAX_TILE_N + uint(simdgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    if (col < vocab) {
        for (uint k = uint(lane); k < hidden; k += 32u) {
            float b = float(lm_head[col * hidden + k]);
            if (0u < num_tokens) acc0 += float(normed_hidden[0u * hidden + k]) * b;
            if (1u < num_tokens) acc1 += float(normed_hidden[1u * hidden + k]) * b;
            if (2u < num_tokens) acc2 += float(normed_hidden[2u * hidden + k]) * b;
            if (3u < num_tokens) acc3 += float(normed_hidden[3u * hidden + k]) * b;
            if (4u < num_tokens) acc4 += float(normed_hidden[4u * hidden + k]) * b;
            if (5u < num_tokens) acc5 += float(normed_hidden[5u * hidden + k]) * b;
            if (6u < num_tokens) acc6 += float(normed_hidden[6u * hidden + k]) * b;
            if (7u < num_tokens) acc7 += float(normed_hidden[7u * hidden + k]) * b;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    threadgroup float tile_vals[8][8];
    threadgroup int tile_indices[8][8];
    if (lane == 0) {
        float totals[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        for (uint token = 0; token < 8u; token++) {
            float score = -INFINITY;
            int idx = 0;
            if (token < num_tokens && col < vocab) {
                half quantized = f16_sat(totals[token]);
                score = float(quantized);
                if (softcap > 0.0f) {
                    score = float(f16_sat(softcap * tanh(score / softcap)));
                }
                idx = int(col);
            }
            tile_vals[token][simdgroup] = score;
            tile_indices[token][simdgroup] = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_tokens) {
        uint token = tid;
        float best_val = tile_vals[token][0];
        int best_idx = tile_indices[token][0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[token][i], tile_indices[token][i], best_val, best_idx)) {
                best_val = tile_vals[token][i];
                best_idx = tile_indices[token][i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

// Batch-four LM-head projection. Like the qualified batch-eight route, this
// shares each vocabulary weight row across the whole cohort, but keeps only
// four request accumulators live to avoid the register pressure of the M8
// kernel on a half-full batch.
kernel void final_lm_head_argmax_tiles_batch4_f16(
    device const half *normed_hidden [[buffer(0)]],
    device const half *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint tile                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    ushort lane                       [[thread_index_in_simdgroup]],
    ushort simdgroup                  [[simdgroup_index_in_threadgroup]]
) {
    if (tile >= tile_count) return;
    uint col = tile * FINAL_ARGMAX_TILE_N + uint(simdgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    if (col < vocab) {
        for (uint k = uint(lane); k < hidden; k += 32u) {
            float weight = float(lm_head[col * hidden + k]);
            acc0 += float(normed_hidden[0u * hidden + k]) * weight;
            acc1 += float(normed_hidden[1u * hidden + k]) * weight;
            acc2 += float(normed_hidden[2u * hidden + k]) * weight;
            acc3 += float(normed_hidden[3u * hidden + k]) * weight;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    threadgroup float tile_vals[4][8];
    threadgroup int tile_indices[4][8];
    if (lane == 0) {
        float totals[4] = {acc0, acc1, acc2, acc3};
        for (uint token = 0; token < 4u; token++) {
            float score = -INFINITY;
            int idx = 0;
            if (col < vocab) {
                half quantized = f16_sat(totals[token]);
                score = float(quantized);
                if (softcap > 0.0f) {
                    score = float(f16_sat(softcap * tanh(score / softcap)));
                }
                idx = int(col);
            }
            tile_vals[token][simdgroup] = score;
            tile_indices[token][simdgroup] = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4u) {
        uint token = tid;
        float best_val = tile_vals[token][0];
        int best_idx = tile_indices[token][0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[token][i], tile_indices[token][i], best_val, best_idx)) {
                best_val = tile_vals[token][i];
                best_idx = tile_indices[token][i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

kernel void final_argmax_reduce_f32(
    device const float *partial_max [[buffer(0)]],
    device const int   *partial_idx [[buffer(1)]],
    device int         *output      [[buffer(2)]],
    constant uint      &num_tokens  [[buffer(3)]],
    constant uint      &tile_count  [[buffer(4)]],
    uint token                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    if (token >= num_tokens) return;

    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    float local_max = -INFINITY;
    int local_idx = 0;
    uint base = token * tile_count;
    for (uint tile = tid; tile < tile_count; tile += tg_size) {
        float val = partial_max[base + tile];
        int idx = partial_idx[base + tile];
        if (argmax_better(val, idx, local_max, local_idx)) {
            local_max = val;
            local_idx = idx;
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

pub fn kernel_source_for_float_type(float_type: MetalFloatType) -> Cow<'static, str> {
    kernel_source_with_options(
        float_type,
        crate::MetalKernelOptions::from_development_environment(),
    )
}

pub fn kernel_source_with_options(
    float_type: MetalFloatType,
    options: crate::MetalKernelOptions,
) -> Cow<'static, str> {
    let base = match float_type {
        MetalFloatType::F16 => Cow::Borrowed(KERNEL_SOURCE),
        MetalFloatType::Bf16 => {
            Cow::Owned(bfloat_kernel_source(options.quantized_bf16_accumulation))
        }
    };
    let candidate = options.research.source();
    if candidate.is_empty() {
        return base;
    }
    let mut source = base.into_owned();
    source.push('\n');
    match float_type {
        MetalFloatType::F16 => source.push_str(candidate),
        MetalFloatType::Bf16 => source.push_str(
            &replace_msl_word(candidate, "half", "bfloat").replace("f16_sat", "bf16_sat"),
        ),
    }
    Cow::Owned(source)
}

fn bfloat_kernel_source(quantized_accumulation: bool) -> String {
    // W4A16/W8A16 have an explicit activation/scales ABI and must remain F16
    // even in a library containing the typed BF16 transformer variants.
    const LOW_BIT_START: &str = "// Group-32 low-bit projection (W4A16 / W8A16)";
    const LOW_BIT_END: &str = "constant uint TILE_M";
    const LOW_BIT_PLACEHOLDER: &str = "/* RVLLM_LOW_BIT_A16_KERNELS */\n";
    let start = KERNEL_SOURCE
        .find(LOW_BIT_START)
        .expect("low-bit kernel start marker");
    let start = KERNEL_SOURCE[..start]
        .rfind("// ============================================================================")
        .expect("low-bit kernel section marker");
    let end = KERNEL_SOURCE[start..]
        .find(LOW_BIT_END)
        .map(|relative| start + relative)
        .expect("low-bit kernel end marker");
    let low_bit_a16 = &KERNEL_SOURCE[start..end];
    let source_without_low_bit = format!(
        "{}{}{}",
        &KERNEL_SOURCE[..start],
        LOW_BIT_PLACEHOLDER,
        &KERNEL_SOURCE[end..]
    );

    let mut source = source_without_low_bit.replace("f16_sat", "bf16_sat");
    source = replace_msl_word(&source, "half", "bfloat");
    source = source.replace(
        "device const float *router_proj",
        "device const bfloat *router_proj",
    );
    source = source.replace(
        "device const float *router_scale",
        "device const bfloat *router_scale",
    );
    source = source.replace(
        "device const float *router_per_expert_scale",
        "device const bfloat *router_per_expert_scale",
    );
    source = source.replace("0.0h", "0.0f");
    source = source.replace(
        "return bfloat(clamp(x, -65504.0f, 65504.0f));",
        "return bfloat(x);",
    );
    source = source.replace(
        "static inline bfloat bf16_sat(float x) {\n    return bfloat(x);\n}\n",
        "static inline bfloat bf16_sat(float x) {\n    return bfloat(x);\n}\n\nstatic inline float bf16_acc(float x) {\n    return float(bfloat(x));\n}\n",
    );
    if quantized_accumulation {
        source = quantize_bfloat_accumulators(&source);
    }
    source.replace(LOW_BIT_PLACEHOLDER, low_bit_a16)
}

fn quantize_bfloat_accumulators(source: &str) -> String {
    let replacements = [
        ("acc += a * b;", "acc = bf16_acc(acc + a * b);"),
        (
            "acc += float(tile_a[tid.x][kk]) * float(tile_b[kk][tid.y]);",
            "acc = bf16_acc(acc + float(tile_a[tid.x][kk]) * float(tile_b[kk][tid.y]));",
        ),
        (
            "acc += float(A[row * K + k]) * float(B[col * K + k]);",
            "acc = bf16_acc(acc + float(A[row * K + k]) * float(B[col * K + k]));",
        ),
        (
            "acc += float(A[token * K + k]) * float(B[row * K + k]);",
            "acc = bf16_acc(acc + float(A[token * K + k]) * float(B[row * K + k]));",
        ),
        (
            "acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);",
            "acc = bf16_acc(acc + float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]));",
        ),
        (
            "acc += float(A[row * K + k]) * float(B[col * K + k]);",
            "acc = bf16_acc(acc + float(A[row * K + k]) * float(B[col * K + k]));",
        ),
        (
            "acc += weight * float(v_cache[v_idx]);",
            "acc = bf16_acc(acc + weight * float(v_cache[v_idx]));",
        ),
        (
            "acc += x * router_proj[tid * hidden_size + h];",
            "acc = bf16_acc(acc + x * float(router_proj[tid * hidden_size + h]));",
        ),
        (
            "gate += x * float(expert_gate_up[gate_base + h]);",
            "gate = bf16_acc(gate + x * float(expert_gate_up[gate_base + h]));",
        ),
        (
            "up += x * float(expert_gate_up[up_base + h]);",
            "up = bf16_acc(up + x * float(expert_gate_up[up_base + h]));",
        ),
        (
            "acc += float(activated[act_base + j]) * float(expert_down[down_base + j]);",
            "acc = bf16_acc(acc + float(activated[act_base + j]) * float(expert_down[down_base + j]));",
        ),
        (
            "acc += normed * float(lm_head[v * hidden + d]);",
            "acc = bf16_acc(acc + normed * float(lm_head[v * hidden + d]));",
        ),
    ];
    let mut out = source.to_owned();
    for (needle, replacement) in replacements {
        out = out.replace(needle, replacement);
    }
    for idx in 0..8 {
        out = out.replace(
            &format!("acc{idx} += a * float(B[col * K + k]);"),
            &format!("acc{idx} = bf16_acc(acc{idx} + a * float(B[col * K + k]));"),
        );
        out = out.replace(
            &format!("acc{idx} += a * float(lm_head[col * hidden + k]);"),
            &format!("acc{idx} = bf16_acc(acc{idx} + a * float(lm_head[col * hidden + k]));"),
        );
    }
    out
}

fn replace_msl_word(source: &str, word: &str, replacement: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;
    while let Some(relative) = source[cursor..].find(word) {
        let start = cursor + relative;
        let end = start + word.len();
        let before = source[..start]
            .chars()
            .next_back()
            .is_some_and(is_msl_ident_char);
        let after = source[end..].chars().next().is_some_and(is_msl_ident_char);
        if before || after {
            out.push_str(&source[cursor..end]);
        } else {
            out.push_str(&source[cursor..start]);
            out.push_str(replacement);
        }
        cursor = end;
    }
    out.push_str(&source[cursor..]);
    out
}

fn is_msl_ident_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

/// Number of kernels defined in the source.
pub const KERNEL_COUNT: usize = KERNEL_NAMES.len();

/// List of all kernel function names.
pub const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f16",
    "rmsnorm_headwise_f16",
    "rmsnorm_headwise_unit_f16",
    "gemm_f16",
    "gemm_f16_tiled16",
    "gemm_f16_vec8",
    "gemm_f16_batch8",
    "qkv_project_f32_batch8",
    "qkv_project_f32_mma32",
    "gemm_f16_mma32",
    "qkv_projected_rmsnorm_rope_cache_f16",
    "gemm_f16_simdgroup8x8",
    "projection_w4a16_f16",
    "projection_w8a16_f16",
    "experimental_projection_w4abf16_bf16",
    "experimental_projection_w8abf16_bf16",
    "experimental_projection_w4abf16_bf16_n4",
    "experimental_projection_w8abf16_bf16_n4",
    "experimental_projection_w4abf16_bf16_n8",
    "experimental_projection_w8abf16_bf16_n8",
    "experimental_projection_w4abf16_bf16_n4_packed2",
    "experimental_projection_w8abf16_bf16_n8_k4",
    "gemm_rmsnorm_f16",
    "gemm_headwise_rmsnorm_f16",
    "gemm_headwise_rmsnorm_unit_f16",
    "qkv_headwise_rmsnorm_f16",
    "qkv_headwise_rmsnorm_rope_cache_f16",
    "copy_f16",
    "copy_ple_layer_f16",
    "split_qkv_f16",
    "rope_partial_f16",
    "kv_cache_write_f16",
    "experimental_kv_quantize_int8_f16",
    "experimental_kv_dequantize_int8_f16",
    "attention_decode_f16",
    "attention_decode_online_f16",
    "attention_prefill_f16",
    "attention_prefill_simdgroup_f16",
    "embedding_gather_f16",
    "gelu_mul_f16",
    "moe_router_topk_f16",
    "moe_expert_gate_up_f16",
    "moe_expert_down_f16",
    "ple_combine_f16",
    "ple_gelu_mul_f16",
    "residual_add_f16",
    "residual_add_then_scale_f16",
    "residual_add_rmsnorm_f16",
    "layer_scale_f16",
    "argmax_f16",
    "softcap_argmax_f16",
    "final_norm_lm_head_argmax_small_f16",
    "final_lm_head_argmax_tiles_f16",
    "final_lm_head_argmax_tiles_batch4_f16",
    "final_lm_head_argmax_tiles_batch8_f16",
    "final_argmax_reduce_f32",
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

#[cfg(all(test, target_os = "macos"))]
#[path = "gemm_reduction_tests.rs"]
mod gemm_reduction_tests;

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

    #[cfg(target_os = "macos")]
    unsafe fn write_f32_region(
        arena: &MetalBufferArena,
        region: &crate::arena::MetalRegion,
        values: &[f32],
    ) {
        let ptr = arena.host_ptr(region) as *mut f32;
        for (idx, value) in values.iter().enumerate() {
            *ptr.add(idx) = *value;
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn write_i32_region(
        arena: &MetalBufferArena,
        region: &crate::arena::MetalRegion,
        values: &[i32],
    ) {
        let ptr = arena.host_ptr(region) as *mut i32;
        for (idx, value) in values.iter().enumerate() {
            *ptr.add(idx) = *value;
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn read_i32_region(
        arena: &MetalBufferArena,
        region: &crate::arena::MetalRegion,
        len: usize,
    ) -> Vec<i32> {
        std::slice::from_raw_parts(arena.host_ptr(region) as *const i32, len).to_vec()
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

    fn moe_router_topk_ref(
        hidden: &[f32],
        router_proj: &[f32],
        router_scale: &[f32],
        per_expert_scale: &[f32],
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        eps: f32,
    ) -> (Vec<i32>, Vec<f32>) {
        assert_eq!(hidden.len(), hidden_size);
        assert_eq!(router_proj.len(), num_experts * hidden_size);
        assert_eq!(router_scale.len(), hidden_size);
        assert_eq!(per_expert_scale.len(), num_experts);

        let inv_rms = 1.0
            / (hidden.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32 + eps)
                .sqrt();
        let scalar_root_size = (hidden_size as f32).powf(-0.5);
        let mut logits = vec![0.0f32; num_experts];
        for expert in 0..num_experts {
            let mut acc = 0.0f32;
            for dim in 0..hidden_size {
                let x = hidden[dim] * inv_rms * router_scale[dim] * scalar_root_size;
                acc += x * router_proj[expert * hidden_size + dim];
            }
            logits[expert] = acc;
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|value| (value - max_logit).exp())
            .collect();
        let sum = probs.iter().sum::<f32>();
        for prob in &mut probs {
            *prob /= sum;
        }

        let mut selected = Vec::with_capacity(top_k);
        let mut selected_probs = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            let mut best_idx = 0usize;
            let mut best = -1.0f32;
            for (expert, &prob) in probs.iter().enumerate() {
                if selected.contains(&(expert as i32)) {
                    continue;
                }
                if prob > best {
                    best = prob;
                    best_idx = expert;
                }
            }
            selected.push(best_idx as i32);
            selected_probs.push(best.max(0.0));
        }
        let selected_sum = selected_probs.iter().sum::<f32>();
        let weights = selected
            .iter()
            .zip(selected_probs.iter())
            .map(|(&expert, &prob)| {
                prob / selected_sum * per_expert_scale[usize::try_from(expert).unwrap()]
            })
            .collect();
        (selected, weights)
    }

    fn moe_experts_ref(
        hidden: &[f32],
        expert_gate_up: &[f32],
        expert_down: &[f32],
        topk_indices: &[i32],
        topk_weights: &[f32],
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        assert_eq!(hidden.len(), hidden_size);
        assert_eq!(
            expert_gate_up.len(),
            num_experts * 2 * intermediate * hidden_size
        );
        assert_eq!(expert_down.len(), num_experts * hidden_size * intermediate);
        assert_eq!(topk_indices.len(), top_k);
        assert_eq!(topk_weights.len(), top_k);

        let mut out = vec![0.0f32; hidden_size];
        for route in 0..top_k {
            let expert = usize::try_from(topk_indices[route]).unwrap();
            let mut activated = vec![0.0f32; intermediate];
            for dim in 0..intermediate {
                let gate_base = (expert * 2 * intermediate + dim) * hidden_size;
                let up_base = (expert * 2 * intermediate + intermediate + dim) * hidden_size;
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for hidden_dim in 0..hidden_size {
                    gate += hidden[hidden_dim] * expert_gate_up[gate_base + hidden_dim];
                    up += hidden[hidden_dim] * expert_gate_up[up_base + hidden_dim];
                }
                activated[dim] = gelu_tanh_ref(gate) * up * topk_weights[route];
            }
            for hidden_dim in 0..hidden_size {
                let down_base = (expert * hidden_size + hidden_dim) * intermediate;
                for dim in 0..intermediate {
                    out[hidden_dim] += activated[dim] * expert_down[down_base + dim];
                }
            }
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

    #[test]
    fn kernel_moe_router_topk_reference_normalizes_and_scales() {
        let hidden = vec![1.0, 0.0];
        let router_proj = vec![1.0, 0.0, 0.5, 0.0, -1.0, 0.0];
        let router_scale = vec![1.0, 1.0];
        let per_expert_scale = vec![1.0, 2.0, 1.0];

        let (indices, weights) = moe_router_topk_ref(
            &hidden,
            &router_proj,
            &router_scale,
            &per_expert_scale,
            2,
            3,
            2,
            0.0,
        );

        assert_eq!(indices, vec![0, 1]);
        assert!((weights[0] - 0.62245935).abs() < 1e-5);
        assert!((weights[1] - 0.7550813).abs() < 1e-5);
    }

    #[test]
    fn kernel_moe_experts_reference_uses_expert_major_layout() {
        let hidden = vec![1.0, 2.0];
        let expert_gate_up = vec![
            // expert 0: gate rows then up rows
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, // expert 1
            0.5, 0.0, 0.0, 0.5, 2.0, 0.0, 0.0, 2.0,
        ];
        let expert_down = vec![
            // expert 0: hidden rows
            1.0, 0.0, 0.0, 1.0, // expert 1
            1.0, 1.0, -1.0, 1.0,
        ];
        let topk_indices = vec![0, 1];
        let topk_weights = vec![0.5, 0.25];

        let got = moe_experts_ref(
            &hidden,
            &expert_gate_up,
            &expert_down,
            &topk_indices,
            &topk_weights,
            2,
            2,
            2,
            2,
        );

        let expert0_dim0 = gelu_tanh_ref(1.0) * 3.0 * 0.5;
        let expert0_dim1 = gelu_tanh_ref(2.0) * 1.0 * 0.5;
        let expert1_dim0 = gelu_tanh_ref(0.5) * 2.0 * 0.25;
        let expert1_dim1 = gelu_tanh_ref(1.0) * 4.0 * 0.25;
        let expected = vec![
            expert0_dim0 + expert1_dim0 + expert1_dim1,
            expert0_dim1 - expert1_dim0 + expert1_dim1,
        ];
        for (got, expected) in got.iter().zip(expected.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gelu_mul_large_intermediate_matches_cpu_and_overwrites_output() -> rvllm_core::Result<()> {
        const NUM_TOKENS: u32 = 4;
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
    fn gemm_beta_zero_ignores_nan_c_for_all_projection_paths_macos() -> rvllm_core::Result<()> {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const M: u32 = 8;
        const N: u32 = 16;
        const K: u32 = 8;
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
        let vec_region = arena.region("gemm_beta0_vec", nan_c.len() * half_bytes, 2)?;
        let batch8_region = arena.region("gemm_beta0_batch8", nan_c.len() * half_bytes, 2)?;

        unsafe {
            write_f16_region(&arena, &a_region, &a_f16);
            write_f16_region(&arena, &b_region, &b_f16);
            write_f16_region(&arena, &general_region, &nan_c);
            write_f16_region(&arena, &tiled_region, &nan_c);
            write_f16_region(&arena, &vec_region, &nan_c);
            write_f16_region(&arena, &batch8_region, &nan_c);
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
        for (kernel, output_offset, tile, is_vec, is_batch8, op) in [
            (
                "gemm_f16",
                general_region.offset,
                8usize,
                false,
                false,
                "gemm_beta0_general",
            ),
            (
                "gemm_f16_tiled16",
                tiled_region.offset,
                16usize,
                false,
                false,
                "gemm_beta0_tiled",
            ),
            (
                "gemm_f16_vec8",
                vec_region.offset,
                8usize,
                true,
                false,
                "gemm_beta0_vec",
            ),
            (
                "gemm_f16_batch8",
                batch8_region.offset,
                8usize,
                false,
                true,
                "gemm_beta0_batch8",
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
                if is_batch8 {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: (M as usize).div_ceil(8),
                            height: (N as usize).div_ceil(8),
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else if is_vec {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: M as usize,
                            height: (N as usize).div_ceil(8),
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
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
                }
                encoder.endEncoding();
            }
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = gemm_ref_f32(&a, &b, M as usize, N as usize, K as usize);
        for (name, region) in [
            ("gemm_f16", &general_region),
            ("gemm_f16_tiled16", &tiled_region),
            ("gemm_f16_vec8", &vec_region),
            ("gemm_f16_batch8", &batch8_region),
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
    #[ignore = "manual Apple Silicon projection microbenchmark"]
    fn gemm_batch8_m4_projection_microbenchmark() -> rvllm_core::Result<()> {
        use std::time::Instant;

        const M: u32 = 8;
        const N: u32 = 2_304;
        const K: u32 = 2_304;
        const WARMUP: usize = 2;
        const ITERS: usize = 10;

        let half_bytes = std::mem::size_of::<f16>();
        let a_len = (M * K) as usize;
        let b_len = (N * K) as usize;
        let c_len = (M * N) as usize;
        let arena_bytes = (a_len + b_len + 2 * c_len) * half_bytes + 4096;
        let (ctx, pipelines, mut arena) = metal_test_context(arena_bytes)?;
        let a_region = arena.region("batch8_bench_a", a_len * half_bytes, 16)?;
        let b_region = arena.region("batch8_bench_b", b_len * half_bytes, 16)?;
        let vec_region = arena.region("batch8_bench_vec", c_len * half_bytes, 16)?;
        let batch_region = arena.region("batch8_bench_batch", c_len * half_bytes, 16)?;
        let a = (0..a_len)
            .map(|idx| f16::from_f32(((idx % 29) as f32 - 14.0) / 128.0))
            .collect::<Vec<_>>();
        let b = (0..b_len)
            .map(|idx| f16::from_f32(((idx % 31) as f32 - 15.0) / 128.0))
            .collect::<Vec<_>>();
        let zero = vec![f16::ZERO; c_len];
        unsafe {
            write_f16_region(&arena, &a_region, &a);
            write_f16_region(&arena, &b_region, &b);
            write_f16_region(&arena, &vec_region, &zero);
            write_f16_region(&arena, &batch_region, &zero);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let run = |kernel: &str, output_offset: usize, iterations: usize| {
            let started = Instant::now();
            for _ in 0..iterations {
                let command_buffer = queue.commandBuffer().ok_or_else(|| {
                    rvllm_core::RvllmError::apple(
                        rvllm_core::AppleError::MetalUnavailable,
                        rvllm_core::AppleCtx {
                            backend: "metal",
                            op: "gemm_batch8_microbench_command_buffer",
                            device: "apple-silicon",
                        },
                    )
                })?;
                let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                    rvllm_core::RvllmError::apple(
                        rvllm_core::AppleError::MetalUnavailable,
                        rvllm_core::AppleCtx {
                            backend: "metal",
                            op: "gemm_batch8_microbench_encoder",
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
                    let groups = if kernel == "gemm_f16_simdgroup8x8" {
                        MTLSize {
                            width: (M as usize).div_ceil(8),
                            height: (N as usize).div_ceil(8),
                            depth: 1,
                        }
                    } else {
                        MTLSize {
                            width: M as usize,
                            height: (N as usize).div_ceil(8),
                            depth: 1,
                        }
                    };
                    let threads = if kernel == "gemm_f16_simdgroup8x8" {
                        32
                    } else {
                        256
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        groups,
                        MTLSize {
                            width: threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.endEncoding();
                }
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
            Ok::<_, rvllm_core::RvllmError>(started.elapsed())
        };

        run("gemm_f16_vec8", vec_region.offset, WARMUP)?;
        run("gemm_f16_simdgroup8x8", batch_region.offset, WARMUP)?;
        let vec_elapsed = run("gemm_f16_vec8", vec_region.offset, ITERS)?;
        let batch_elapsed = run("gemm_f16_simdgroup8x8", batch_region.offset, ITERS)?;
        let vec_output = unsafe { read_f16_region(&arena, &vec_region, c_len) };
        let batch_output = unsafe { read_f16_region(&arena, &batch_region, c_len) };
        let max_abs_delta = vec_output
            .iter()
            .zip(&batch_output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_delta <= 0.015625, "max delta={max_abs_delta}");

        let vec_ms = vec_elapsed.as_secs_f64() * 1_000.0 / ITERS as f64;
        let batch_ms = batch_elapsed.as_secs_f64() * 1_000.0 / ITERS as f64;
        eprintln!(
            "gemm M={M} N={N} K={K}: vec8={vec_ms:.3}ms simdgroup8x8={batch_ms:.3}ms speedup={:.2}x max_abs_delta={max_abs_delta}",
            vec_ms / batch_ms
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn final_lm_head_argmax_tiles_matches_cpu_reference() -> rvllm_core::Result<()> {
        // The batch4 kernel is deliberately specialized for exactly four
        // rows. Keeping the shared fixture at four prevents a test-only
        // out-of-bounds launch from hiding real output corruption.
        const NUM_TOKENS: u32 = 4;
        const HIDDEN: u32 = 11;
        const VOCAB: u32 = 19;
        const GUARD_WORDS: usize = 4;
        const GUARD_VALUE: i32 = 0x5A5A_3C3C;
        let tile_count = VOCAB.div_ceil(8);
        let half_bytes = std::mem::size_of::<f16>();
        let partial_len = (NUM_TOKENS * tile_count) as usize;
        let (ctx, pipelines, mut arena) = metal_test_context(
            NUM_TOKENS as usize * HIDDEN as usize * half_bytes
                + VOCAB as usize * HIDDEN as usize * half_bytes
                + 2 * partial_len * std::mem::size_of::<f32>()
                + 2 * partial_len * std::mem::size_of::<i32>()
                + 2 * NUM_TOKENS as usize * std::mem::size_of::<i32>()
                + 3 * GUARD_WORDS * std::mem::size_of::<i32>()
                + 4096,
        )?;
        let hidden_region = arena.region(
            "final_argmax_hidden",
            NUM_TOKENS as usize * HIDDEN as usize * half_bytes,
            16,
        )?;
        let lm_head_region = arena.region(
            "final_argmax_lm_head",
            VOCAB as usize * HIDDEN as usize * half_bytes,
            16,
        )?;
        let partial_max_region = arena.region(
            "final_argmax_partial_max",
            partial_len * std::mem::size_of::<f32>(),
            16,
        )?;
        let partial_idx_region = arena.region(
            "final_argmax_partial_idx",
            partial_len * std::mem::size_of::<i32>(),
            16,
        )?;
        let sampled_region = arena.region(
            "final_argmax_sampled",
            NUM_TOKENS as usize * std::mem::size_of::<i32>(),
            4,
        )?;
        let batch_partial_max_region = arena.region(
            "final_argmax_batch_partial_max",
            partial_len * std::mem::size_of::<f32>(),
            16,
        )?;
        let batch_partial_max_guard = arena.region(
            "final_argmax_batch_partial_max_guard",
            GUARD_WORDS * std::mem::size_of::<i32>(),
            16,
        )?;
        let batch_partial_idx_region = arena.region(
            "final_argmax_batch_partial_idx",
            partial_len * std::mem::size_of::<i32>(),
            16,
        )?;
        let batch_partial_idx_guard = arena.region(
            "final_argmax_batch_partial_idx_guard",
            GUARD_WORDS * std::mem::size_of::<i32>(),
            16,
        )?;
        let batch_sampled_region = arena.region(
            "final_argmax_batch_sampled",
            NUM_TOKENS as usize * std::mem::size_of::<i32>(),
            4,
        )?;
        let batch_sampled_guard = arena.region(
            "final_argmax_batch_sampled_guard",
            GUARD_WORDS * std::mem::size_of::<i32>(),
            16,
        )?;

        let mut hidden = Vec::with_capacity((NUM_TOKENS * HIDDEN) as usize);
        for token in 0..NUM_TOKENS as usize {
            for dim in 0..HIDDEN as usize {
                hidden.push(f16::from_f32(((token * 13 + dim * 7) as f32 - 31.0) / 17.0));
            }
        }
        let mut lm_head = Vec::with_capacity((VOCAB * HIDDEN) as usize);
        for vocab in 0..VOCAB as usize {
            for dim in 0..HIDDEN as usize {
                let mut value = (((vocab * 11 + dim * 5) % 37) as f32 - 18.0) / 19.0;
                if vocab == 7 {
                    value += 0.75;
                }
                if vocab == 16 {
                    value += 1.25;
                }
                lm_head.push(f16::from_f32(value));
            }
        }
        let softcap = 30.0f32;
        let mut expected = Vec::with_capacity(NUM_TOKENS as usize);
        for token in 0..NUM_TOKENS as usize {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx = 0i32;
            for vocab in 0..VOCAB as usize {
                let mut acc = 0.0f32;
                for dim in 0..HIDDEN as usize {
                    acc += hidden[token * HIDDEN as usize + dim].to_f32()
                        * lm_head[vocab * HIDDEN as usize + dim].to_f32();
                }
                let mut score = f16::from_f32(acc).to_f32();
                score = f16::from_f32(softcap * (score / softcap).tanh()).to_f32();
                if score > best_val || (score == best_val && (vocab as i32) < best_idx) {
                    best_val = score;
                    best_idx = vocab as i32;
                }
            }
            expected.push(best_idx);
        }

        unsafe {
            write_f16_region(&arena, &hidden_region, &hidden);
            write_f16_region(&arena, &lm_head_region, &lm_head);
            write_f32_region(&arena, &partial_max_region, &vec![f32::NAN; partial_len]);
            write_i32_region(&arena, &partial_idx_region, &vec![-1; partial_len]);
            write_i32_region(&arena, &sampled_region, &vec![-1; NUM_TOKENS as usize]);
            write_f32_region(
                &arena,
                &batch_partial_max_region,
                &vec![f32::NAN; partial_len],
            );
            write_i32_region(
                &arena,
                &batch_partial_max_guard,
                &vec![GUARD_VALUE; GUARD_WORDS],
            );
            write_i32_region(&arena, &batch_partial_idx_region, &vec![-1; partial_len]);
            write_i32_region(
                &arena,
                &batch_partial_idx_guard,
                &vec![GUARD_VALUE; GUARD_WORDS],
            );
            write_i32_region(
                &arena,
                &batch_sampled_region,
                &vec![-1; NUM_TOKENS as usize],
            );
            write_i32_region(
                &arena,
                &batch_sampled_guard,
                &vec![GUARD_VALUE; GUARD_WORDS],
            );
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_argmax_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        for (kernel, max_offset, idx_offset, batched) in [
            (
                "final_lm_head_argmax_tiles_f16",
                partial_max_region.offset,
                partial_idx_region.offset,
                false,
            ),
            (
                "final_lm_head_argmax_tiles_batch4_f16",
                batch_partial_max_region.offset,
                batch_partial_idx_region.offset,
                true,
            ),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "final_argmax_tiles_encoder",
                        device: "apple-silicon",
                    },
                )
            })?;
            unsafe {
                encoder.setComputePipelineState(pipelines.get(kernel)?);
                encoder.setBuffer_offset_atIndex(Some(buf), hidden_region.offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), lm_head_region.offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), max_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(buf), idx_offset, 3);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&NUM_TOKENS as *const _ as *mut _),
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&VOCAB as *const _ as *mut _),
                    4,
                    5,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                    4,
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&tile_count as *const _ as *mut _),
                    4,
                    7,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&softcap as *const _ as *mut _),
                    4,
                    8,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    if batched {
                        MTLSize {
                            width: tile_count as usize,
                            height: 1,
                            depth: 1,
                        }
                    } else {
                        MTLSize {
                            width: NUM_TOKENS as usize,
                            height: tile_count as usize,
                            depth: 1,
                        }
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
        for (max_offset, idx_offset, sampled_offset) in [
            (
                partial_max_region.offset,
                partial_idx_region.offset,
                sampled_region.offset,
            ),
            (
                batch_partial_max_region.offset,
                batch_partial_idx_region.offset,
                batch_sampled_region.offset,
            ),
        ] {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "final_argmax_reduce_encoder",
                        device: "apple-silicon",
                    },
                )
            })?;
            unsafe {
                encoder.setComputePipelineState(pipelines.get("final_argmax_reduce_f32")?);
                encoder.setBuffer_offset_atIndex(Some(buf), max_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), idx_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), sampled_offset, 2);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&NUM_TOKENS as *const _ as *mut _),
                    4,
                    3,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&tile_count as *const _ as *mut _),
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

        let got = unsafe { read_i32_region(&arena, &sampled_region, NUM_TOKENS as usize) };
        assert_eq!(got, expected);
        let batch_got =
            unsafe { read_i32_region(&arena, &batch_sampled_region, NUM_TOKENS as usize) };
        assert_eq!(batch_got, expected);
        for guard in [
            &batch_partial_max_guard,
            &batch_partial_idx_guard,
            &batch_sampled_guard,
        ] {
            assert_eq!(
                unsafe { read_i32_region(&arena, guard, GUARD_WORDS) },
                vec![GUARD_VALUE; GUARD_WORDS],
                "batch4 final-sampling kernel wrote outside its qualified buffers"
            );
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
        let q_fused_region =
            arena.region("qkv_headwise_q_fused", (M * Q_DIM) as usize * half_bytes, 2)?;
        let k_fused_region = arena.region(
            "qkv_headwise_k_fused",
            (M * KV_DIM) as usize * half_bytes,
            2,
        )?;
        let v_fused_region = arena.region(
            "qkv_headwise_v_fused",
            (M * KV_DIM) as usize * half_bytes,
            2,
        )?;
        let q_rope_cache_region = arena.region(
            "qkv_headwise_q_rope_cache",
            (M * Q_DIM) as usize * half_bytes,
            2,
        )?;
        let k_rope_cache_region = arena.region(
            "qkv_headwise_k_rope_cache",
            (M * KV_DIM) as usize * half_bytes,
            2,
        )?;
        let v_rope_cache_region = arena.region(
            "qkv_headwise_v_rope_cache",
            (M * KV_DIM) as usize * half_bytes,
            2,
        )?;
        let cos = [1.0f32, 1.0, 0.5, 0.25];
        let sin = [0.0f32, 0.0, 0.25, 0.5];
        let positions = [0i32, 1];
        let slots = [1i32, 0];
        let cos_region = arena.region("qkv_headwise_cos", std::mem::size_of_val(&cos), 4)?;
        let sin_region = arena.region("qkv_headwise_sin", std::mem::size_of_val(&sin), 4)?;
        let positions_region = arena.region(
            "qkv_headwise_positions",
            std::mem::size_of_val(&positions),
            4,
        )?;
        let slots_region = arena.region("qkv_headwise_slots", std::mem::size_of_val(&slots), 4)?;
        let k_cache_region = arena.region(
            "qkv_headwise_k_cache",
            (2 * KV_DIM) as usize * half_bytes,
            2,
        )?;
        let v_cache_region = arena.region(
            "qkv_headwise_v_cache",
            (2 * KV_DIM) as usize * half_bytes,
            2,
        )?;
        unsafe {
            write_f16_region(&arena, &a_region, &a_f16);
            write_f16_region(&arena, &b_region, &b_f16);
            write_f16_region(&arena, &q_gamma_region, &q_gamma_f16);
            write_f16_region(&arena, &k_gamma_region, &k_gamma_f16);
            write_f16_region(&arena, &q_region, &vec![f16::NAN; (M * Q_DIM) as usize]);
            write_f16_region(&arena, &k_region, &vec![f16::NAN; (M * KV_DIM) as usize]);
            write_f16_region(&arena, &v_region, &vec![f16::NAN; (M * KV_DIM) as usize]);
            write_f16_region(
                &arena,
                &q_fused_region,
                &vec![f16::NAN; (M * Q_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &k_fused_region,
                &vec![f16::NAN; (M * KV_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &v_fused_region,
                &vec![f16::NAN; (M * KV_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &q_rope_cache_region,
                &vec![f16::NAN; (M * Q_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &k_rope_cache_region,
                &vec![f16::NAN; (M * KV_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &v_rope_cache_region,
                &vec![f16::NAN; (M * KV_DIM) as usize],
            );
            write_f32_region(&arena, &cos_region, &cos);
            write_f32_region(&arena, &sin_region, &sin);
            write_i32_region(&arena, &positions_region, &positions);
            write_i32_region(&arena, &slots_region, &slots);
            write_f16_region(
                &arena,
                &k_cache_region,
                &vec![f16::NAN; (2 * KV_DIM) as usize],
            );
            write_f16_region(
                &arena,
                &v_cache_region,
                &vec![f16::NAN; (2 * KV_DIM) as usize],
            );
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
        {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "qkv_fused_headwise_projection_norm",
                        device: "apple-silicon",
                    },
                )
            })?;
            let q_row_offset = 0u32;
            let k_row_offset = Q_DIM;
            let v_row_offset = Q_DIM + KV_DIM;
            let v_has_gamma = 0u32;
            unsafe {
                encoder.setComputePipelineState(pipelines.get("qkv_headwise_rmsnorm_f16")?);
                encoder.setBuffer_offset_atIndex(Some(buf), a_region.offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), b_region.offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_region.offset, 2);
                encoder.setBuffer_offset_atIndex(Some(buf), k_gamma_region.offset, 3);
                encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_region.offset, 4);
                encoder.setBuffer_offset_atIndex(Some(buf), q_fused_region.offset, 5);
                encoder.setBuffer_offset_atIndex(Some(buf), k_fused_region.offset, 6);
                encoder.setBuffer_offset_atIndex(Some(buf), v_fused_region.offset, 7);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&M as *const _ as *mut _),
                    4,
                    8,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&K as *const _ as *mut _),
                    4,
                    9,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HEAD_DIM as *const _ as *mut _),
                    4,
                    10,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&Q_HEADS as *const _ as *mut _),
                    4,
                    11,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&KV_HEADS as *const _ as *mut _),
                    4,
                    12,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&q_row_offset as *const _ as *mut _),
                    4,
                    13,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&k_row_offset as *const _ as *mut _),
                    4,
                    14,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&v_row_offset as *const _ as *mut _),
                    4,
                    15,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                    4,
                    16,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&v_has_gamma as *const _ as *mut _),
                    4,
                    17,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (M * (Q_HEADS + 2 * KV_HEADS)) as usize,
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
        {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "qkv_fused_headwise_projection_norm_rope_cache",
                        device: "apple-silicon",
                    },
                )
            })?;
            let q_row_offset = 0u32;
            let k_row_offset = Q_DIM;
            let v_row_offset = Q_DIM + KV_DIM;
            let v_has_gamma = 0u32;
            let rope_dim = HEAD_DIM;
            unsafe {
                encoder
                    .setComputePipelineState(pipelines.get("qkv_headwise_rmsnorm_rope_cache_f16")?);
                encoder.setBuffer_offset_atIndex(Some(buf), a_region.offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), b_region.offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_region.offset, 2);
                encoder.setBuffer_offset_atIndex(Some(buf), k_gamma_region.offset, 3);
                encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_region.offset, 4);
                encoder.setBuffer_offset_atIndex(Some(buf), q_rope_cache_region.offset, 5);
                encoder.setBuffer_offset_atIndex(Some(buf), k_rope_cache_region.offset, 6);
                encoder.setBuffer_offset_atIndex(Some(buf), v_rope_cache_region.offset, 7);
                encoder.setBuffer_offset_atIndex(Some(buf), cos_region.offset, 8);
                encoder.setBuffer_offset_atIndex(Some(buf), sin_region.offset, 9);
                encoder.setBuffer_offset_atIndex(Some(buf), positions_region.offset, 10);
                encoder.setBuffer_offset_atIndex(Some(buf), slots_region.offset, 11);
                encoder.setBuffer_offset_atIndex(Some(buf), k_cache_region.offset, 12);
                encoder.setBuffer_offset_atIndex(Some(buf), v_cache_region.offset, 13);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&M as *const _ as *mut _),
                    4,
                    14,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&K as *const _ as *mut _),
                    4,
                    15,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&HEAD_DIM as *const _ as *mut _),
                    4,
                    16,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&Q_HEADS as *const _ as *mut _),
                    4,
                    17,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&KV_HEADS as *const _ as *mut _),
                    4,
                    18,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&q_row_offset as *const _ as *mut _),
                    4,
                    19,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&k_row_offset as *const _ as *mut _),
                    4,
                    20,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&v_row_offset as *const _ as *mut _),
                    4,
                    21,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
                    4,
                    22,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&v_has_gamma as *const _ as *mut _),
                    4,
                    23,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&rope_dim as *const _ as *mut _),
                    4,
                    24,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (M * (Q_HEADS + 2 * KV_HEADS)) as usize,
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
        let apply_rope = |values: &[f32], heads: usize, dim: usize| -> Vec<f32> {
            let mut out = values
                .iter()
                .map(|value| f16::from_f32(*value).to_f32())
                .collect::<Vec<_>>();
            let half_rope = HEAD_DIM as usize / 2;
            for token in 0..M as usize {
                let pos = positions[token] as usize;
                for head in 0..heads {
                    let base = token * dim + head * HEAD_DIM as usize;
                    for pair in 0..half_rope {
                        let x0 = out[base + pair];
                        let x1 = out[base + pair + HEAD_DIM as usize / 2];
                        let cos_val = cos[pos * half_rope + pair];
                        let sin_val = sin[pos * half_rope + pair];
                        out[base + pair] = f16::from_f32(x0 * cos_val - x1 * sin_val).to_f32();
                        out[base + pair + HEAD_DIM as usize / 2] =
                            f16::from_f32(x0 * sin_val + x1 * cos_val).to_f32();
                    }
                }
            }
            out
        };
        let q_rope_expected = apply_rope(&q_expected, Q_HEADS as usize, Q_DIM as usize);
        let k_rope_expected = apply_rope(&k_expected, KV_HEADS as usize, KV_DIM as usize);
        let v_cache_expected = v_expected
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect::<Vec<_>>();
        let mut k_cache_expected = vec![f32::NAN; (2 * KV_DIM) as usize];
        let mut v_cache_expected_slots = vec![f32::NAN; (2 * KV_DIM) as usize];
        for token in 0..M as usize {
            let slot = slots[token] as usize;
            let src = token * KV_DIM as usize;
            let dst = slot * KV_DIM as usize;
            k_cache_expected[dst..dst + KV_DIM as usize]
                .copy_from_slice(&k_rope_expected[src..src + KV_DIM as usize]);
            v_cache_expected_slots[dst..dst + KV_DIM as usize]
                .copy_from_slice(&v_cache_expected[src..src + KV_DIM as usize]);
        }
        for (name, region, expected) in [
            ("q", &q_region, q_expected.clone()),
            ("k", &k_region, k_expected.clone()),
            ("v_unit", &v_region, v_expected.clone()),
            ("q_fused", &q_fused_region, q_expected),
            ("k_fused", &k_fused_region, k_expected),
            ("v_fused", &v_fused_region, v_expected),
            ("q_rope_cache", &q_rope_cache_region, q_rope_expected),
            ("k_rope_cache", &k_rope_cache_region, k_rope_expected),
            ("v_rope_cache", &v_rope_cache_region, v_cache_expected),
            ("k_cache", &k_cache_region, k_cache_expected),
            ("v_cache", &v_cache_region, v_cache_expected_slots),
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
        let layer_scale = (0..HIDDEN as usize)
            .map(|idx| 0.5 + idx as f32 * 0.03125)
            .collect::<Vec<_>>();

        let mut expected_residual = vec![0.0f32; residual.len()];
        let mut expected_normed = vec![0.0f32; residual.len()];
        for token in 0..NUM_TOKENS as usize {
            let base = token * HIDDEN as usize;
            let mut sum = 0.0f32;
            for dim in 0..HIDDEN as usize {
                let updated =
                    f16::from_f32(residual[base + dim] + addition[base + dim] * layer_scale[dim])
                        .to_f32();
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
        let layer_scale_f16 = layer_scale
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
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
        let scale_region = arena.region(
            "residual_add_rmsnorm_layer_scale",
            layer_scale_f16.len() * half_bytes,
            2,
        )?;
        unsafe {
            write_f16_region(&arena, &residual_region, &residual_f16);
            write_f16_region(&arena, &addition_region, &addition_f16);
            write_f16_region(&arena, &gamma_region, &gamma_f16);
            write_f16_region(&arena, &scale_region, &layer_scale_f16);
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
            encoder.setBuffer_offset_atIndex(Some(buf), scale_region.offset, 6);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                4,
                7,
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
    fn residual_add_then_scale_matches_sequential_half_rounded_reference() -> rvllm_core::Result<()>
    {
        let (ctx, pipelines, mut arena) = metal_test_context(16 * 1024)?;
        const HIDDEN: u32 = 8;
        const COUNT: u32 = 16;
        let residual = (0..COUNT as usize)
            .map(|idx| (idx as f32 - 5.0) * 0.25)
            .collect::<Vec<_>>();
        let addition = (0..COUNT as usize)
            .map(|idx| {
                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                sign * (0.5 + (idx % 3) as f32 * 0.125)
            })
            .collect::<Vec<_>>();
        let layer_scale = (0..HIDDEN as usize)
            .map(|idx| 0.5 + idx as f32 * 0.03125)
            .collect::<Vec<_>>();
        let expected = residual
            .iter()
            .zip(addition.iter())
            .enumerate()
            .map(|(idx, (r, a))| {
                let added = f16::from_f32(*r + *a).to_f32();
                f16::from_f32(added * layer_scale[idx % HIDDEN as usize]).to_f32()
            })
            .collect::<Vec<_>>();

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
        let layer_scale_f16 = layer_scale
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let half_bytes = std::mem::size_of::<f16>();
        let residual_region = arena.region(
            "residual_add_then_scale_residual",
            residual_f16.len() * half_bytes,
            2,
        )?;
        let addition_region = arena.region(
            "residual_add_then_scale_addition",
            addition_f16.len() * half_bytes,
            2,
        )?;
        let scale_region = arena.region(
            "residual_add_then_scale_scale",
            layer_scale_f16.len() * half_bytes,
            2,
        )?;
        unsafe {
            write_f16_region(&arena, &residual_region, &residual_f16);
            write_f16_region(&arena, &addition_region, &addition_f16);
            write_f16_region(&arena, &scale_region, &layer_scale_f16);
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "residual_add_then_scale_command_buffer",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "residual_add_then_scale_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encoder.setComputePipelineState(pipelines.get("residual_add_then_scale_f16")?);
            encoder.setBuffer_offset_atIndex(Some(buf), residual_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), addition_region.offset, 1);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&COUNT as *const _ as *mut _),
                4,
                2,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                4,
                3,
            );
            encoder.setBuffer_offset_atIndex(Some(buf), scale_region.offset, 4);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&HIDDEN as *const _ as *mut _),
                4,
                5,
            );
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: COUNT as usize,
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

        let got = unsafe { read_f16_region(&arena, &residual_region, residual.len()) };
        for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() < 0.0001,
                "residual[{idx}]: got={got} expected={expected}"
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
    fn low_bit_projection_abi_stays_a16_in_typed_bf16_library() {
        let source = bfloat_kernel_source(false);
        let start = source.find("kernel void projection_w4a16_f16").unwrap();
        let end = source[start..]
            .find("kernel void experimental_projection_w4abf16_bf16")
            .map(|relative| start + relative)
            .unwrap();
        let low_bit = &source[start..end];
        assert!(low_bit.contains("device const half  *A"));
        assert!(low_bit.contains("device const half  *scales"));
        assert!(low_bit.contains("device half        *C"));
        assert!(!low_bit.contains("bfloat"));
    }

    #[test]
    fn experimental_low_bit_bf16_abi_is_explicit_and_keeps_f16_scales() {
        let source = bfloat_kernel_source(false);
        for name in [
            "experimental_projection_w4abf16_bf16",
            "experimental_projection_w8abf16_bf16",
        ] {
            let start = source
                .find(&format!("kernel void {name}"))
                .expect("experimental BF16 kernel is present");
            let tail = &source[start..];
            let end = tail[1..]
                .find("kernel void ")
                .map_or(tail.len(), |relative| relative + 1);
            let kernel = &tail[..end];
            assert!(kernel.contains("device const bfloat *A"));
            assert!(kernel.contains("device const half   *scales"));
            assert!(kernel.contains("device bfloat       *C"));
            assert!(kernel.contains("float partial = 0.0f"));
            assert!(kernel.contains("float total = simd_sum(partial)"));
        }
    }

    #[test]
    fn kernel_count_matches_names() {
        assert_eq!(KERNEL_COUNT, KERNEL_NAMES.len());
    }

    #[test]
    fn every_paged_attention_kernel_has_the_exact_window_abi() {
        assert_eq!(
            KERNEL_SOURCE
                .matches("constant uint     &attention_window")
                .count(),
            3
        );
        assert!(KERNEL_SOURCE.contains(
            "uint attn_start = attention_window == 0\n        ? 0\n        : uint(ctx_len) - min(uint(ctx_len), attention_window);"
        ));
        assert!(KERNEL_SOURCE.contains(
            "uint attn_start = attention_window == 0\n        ? 0\n        : ctx_len - min(ctx_len, attention_window);"
        ));
        assert!(KERNEL_SOURCE.contains(
            "uint attn_start = attention_window == 0\n        ? 0\n        : attn_len - min(attn_len, attention_window);"
        ));
    }
}
