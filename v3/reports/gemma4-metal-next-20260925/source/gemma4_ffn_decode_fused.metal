#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Decode-only fused gate+up projection + GELU multiply for Gemma 4 dense FFN.
// Exact initial target: hidden K=3840, intermediate I=15360, M=1.
// Q4 affine group-64 storage is the same as gemma4_qmv_g64.metal.
// Weight rows are stacked [gate(0..I), up(I..2I)].
// One SIMD group computes 4 intermediate outputs (8 projection rows), and two
// SIMD groups / TG therefore materialize 8 activated FFN values. This removes:
//   * a second read of x for gate/up,
//   * the 2*I BF16 intermediate write,
//   * the 2*I BF16 intermediate read by a separate GELU*mul kernel,
//   * one command dispatch.

static inline float bf16_to_f32(ushort x) {
    return as_type<float>(uint(x) << 16);
}
static inline ushort f32_to_bf16_rne(float x) {
    uint u = as_type<uint>(x);
    uint lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return ushort(u >> 16);
}
static inline float gelu_tanh_f32(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

kernel void gemma4_ffn_gateup_gelu_q4_g64_r4_sg2(
    device const ushort *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device ushort *activated [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &intermediate [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (threads != 64) return;
    if (hidden != 3840u || intermediate != 15360u) return;
    if ((hidden & 63u) != 0) return;

    constexpr uint ROWS = 4;
    const uint out_base = tg * 8u + uint(sg) * ROWS;
    if (out_base >= intermediate) return;

    const uint groups = hidden >> 6;
    const uint packed_stride = hidden >> 1;
    float gate_acc[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[ROWS]   = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint g = 0; g < groups; ++g) {
        const uint k0 = (g << 6) + (uint(lane) << 1);
        const float x0 = bf16_to_f32(x[k0]);
        const float x1 = bf16_to_f32(x[k0 + 1]);
        const float sx = simd_sum(x0 + x1);

        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint out = out_base + r;
            if (out < intermediate) {
                const uint gate_row = out;
                const uint up_row = intermediate + out;
                const ulong go = ulong(gate_row) * packed_stride + ulong(g) * 32ul + lane;
                const ulong uo = ulong(up_row)   * packed_stride + ulong(g) * 32ul + lane;
                const uchar gp = w[go];
                const uchar up = w[uo];
                const float gd = simd_sum(float(gp & 15u) * x0 + float(gp >> 4) * x1);
                const float ud = simd_sum(float(up & 15u) * x0 + float(up >> 4) * x1);
                if (lane == 0) {
                    const ulong gm = ulong(gate_row) * groups + g;
                    const ulong um = ulong(up_row) * groups + g;
                    gate_acc[r] += bf16_to_f32(scales[gm]) * gd + bf16_to_f32(biases[gm]) * sx;
                    up_acc[r]   += bf16_to_f32(scales[um]) * ud + bf16_to_f32(biases[um]) * sx;
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint out = out_base + r;
            if (out < intermediate) {
                activated[out] = f32_to_bf16_rne(gelu_tanh_f32(gate_acc[r]) * up_acc[r]);
            }
        }
    }
}

// Dense BF16 reference candidate for the same fusion. It is not expected to
// beat Q4 on bandwidth, but isolates the value of fusion from quantization.
kernel void gemma4_ffn_gateup_gelu_bf16_r4_sg2(
    device const ushort *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]], // [2I, K]
    device ushort *activated [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &intermediate [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (threads != 64) return;
    if (hidden != 3840u || intermediate != 15360u) return;

    constexpr uint ROWS = 4;
    const uint out_base = tg * 8u + uint(sg) * ROWS;
    if (out_base >= intermediate) return;
    float gate_acc[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[ROWS]   = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = uint(lane); k < hidden; k += 32u) {
        const float xv = bf16_to_f32(x[k]);
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint out = out_base + r;
            if (out < intermediate) {
                gate_acc[r] += xv * bf16_to_f32(w[ulong(out) * hidden + k]);
                up_acc[r]   += xv * bf16_to_f32(w[ulong(intermediate + out) * hidden + k]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) {
        gate_acc[r] = simd_sum(gate_acc[r]);
        up_acc[r] = simd_sum(up_acc[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint out = out_base + r;
            if (out < intermediate) {
                activated[out] = f32_to_bf16_rne(gelu_tanh_f32(gate_acc[r]) * up_acc[r]);
            }
        }
    }
}
