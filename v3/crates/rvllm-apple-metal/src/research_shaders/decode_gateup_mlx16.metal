// Decode-only Gemma 4 12B fused Gate||Up projection + GELU*mul.
//
// This removes the materialized [1, 2*I] gate_up tensor and the dependent
// gelu_mul dispatch. It intentionally PRESERVES the incumbent storage
// boundaries: gate and up are each rounded to the selected 16-bit storage
// type before GELU/multiply, then the activated result is rounded again.
//
// Geometry follows the dense MLX decode family:
//   4 SIMD groups/TG * 4 intermediate rows/group = 16 activated rows/TG.
//   4 contiguous K values/lane = 128 K values/SIMD iteration.
// No threadgroup scratch or barriers.
kernel void research_decode_gateup_mlx16(
    device const half *A          [[buffer(0)]], // [H]
    device const half *W          [[buffer(1)]], // [2*I,H], gate then up
    device half       *activated  [[buffer(2)]], // [I]
    constant uint     &H          [[buffer(3)]],
    constant uint     &I          [[buffer(4)]],
    uint tg                       [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    const uint n0 = tg * 16u + uint(simdgroup) * 4u;
    if (n0 >= I || simdgroup >= 4u) return;

    float4 gate = float4(0.0f);
    float4 up = float4(0.0f);

    for (uint k0 = uint(lane) * 4u; k0 < H; k0 += 128u) {
        const float x0 = float(A[k0 + 0u]);
        const float x1 = float(A[k0 + 1u]);
        const float x2 = float(A[k0 + 2u]);
        const float x3 = float(A[k0 + 3u]);

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < I) {
                const uint gw = n * H + k0;
                const uint uw = (I + n) * H + k0;
                gate[row] += x0 * float(W[gw + 0u]);
                gate[row] += x1 * float(W[gw + 1u]);
                gate[row] += x2 * float(W[gw + 2u]);
                gate[row] += x3 * float(W[gw + 3u]);
                up[row] += x0 * float(W[uw + 0u]);
                up[row] += x1 * float(W[uw + 1u]);
                up[row] += x2 * float(W[uw + 2u]);
                up[row] += x3 * float(W[uw + 3u]);
            }
        }
    }

    gate.x = simd_sum(gate.x);
    gate.y = simd_sum(gate.y);
    gate.z = simd_sum(gate.z);
    gate.w = simd_sum(gate.w);
    up.x = simd_sum(up.x);
    up.y = simd_sum(up.y);
    up.z = simd_sum(up.z);
    up.w = simd_sum(up.w);

    if (lane == 0u) {
#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < I) {
                // Match gemm(alpha=1,beta=0) -> 16-bit scratch -> gelu_mul.
                half rounded_gate = f16_sat(gate[row] + 0.0f);
                half rounded_up = f16_sat(up[row] + 0.0f);
                activated[n] =
                    f16_sat(gelu_tanh(float(rounded_gate)) * float(rounded_up));
            }
        }
    }
}
