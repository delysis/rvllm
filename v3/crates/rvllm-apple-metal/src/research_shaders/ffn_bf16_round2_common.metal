// Scheduling ablations only. Same native Gate||Up layout, K traversal, fixed
// SIMD32 reduction, BF16 intermediate boundary and GELU as the measured r4/sg2.
// Source accumulator counts are NOT compiler register counts.
template<uint R, uint SG>
inline void round2_ffn_bf16(
    device const ushort *x, device const ushort *w, device ushort *activated,
    uint m, uint hidden, uint intermediate, uint3 group, ushort sg, ushort lane,
    uint3 threads) {
    constexpr uint ROWS = R * SG;
    if (any(threads != uint3(32u*SG,1,1)) || m != 1u || hidden != 3840u
        || intermediate != 15360u || group.y != 0u || group.z != 0u
        || group.x >= 15360u/ROWS) return;
    uint row = group.x * ROWS + uint(sg) * R;
    float g[R] = {}, u[R] = {};
    for (uint k = uint(lane); k < 3840u; k += 32u) {
        float a = round_decode_bf16(x[k]);
        #pragma unroll
        for (uint r = 0; r < R; ++r) {
            g[r] = fma(a, round_decode_bf16(w[size_t(row+r)*3840u+k]), g[r]);
            u[r] = fma(a, round_decode_bf16(w[size_t(15360u+row+r)*3840u+k]), u[r]);
        }
    }
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
        float gate = round_decode_bf16(round_decode_rne(round_decode_sum32(g[r])));
        float up = round_decode_bf16(round_decode_rne(round_decode_sum32(u[r])));
        if (lane == 0) activated[row+r] = round_decode_rne(round_decode_gelu(gate) * up);
    }
}
