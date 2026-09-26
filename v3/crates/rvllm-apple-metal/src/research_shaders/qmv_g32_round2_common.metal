// Exact-shape row/SIMDgroup scheduling ablations. Preserve authenticated
// signed storage, row-major group-32 FP16 scales, scale-before-FMA, BF16 A/Y.
// Neither scale hoisting nor unpack/ISA behavior is inferred from this source.
template<bool W4, uint R, uint SG, uint K>
inline void round2_qmv_g32(
    device const ushort *a, device const uchar *w, device const half *scales,
    device ushort *out, uint m, uint n, uint k, uint stride, uint column,
    uint3 group, ushort sg, ushort lane, uint3 threads) {
    if (any(threads != uint3(32u*SG,1,1)) || m != 1u || n != 3840u
        || k != K
        || stride < n || column > stride - n
        || group.y != 0u || group.z != 0u || group.x >= 3840u/(R*SG)) return;
    uint row0 = group.x * (R*SG) + uint(sg) * R;
    uint groups = k / 32u;
    float accum[R] = {};
    // A lane owns the same K coordinate in each 32-element quantization group.
    // Per-row scale is applied BEFORE the FP32 FMA, exactly as the sidecar
    // dequantizer; it is not moved across a reduction or replaced with BF16.
    for (uint g = 0; g < groups; ++g) {
        uint d = g * 32u + uint(lane);
        float av = round_decode_bf16(a[d]);
        #pragma unroll
        for (uint r = 0; r < R; ++r) {
            uint row = row0 + r;
            int q;
            if (W4) {
                uchar packed = w[size_t(row)*(k/2u)+d/2u];
                q = int((d & 1u) == 0u ? packed & 15u : packed >> 4u);
                q = q >= 8 ? q - 16 : q;
            } else q = int(as_type<char>(w[size_t(row)*k+d]));
            float scale = float(scales[size_t(row)*groups+g]);
            accum[r] = fma(av, float(q) * scale, accum[r]);
        }
    }
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
        float total = round_decode_sum32(accum[r]);
        if (lane == 0) out[column+row0+r] = round_decode_rne(total);
    }
}
