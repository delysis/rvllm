// Schedule port only: signed row-major nibbles/bytes and native FP16 group-32
// scales are the rvLLM authenticated sidecar ABI. NO biases, repacker, G64, or
// dense materialization. Two SIMDgroups each own eight rows (16 rows/TG).
template<bool W4>
inline void round_qmv_g32_r8_sg2(
    device const ushort *a, device const uchar *w, device const half *scales,
    device ushort *out, uint m, uint n, uint k, uint stride, uint column,
    uint3 group, ushort sg, ushort lane, uint3 threads) {
    if (any(threads != uint3(64,1,1)) || m != 1u || n != 3840u
        || (W4 ? k != 15360u : (k != 4096u && k != 8192u))
        || stride < n || column > stride - n
        || group.y != 0u || group.z != 0u || group.x >= 240u) return;
    uint row0 = group.x * 16u + uint(sg) * 8u;
    uint groups = k / 32u;
    float accum[8] = {0,0,0,0,0,0,0,0};
    // A lane owns the same K coordinate in each 32-element quantization group.
    // Per-row scale is applied BEFORE the FP32 FMA, exactly as the sidecar
    // dequantizer; it is not moved across a reduction or replaced with BF16.
    for (uint g = 0; g < groups; ++g) {
        uint d = g * 32u + uint(lane);
        float av = round_decode_bf16(a[d]);
        #pragma unroll
        for (uint r = 0; r < 8u; ++r) {
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
    for (uint r = 0; r < 8u; ++r) {
        float total = round_decode_sum32(accum[r]);
        if (lane == 0) out[column+row0+r] = round_decode_rne(total);
    }
}
