// Donor-inspired schedule on rvLLM's authenticated signed W8/G32 ABI.
// Eight adjacent K values share one FP16 scale; four output rows share the
// activation load in each SIMDgroup. No affine conversion or requantization.
#pragma METAL fp math_mode(safe)
kernel void research_qmv_w8_g32_r4_sg8_k8(
    device const ushort *a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const half *scales [[buffer(2)]], device ushort *out [[buffer(3)]],
    constant uint &m [[buffer(4)]], constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]], constant uint &stride [[buffer(7)]],
    constant uint &column [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if (any(threads != uint3(256,1,1)) || m != 1u || n != 3840u
        || (k != 4096u && k != 8192u) || stride < n || column > stride - n
        || group.y != 0u || group.z != 0u || group.x >= 120u) return;
    uint row0 = group.x * 32u + uint(sg) * 4u;
    uint groups = k / 32u;
    uint words_per_row = k / 8u;
    device const uint2 *words = reinterpret_cast<device const uint2 *>(w);
    float accum[4] = {0,0,0,0};
    for (uint kb = 0; kb < k; kb += 256u) {
        uint k0 = kb + uint(lane) * 8u;
        float av[8];
        #pragma unroll
        for (uint j = 0; j < 8u; ++j) av[j] = round_decode_bf16(a[k0+j]);
        uint g = k0 / 32u;
        uint word = k0 / 8u;
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            uint row = row0 + r;
            uint2 packed = words[size_t(row)*words_per_row+word];
            float scale = float(scales[size_t(row)*groups+g]);
            float part = 0.0f;
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                uint bits = j < 4u ? packed.x : packed.y;
                uint shift = (j & 3u) * 8u;
                int q = int((bits >> shift) & 255u);
                q = q >= 128 ? q - 256 : q;
                part = fma(av[j], float(q)*scale, part);
            }
            accum[r] += part;
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        float total = round_decode_sum32(accum[r]);
        if (lane == 0) out[column+row0+r] = round_decode_rne(total);
    }
}
