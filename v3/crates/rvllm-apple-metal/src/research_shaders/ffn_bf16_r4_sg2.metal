// Decode only. The existing arena stores Gate||Up as [2*15360,3840].
// Preserve the materialized BF16 gate/up rounding boundary BEFORE GELU; this
// fuses storage/dispatch, not checkpoint precision or the activation function.
kernel void research_ffn_bf16_r4_sg2(
    device const ushort *x [[buffer(0)]], device const ushort *w [[buffer(1)]],
    device ushort *activated [[buffer(2)]],
    constant uint &m [[buffer(3)]], constant uint &hidden [[buffer(4)]],
    constant uint &intermediate [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (any(threads != uint3(64,1,1)) || m != 1u || hidden != 3840u
        || intermediate != 15360u || group.y != 0u || group.z != 0u
        || group.x >= 1920u) return;
    uint row = group.x * 8u + uint(sg) * 4u;
    float g[4] = {0,0,0,0}, u[4] = {0,0,0,0};
    for (uint k = uint(lane); k < 3840u; k += 32u) {
        float a = round_decode_bf16(x[k]);
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            g[r] = fma(a, round_decode_bf16(w[size_t(row+r)*3840u+k]), g[r]);
            u[r] = fma(a, round_decode_bf16(w[size_t(15360u+row+r)*3840u+k]), u[r]);
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        float gate = round_decode_bf16(round_decode_rne(round_decode_sum32(g[r])));
        float up = round_decode_bf16(round_decode_rne(round_decode_sum32(u[r])));
        if (lane == 0) activated[row+r] = round_decode_rne(round_decode_gelu(gate) * up);
    }
}
