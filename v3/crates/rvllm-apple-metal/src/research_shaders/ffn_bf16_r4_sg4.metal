// 4 gate and 4 up FP32 accumulators/lane, 16 output rows/TG.
kernel void research_ffn_bf16_r4_sg4(
    device const ushort *x [[buffer(0)]], device const ushort *w [[buffer(1)]],
    device ushort *activated [[buffer(2)]],
    constant uint &m [[buffer(3)]], constant uint &hidden [[buffer(4)]],
    constant uint &intermediate [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    round2_ffn_bf16<4,4>(x,w,activated,m,hidden,intermediate,group,sg,lane,threads);
}
