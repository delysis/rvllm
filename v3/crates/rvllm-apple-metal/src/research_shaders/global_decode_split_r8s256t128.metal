kernel void research_global_d512_split_r8s256t128_partial(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device float *partials [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup ushort qt[8 * 512];
    threadgroup ushort stage[8 * 64];
    threadgroup float scores[8 * 8], alpha[8 * 8], weight[8 * 8];
    threadgroup int pages[8];
    global_decode_split_partial_body<8, 256, 64, 128>(q, k, v, partials, table,
        contexts, positions, p, group, tid, sg, lane, threads, qt, stage, scores,
        alpha, weight, pages);
}

kernel void research_global_d512_split_r8s256t128_merge(
    device const float *partials [[buffer(0)]], device uchar *output [[buffer(1)]],
    device const int *table [[buffer(2)]], device const int *contexts [[buffer(3)]],
    device const int *positions [[buffer(4)]], constant GlobalDecodeParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    global_decode_split_merge_body(partials, output, table, contexts, positions, p,
        group.x, lane, threads);
}
