kernel void research_global_d512_split_mma_r8k32s256t128_partial(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device float *partials [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float qt[8 * 64], stage[32 * 64], scores[8 * 32], weights[8 * 32], state[3 * 8];
    threadgroup int pages[32];
    global_decode_split_matrix_partial_body(q, k, v, partials, table, contexts, positions, p,
        group, tid, sg, lane, threads, qt, stage, scores, weights, state, pages);
}

kernel void research_global_d512_split_mma_r8k32s256t128_merge(
    device const float *partials [[buffer(0)]], device uchar *output [[buffer(1)]],
    device const int *table [[buffer(2)]], device const int *contexts [[buffer(3)]],
    device const int *positions [[buffer(4)]], constant GlobalDecodeParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    global_decode_split_merge_body(partials, output, table, contexts, positions, p,
        group.x, lane, threads);
}
