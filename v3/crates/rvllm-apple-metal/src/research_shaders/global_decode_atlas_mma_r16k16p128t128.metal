kernel void research_global_d512_atlas_mma_r16k16p128t128(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device uchar *output [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float qt[16 * 128], stage[16 * 128], scores[16 * 16], weights[16 * 16], state[3 * 16];
    threadgroup int pages[16];
    global_decode_matrix_body<16, 16, 128, 128>(q, k, v, output, table, contexts, positions, p,
        group, tid, sg, lane, threads, qt, stage, scores, weights, state, pages);
}
