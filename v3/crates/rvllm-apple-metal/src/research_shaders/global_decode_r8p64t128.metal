// Exact unsplit specialization: packed rows 8, D panel 64, 128 threads.
kernel void research_global_d512_r8p64t128(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device uchar *output [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup ushort qt[8 * 512];
    threadgroup ushort stage[8 * 64];
    threadgroup float scores[8 * 8], alpha[8 * 8], weight[8 * 8];
    threadgroup int pages[8];
    global_decode_body<8, 64, 128>(q, k, v, output, table, contexts, positions, p,
        group, tid, sg, lane, threads, qt, stage, scores, alpha, weight, pages);
}
