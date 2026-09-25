// Headwise SIMD control: one D512 query head, D panel 128, 32 threads.
// This deliberately rereads the shared K/V stream for every query head. It is
// a sane occupancy control for the scalar incumbent, not the final GQA16 path.
kernel void research_global_d512_r1p128t32(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device uchar *output [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup ushort qt[1 * 512];
    threadgroup ushort stage[8 * 128];
    threadgroup float scores[1 * 8], alpha[1 * 8], weight[1 * 8];
    threadgroup int pages[8];
    global_decode_body<1, 128, 32>(q, k, v, output, table, contexts, positions, p,
        group, tid, sg, lane, threads, qt, stage, scores, alpha, weight, pages);
}
