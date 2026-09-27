// Eight fixed 256-token partitions, bounded capacity 2048. The host adapter
// encodes partial then merge, with complete work timed in one command buffer.
kernel void research_global_d512_split_stream_r4s256t128_c2048_partial(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device float *partials [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup ushort kt[512], vt[512];
    global_stream_r4<true>(q,k,v,reinterpret_cast<device uchar *>(partials),table,
        contexts,positions,p,group,tid,sg,lane,threads,kt,vt);
}

kernel void research_global_d512_split_stream_r4s256t128_c2048_merge(
    device const float *partials [[buffer(0)]], device uchar *output [[buffer(1)]],
    device const int *table [[buffer(2)]], device const int *contexts [[buffer(3)]],
    device const int *positions [[buffer(4)]], constant GlobalDecodeParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    // The reused merge has a 4096 bound. Refuse our tighter 2048 contract
    // here, including malformed dispatch y/z, before it can touch output.
    if (group.y != 0u || group.z != 0u
        || global_stream_visible_end(p,table,contexts,positions) == 0u) return;
    global_decode_split_merge_body<8>(partials, output, table, contexts, positions, p,
        group.x, lane, threads);
}
