// Same cooperating schedule as short-r4t128; new host AND shader capacity.
kernel void research_global_d512_stream_r4t128_c2048(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device uchar *out [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup ushort kt[512], vt[512];
    global_stream_r4<false>(q,k,v,out,table,contexts,positions,p,group,tid,sg,lane,threads,kt,vt);
}
