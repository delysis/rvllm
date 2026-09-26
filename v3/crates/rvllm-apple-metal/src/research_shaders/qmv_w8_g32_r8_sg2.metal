kernel void research_qmv_w8_g32_r8_sg2(
    device const ushort *a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const half *scales [[buffer(2)]], device ushort *out [[buffer(3)]],
    constant uint &m [[buffer(4)]], constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]], constant uint &stride [[buffer(7)]],
    constant uint &column [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    round_qmv_g32_r8_sg2<false>(a,w,scales,out,m,n,k,stride,column,group,sg,lane,threads);
}
