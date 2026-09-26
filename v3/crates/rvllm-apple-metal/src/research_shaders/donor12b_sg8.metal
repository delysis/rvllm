// Exactly one prepared donor family. Explicit BF16 I/O and FP16 scales.

kernel void research_donor12b_sg8_w4(
    device const ushort *x [[buffer(0)]], device const uchar *w [[buffer(1)]], device const half *sc [[buffer(2)]], device uchar *out [[buffer(3)]], constant Donor12bParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_qmv<true,8>(x,w,sc,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_w8(
    device const ushort *x [[buffer(0)]], device const uchar *w [[buffer(1)]], device const half *sc [[buffer(2)]], device uchar *out [[buffer(3)]], constant Donor12bParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_qmv<false,8>(x,w,sc,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_batch_w4(
    device const ushort *x [[buffer(0)]], device const uchar *w [[buffer(1)]], device const half *sc [[buffer(2)]], device uchar *out [[buffer(3)]], constant Donor12bParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_batch8<true,8>(x,w,sc,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_batch_w8(
    device const ushort *x [[buffer(0)]], device const uchar *w [[buffer(1)]], device const half *sc [[buffer(2)]], device uchar *out [[buffer(3)]], constant Donor12bParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_batch8<false,8>(x,w,sc,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_gate_w4(
    device const ushort *x [[buffer(0)]], device const uchar *wg [[buffer(1)]], device const half *sgate [[buffer(2)]], device const uchar *wu [[buffer(3)]], device const half *sup [[buffer(4)]], device ushort *out [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_gate<true,8>(x,wg,sgate,wu,sup,out,group,sg,lane);
}

kernel void research_donor12b_sg8_gate_w8(
    device const ushort *x [[buffer(0)]], device const uchar *wg [[buffer(1)]], device const half *sgate [[buffer(2)]], device const uchar *wu [[buffer(3)]], device const half *sup [[buffer(4)]], device ushort *out [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_gate<false,8>(x,wg,sgate,wu,sup,out,group,sg,lane);
}

kernel void research_donor12b_sg8_qkv_w4(
    device const ushort *x [[buffer(0)]], device const uchar *wq [[buffer(1)]], device const half *sq [[buffer(2)]], device const uchar *wk [[buffer(3)]], device const half *sk [[buffer(4)]], device const uchar *wv [[buffer(5)]], device const half *sv [[buffer(6)]], device ushort *out [[buffer(7)]], constant Donor12bParams &p [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_qkv<true,8>(x,wq,sq,wk,sk,wv,sv,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_qkv_w8(
    device const ushort *x [[buffer(0)]], device const uchar *wq [[buffer(1)]], device const half *sq [[buffer(2)]], device const uchar *wk [[buffer(3)]], device const half *sk [[buffer(4)]], device const uchar *wv [[buffer(5)]], device const half *sv [[buffer(6)]], device ushort *out [[buffer(7)]], constant Donor12bParams &p [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_qkv<false,8>(x,wq,sq,wk,sk,wv,sv,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_native_gate(
    device const ushort *x [[buffer(0)]], device const ushort *w [[buffer(1)]], device ushort *out [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    d12_native_gate<8>(x,w,out,group,sg,lane);
}

kernel void research_donor12b_sg8_native_projection(
    device const ushort *x [[buffer(0)]], device const ushort *w [[buffer(1)]], device uchar *out [[buffer(2)]], constant Donor12bParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    if(p.m==1) d12_native_projection<8,1,4>(x,w,out,p,group,sg,lane);
    else d12_native_projection<8,8,2>(x,w,out,p,group,sg,lane);
}

kernel void research_donor12b_sg8_local_attention(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]], device const ushort *v [[buffer(2)]], device ushort *out [[buffer(3)]], device const int *table [[buffer(4)]], device const int *length [[buffer(5)]], device const int *position [[buffer(6)]], constant Donor12bAttentionParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=512 || threads.y!=1 || threads.z!=1) return;
    threadgroup float partials[4128];
    d12_attention<256,8,16>(q,k,v,out,table,length,position,p,partials,group,sg,lane);
}

kernel void research_donor12b_sg8_global_attention(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]], device const ushort *v [[buffer(2)]], device ushort *out [[buffer(3)]], device const int *table [[buffer(4)]], device const int *length [[buffer(5)]], device const int *position [[buffer(6)]], constant Donor12bAttentionParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if(threads.x!=256 || threads.y!=1 || threads.z!=1) return;
    threadgroup float partials[4112];
    d12_attention<512,1,8>(q,k,v,out,table,length,position,p,partials,group,sg,lane);
}
