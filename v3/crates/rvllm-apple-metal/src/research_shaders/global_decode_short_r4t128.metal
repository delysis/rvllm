// rvLLM paged-cache ABI, NOT the adjudicated contiguous prototype ABI.
// Producer ownership/order is unchanged: transformed newest K/V are committed
// by the preceding cache-write encoder (or the shared-KV owning layer).
// Read-only pages, holes and position-bounded visibility also cover restored
// prefixes and rollback. No persistent state, merge, or global scratch.
kernel void research_global_d512_short_r4t128(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device uchar *out [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (any(threads != uint3(128,1,1)) || group.x >= 4u || group.y != 0u || group.z != 0u
        || p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u || p.head_dim != 512u
        || p.window != 0u || p.scale != 1.0f || p.output_kind > 1u
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || size_t(p.max_blocks) * p.block_size > 512u) return;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return;
    uint end = uint(position) + 1u;
    // Uniform validation BEFORE output writes; corrupt future suffix ignored.
    for (uint b = 0; b <= (end-1u)/p.block_size; ++b)
        if (table[b] >= 0 && uint(table[b]) >= p.num_blocks) return;
    uint head = group.x*4u + uint(sg);
    threadgroup ushort kt[512], vt[512];
    float qr[16], acc[16];
    for (uint s = 0; s < 16u; ++s) {
        qr[s] = global_decode_widen(q[head*512u+uint(lane)+s*32u]);
        acc[s] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint t = 0; t < end; ++t) {
        int page = table[t/p.block_size];
        if (page < 0) continue; // uniform across ALL SIMDgroups
        size_t base = (size_t(page)*p.block_size + t%p.block_size)*512u;
        for (uint d = tid; d < 512u; d += 128u) { kt[d] = k[base+d]; vt[d] = v[base+d]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Retain the established fixed-64 FP32 QK tree, despite full-D staging.
        float score = 0.0f;
        for (uint panel = 0; panel < 8u; ++panel) {
            float part = fma(qr[panel*2u], global_decode_widen(kt[uint(lane)+panel*64u]), 0.0f);
            part = fma(qr[panel*2u+1u], global_decode_widen(kt[uint(lane)+panel*64u+32u]), part);
            score += simd_broadcast(global_decode_sum32(part), 0u);
        }
        float next = max(maximum, score);
        float a = denominator == 0.0f ? 0.0f : (maximum == next ? 1.0f : precise::exp(maximum-next));
        float b = score == next ? 1.0f : precise::exp(score-next);
        denominator = fma(denominator, a, b);
        for (uint s = 0; s < 16u; ++s)
            acc[s] = fma(b, global_decode_widen(vt[uint(lane)+s*32u]), acc[s]*a);
        maximum = next;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = denominator > 0.0f ? 1.0f/denominator : 0.0f;
    for (uint s = 0; s < 16u; ++s) {
        uint i = head*512u+uint(lane)+s*32u;
        float result = acc[s]*inverse;
        if (p.output_kind == 1u) reinterpret_cast<device float *>(out)[i] = result;
        else reinterpret_cast<device ushort *>(out)[i] = global_decode_round(result);
    }
}
