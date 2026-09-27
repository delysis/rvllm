// Shared implementation for the bounded cooperating and split-stream arms.
// K/V are READ ONLY. The preceding owner cache-write encoder remains responsible
// for newest-token publication; no cross-threadgroup cache-write race is added.
inline uint global_stream_visible_end(
    constant GlobalDecodeParams &p, device const int *table,
    device const int *contexts, device const int *positions) {
    if (p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u || p.head_dim != 512u
        || p.window != 0u || p.scale != 1.0f || p.output_kind > 1u
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || size_t(p.max_blocks) * p.block_size > 2048u) return 0u;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return 0u;
    uint end = uint(position) + 1u;
    // Validate only visible metadata. Negative entries are holes; an invalid
    // positive entry vetoes the whole launch before any output/scratch write.
    for (uint b = 0; b <= (end-1u)/p.block_size; ++b)
        if (table[b] >= 0 && uint(table[b]) >= p.num_blocks) return 0u;
    return end;
}

template<bool SPLIT>
inline void global_stream_r4(
    device const ushort *q, device const ushort *k, device const ushort *v,
    device uchar *out, device const int *table, device const int *contexts,
    device const int *positions, constant GlobalDecodeParams &p,
    uint3 group, uint tid, ushort sg, ushort lane, uint3 threads,
    threadgroup ushort *kt, threadgroup ushort *vt) {
    if (any(threads != uint3(128,1,1)) || group.x >= 4u
        || group.y >= (SPLIT ? 8u : 1u) || group.z != 0u) return;
    uint end = global_stream_visible_end(p,table,contexts,positions);
    if (end == 0u) return;
    uint first = SPLIT ? group.y * 256u : 0u;
    uint stop = SPLIT ? min(end, first+256u) : end;
    uint head = group.x*4u + uint(sg);
    float qr[16], acc[16];
    for (uint s = 0; s < 16u; ++s) {
        qr[s] = global_decode_widen(q[head*512u+uint(lane)+s*32u]);
        acc[s] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint t = first; t < stop; ++t) {
        int page = table[t/p.block_size];
        if (page < 0) continue; // uniform across every thread in this TG
        size_t base = (size_t(page)*p.block_size + t%p.block_size)*512u;
        for (uint d = tid; d < 512u; d += 128u) { kt[d] = k[base+d]; vt[d] = v[base+d]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    if (SPLIT) {
        // Every valid partition, including wholly future/all-hole partitions,
        // overwrites its entire 514-float record. No previous-use state leaks.
        device float *partials = reinterpret_cast<device float *>(out);
        size_t base = (size_t(group.y)*16u + head)*514u;
        for (uint s = 0; s < 16u; ++s) partials[base+uint(lane)+s*32u] = acc[s];
        if (lane == 0u) { partials[base+512u] = maximum; partials[base+513u] = denominator; }
    } else {
        float inverse = denominator > 0.0f ? 1.0f/denominator : 0.0f;
        for (uint s = 0; s < 16u; ++s) {
            uint i = head*512u+uint(lane)+s*32u;
            float result = acc[s]*inverse;
            if (p.output_kind == 1u) reinterpret_cast<device float *>(out)[i] = result;
            else reinterpret_cast<device ushort *>(out)[i] = global_decode_round(result);
        }
    }
}
