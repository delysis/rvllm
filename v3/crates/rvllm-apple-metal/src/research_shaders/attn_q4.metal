// metal-attn-q4: four consecutive query positions share K/V staging.
// This is not the head-group reuse in metal-gqa-kv8. Probability and output
// accumulation remain FP32; keys are visited in ascending logical order.
template <uint D, uint ROWS>
static inline void research_attn_q4_body(
    device const half *q, device const half *kc, device const half *vc, device half *out,
    device const int *blocks, device const int *contexts, device const int *cu,
    device const int *positions, uint total_q, uint batch, uint heads, uint kv_heads,
    uint head_dim, uint block_size, uint max_blocks, float scale, uint window, uint num_blocks,
    uint3 group, uint tid, ushort sg, ushort lane, uint3 threads,
    threadgroup half *kt, threadgroup half *vt, threadgroup uint *valid) {
    // All early returns are threadgroup-uniform and precede every barrier.
    if (threads.x != 128 || threads.y != 1 || threads.z != 1 || batch != 1 ||
        heads != 16 || head_dim != D || block_size == 0 || max_blocks == 0 ||
        num_blocks == 0 || total_q < 64 || total_q > 1024 || group.y >= heads ||
        group.x >= (total_q + 3) / 4 || group.z != 0 ||
        scale != 1.0f) return;
    if (!((D == 256 && kv_heads == 8 && window == 1024) ||
          (D == 512 && kv_heads == 1 && window == 0))) return;
    const uint query = group.x * 4 + sg;
    const bool active = query < total_q;
    const uint kv_head = group.y / (heads / kv_heads);
    const ulong logical_capacity = ulong(max_blocks) * block_size;
    const int context = contexts[0];
    bool invalid = cu[0] != 0 || cu[1] != int(total_q) || context <= 0 || context > 1024 ||
                   ulong(max(context, 0)) > logical_capacity;
    uint first_key = uint(max(context, 0)), last_key = 0;
    // Read the four positions in every lane to derive uniform staging bounds.
    // Only active queries are read. Tail SIMD groups still reach all barriers.
    for (uint j = 0; j < 4; ++j) {
        const uint row = group.x * 4 + j;
        if (row < total_q) {
            const int p = positions[row];
            invalid = invalid || p < 0 || p >= context;
            const uint end = (p >= 0 && p < context) ? uint(p) + 1 : 0;
            const uint start = window == 0 ? 0 : end - min(end, window);
            first_key = min(first_key, start);
            last_key = max(last_key, end);
        }
    }
    if (invalid) { first_key = 0; last_key = 0; }
    const int position = active ? positions[query] : -1;
    const uint end = (position >= 0 && position < context) ? uint(position) + 1 : 0;
    const uint start = window == 0 ? 0 : end - min(end, window);
    float query_lane[D / 32], result[D / 32];
    for (uint i = 0; i < D / 32; ++i) {
        query_lane[i] = active ? float(q[(ulong(query) * heads + group.y) * D + lane + 32 * i]) : 0.0f;
        result[i] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    bool nonfinite_query = false;
    for (uint i = 0; i < D / 32; ++i) nonfinite_query |= !isfinite(query_lane[i]);
    // A hole-only cache must not hide a malformed query. All 32 lanes call it.
    bool poisoned = invalid || simd_any(nonfinite_query);
    for (uint first = first_key; first < last_key; first += ROWS) {
        if (tid < ROWS) {
            const uint t = first + tid;
            const int page = t < last_key ? blocks[t / block_size] : -1;
            valid[tid] = page < 0 ? 0 : (uint(page) < num_blocks ? 1 : 2);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < ROWS * D; index += 128) {
            const uint row = index / D, d = index % D, t = first + row;
            half k = half(0), v = half(0);
            if (valid[row] == 1) {
                const uint page = uint(blocks[t / block_size]);
                const ulong address = ((ulong(page) * block_size + t % block_size) * kv_heads + kv_head) * D + d;
                k = kc[address];
                v = vc[address];
            }
            kt[index] = k;
            vt[index] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = 0; row < ROWS; ++row) {
            const uint t = first + row;
            const bool visible = active && t >= start && t < end;
            if (visible && valid[row] == 2) poisoned = true;
            if (visible && valid[row] == 1) {
                float dot = 0.0f;
                for (uint i = 0; i < D / 32; ++i)
                    dot += query_lane[i] * float(kt[row * D + lane + 32 * i]);
                const float score = simd_sum(dot) * scale;
                poisoned = poisoned || !isfinite(score);
                const float next_maximum = max(maximum, score);
                const float prior_weight = isfinite(maximum) ? exp(maximum - next_maximum) : 0.0f;
                const float weight = exp(score - next_maximum);
                denominator = denominator * prior_weight + weight;
                for (uint i = 0; i < D / 32; ++i)
                    result[i] = result[i] * prior_weight + weight * float(vt[row * D + lane + 32 * i]);
                maximum = next_maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    if (active) {
        for (uint i = 0; i < D / 32; ++i) {
            const ulong address = (ulong(query) * heads + group.y) * D + lane + 32 * i;
            // Never pass the malformed-metadata sentinel through saturation.
            const float value = result[i] * inverse;
            out[address] = poisoned || !isfinite(value) ? half(NAN) : f16_sat(value);
        }
    }
}

kernel void research_attn_q4_d256(
    device const half *q [[buffer(0)]], device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]], device half *out [[buffer(3)]],
    device const int *blocks [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]], constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup half kt[16 * 256], vt[16 * 256];
    threadgroup uint valid[16];
    research_attn_q4_body<256, 16>(q, kc, vc, out, blocks, contexts, cu, positions,
        total_q, batch, heads, kv_heads, head_dim, block_size, max_blocks, scale, window,
        num_blocks, group, tid, sg, lane, threads, kt, vt, valid);
}

kernel void research_attn_q4_d512(
    device const half *q [[buffer(0)]], device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]], device half *out [[buffer(3)]],
    device const int *blocks [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]], constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup half kt[8 * 512], vt[8 * 512];
    threadgroup uint valid[8];
    research_attn_q4_body<512, 8>(q, kc, vc, out, blocks, contexts, cu, positions,
        total_q, batch, heads, kv_heads, head_dim, block_size, max_blocks, scale, window,
        num_blocks, group, tid, sg, lane, threads, kt, vt, valid);
}
