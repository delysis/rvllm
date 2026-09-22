// metal-gqa-kv8: stage eight K/V rows once per group of related Q heads.
// Each SIMD group keeps the existing per-key FP32 online-softmax update order.
// No low-precision probability buffer and no cross-SIMD arithmetic reduction.
template<uint D>
inline void research_gqa_kv8_body(
    device const half *q, device const half *k_cache, device const half *v_cache,
    device half *output, device const int *block_tables, device const int *context_lens,
    device const int *cu_seqlens, device const int *positions,
    uint total_q, uint batch_size, uint num_heads, uint num_kv_heads,
    uint head_dim, uint block_size, uint max_blocks, float scale,
    uint attention_window, uint num_blocks, uint3 group, uint tid, ushort sg,
    ushort lane, uint threads, threadgroup half *kt, threadgroup half *vt,
    threadgroup uint *valid) {
    // All exits before the first barrier are uniform across the threadgroup.
    if (group.x >= total_q || batch_size != 1u || num_heads != 16u || head_dim != D
        || block_size == 0u || max_blocks == 0u || num_blocks == 0u) return;
    if (!((D == 256u && num_kv_heads == 8u && attention_window == 1024u)
        || (D == 512u && num_kv_heads == 1u && attention_window == 0u))) return;
    uint gqa = num_heads / num_kv_heads;
    uint heads_in_group = min(gqa, 4u);
    if (threads != heads_in_group * 32u || group.y >= num_kv_heads
        || group.z >= (gqa + 3u) / 4u) return;
    uint head = group.y * gqa + group.z * 4u + uint(sg);
    size_t q_base = size_t(group.x) * (num_heads * D) + head * D;
    bool poisoned = cu_seqlens[0] != 0 || cu_seqlens[1] != int(total_q)
        || positions[group.x] < 0 || positions[group.x] >= context_lens[0]
        || context_lens[0] <= 0 || size_t(max(context_lens[0], 0)) > size_t(max_blocks) * block_size;
    uint context = uint(max(context_lens[0], 0));
    // The cap bounds table access even for corrupt metadata; poisoned output
    // is nonfinite, not a plausible silent result for an unsupported layout.
    context = uint(min(size_t(context), size_t(max_blocks) * block_size));
    uint end = min(context, uint(max(positions[group.x], 0)) + 1u);
    uint start = attention_window == 0u ? 0u : end - min(end, attention_window);
    float q_lane[D / 32u], out_lane[D / 32u];
    for (uint slot = 0; slot < D / 32u; slot++) {
        q_lane[slot] = float(q[q_base + uint(lane) + slot * 32u]);
        out_lane[slot] = 0.0f;
    }
    float max_score = -INFINITY, sum_exp = 0.0f;
    for (uint first = start; first < end; first += 8u) {
        // Physical page validation precedes every K/V load, including tails.
        for (uint j = tid; j < 8u; j += threads) {
            uint t = first + j;
            int page = t < end ? block_tables[t / block_size] : -1;
            valid[j] = page < 0 ? 0u : (uint(page) < num_blocks ? 1u : 2u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * D; i += threads) {
            uint j = i / D, d = i % D, t = first + j;
            if (valid[j] == 1u) {
                uint page = uint(block_tables[t / block_size]);
                size_t offset = ((size_t(page) * block_size + t % block_size)
                    * num_kv_heads + group.y) * D + d;
                kt[i] = k_cache[offset];
                vt[i] = v_cache[offset];
            } else {
                kt[i] = half(0.0f);
                vt[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8u; j++) {
            if (valid[j] == 2u) poisoned = true;
            if (valid[j] != 1u) continue;
            float partial_score = 0.0f;
            for (uint slot = 0; slot < D / 32u; slot++) {
                uint d = uint(lane) + slot * 32u;
                partial_score += q_lane[slot] * float(kt[j * D + d]);
            }
            float score = simd_sum(partial_score) * scale;
            float next_max = max(max_score, score);
            float correction = exp(max_score - next_max);
            float weight = exp(score - next_max);
            sum_exp = sum_exp * correction + weight;
            for (uint slot = 0; slot < D / 32u; slot++) {
                uint d = uint(lane) + slot * 32u;
                out_lane[slot] = out_lane[slot] * correction + weight * float(vt[j * D + d]);
            }
            max_score = next_max;
        }
        // Every consumer must retire before any group overwrites staging.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (uint slot = 0; slot < D / 32u; slot++) {
        output[q_base + uint(lane) + slot * 32u] =
            poisoned ? half(NAN) : f16_sat(out_lane[slot] * inv_sum);
    }
}

kernel void research_gqa_kv8_d256(
    device const half *q [[buffer(0)]], device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]], device half *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]], device const int *context_lens [[buffer(5)]],
    device const int *cu_seqlens [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]], constant uint &batch_size [[buffer(9)]],
    constant uint &num_heads [[buffer(10)]], constant uint &num_kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]], constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &attention_window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup half kt[8 * 256];
    threadgroup half vt[8 * 256];
    threadgroup uint valid[8];
    research_gqa_kv8_body<256>(q, k_cache, v_cache, output, block_tables, context_lens,
        cu_seqlens, positions, total_q, batch_size, num_heads, num_kv_heads,
        head_dim, block_size, max_blocks, scale, attention_window, num_blocks,
        group, tid, sg, lane, threads, kt, vt, valid);
}

kernel void research_gqa_kv8_d512(
    device const half *q [[buffer(0)]], device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]], device half *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]], device const int *context_lens [[buffer(5)]],
    device const int *cu_seqlens [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]], constant uint &batch_size [[buffer(9)]],
    constant uint &num_heads [[buffer(10)]], constant uint &num_kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]], constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &attention_window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup half kt[8 * 512];
    threadgroup half vt[8 * 512];
    threadgroup uint valid[8];
    research_gqa_kv8_body<512>(q, k_cache, v_cache, output, block_tables, context_lens,
        cu_seqlens, positions, total_q, batch_size, num_heads, num_kv_heads,
        head_dim, block_size, max_blocks, scale, attention_window, num_blocks,
        group, tid, sg, lane, threads, kt, vt, valid);
}
