// Four causal query positions/TG, one query/SIMDgroup, 16 keys/tile.
// No probability/score BF16 round and no undocumented matrix fragment access.
// K and V share one panel with explicit lifetime barriers. Cache is read-only.
template <uint D>
inline void pr26_attention(
    device const half *Q, device const half *K, device const half *V, device half *O,
    device const int *tables, device const int *contexts, device const int *cu,
    device const int *positions, uint M, uint batch, uint heads, uint kv_heads,
    uint dim, uint block, uint max_blocks, float scale, uint window, uint num_blocks,
    uint3 group, ushort tid, ushort sg, ushort lane, uint3 threads,
    threadgroup half *panel, threadgroup float *scores, threadgroup int *pages) {
    if (threads.x != 128u || threads.y != 1u || threads.z != 1u) return;
    if (M < 6u || M > 2048u || batch != 1u || heads != 16u || dim != D || scale != 1.0f) return;
    if ((D == 256u && (kv_heads != 8u || window != 1024u)) ||
        (D == 512u && (kv_heads != 1u || window != 0u))) return;
    // This ordering makes the multiplication below overflow-free.
    if (block == 0u || block > 4096u || max_blocks == 0u ||
        max_blocks > 4096u / block || num_blocks == 0u) return;
    if (group.z != 0u || group.x >= (M + 3u) / 4u || group.y >= heads) return;
    const uint capacity = block * max_blocks, first = group.x * 4u;
    bool bad_meta = contexts[0] <= 0 || uint(contexts[0]) > capacity || cu[0] != 0 || cu[1] != int(M);
    for (uint j = 0u; j < 4u && first + j < M; ++j)
        bad_meta = bad_meta || positions[first + j] < 0 || positions[first + j] >= contexts[0];
    if (bad_meta) {
        // Invalid live metadata is conspicuous, never a plausible empty result.
        for (uint i = uint(tid); i < 4u * D; i += 128u) {
            const uint row = first + i / D;
            if (row < M) O[(size_t(row) * heads + group.y) * D + i % D] = half(NAN);
        }
        return;
    }
    uint begin = capacity, end = 0u;
    for (uint j = 0u; j < 4u && first + j < M; ++j) {
        const uint e = uint(positions[first + j]) + 1u;
        const uint s = window == 0u ? 0u : (e > window ? e - window : 0u);
        begin = min(begin, s); end = max(end, e);
    }
    const uint row = first + uint(sg), head = group.y;
    const bool active = row < M;
    const uint own_end = active ? uint(positions[row]) + 1u : 0u;
    const uint own_begin = window == 0u ? 0u : (own_end > window ? own_end - window : 0u);
    const uint kv_head = head / (heads / kv_heads), kv_dim = kv_heads * D;
    float q_lane[D / 32u], out_lane[D / 32u];
    #pragma unroll
    for (uint s = 0u; s < D / 32u; ++s) {
        q_lane[s] = active ? float(Q[(size_t(row) * heads + head) * D + uint(lane) + 32u * s]) : 0.0f;
        out_lane[s] = 0.0f;
    }
    float running_max = -INFINITY, running_sum = 0.0f;
    bool poisoned = false;
    for (uint kb = (begin / 16u) * 16u; kb < end; kb += 16u) {
        if (uint(tid) < 16u) {
            const uint t = kb + uint(tid);
            pages[tid] = t < end ? tables[t / block] : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = uint(tid); i < 16u * D; i += 128u) {
            const uint j = i / D, t = kb + j;
            const int page = pages[j];
            panel[i] = page >= 0 && uint(page) < num_blocks
                ? K[(size_t(uint(page)) * block + t % block) * kv_dim + kv_head * D + i % D]
                : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const uint t = kb + j;
            const int page = pages[j];
            const bool visible = active && t >= own_begin && t < own_end;
            poisoned = poisoned || (visible && page >= 0 && uint(page) >= num_blocks);
            float partial = 0.0f;
            #pragma unroll
            for (uint s = 0u; s < D / 32u; ++s)
                partial += q_lane[s] * float(panel[j * D + uint(lane) + 32u * s]);
            const float score = simd_sum(partial) * scale;
            const bool present = visible && page >= 0 && uint(page) < num_blocks;
            poisoned = poisoned || (present && !isfinite(score));
            if (lane == 0u) scores[uint(sg) * 16u + j] = present && isfinite(score) ? score : -INFINITY;
        }
        // Last K reader is finished. Score values are also ready for all lanes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = lane < 16u ? scores[uint(sg) * 16u + uint(lane)] : -INFINITY;
        const float tile_max = simd_max(score);
        const float next_max = max(running_max, tile_max);
        const float correction = running_sum > 0.0f ? exp(running_max - next_max) : 0.0f;
        const float weight = score > -INFINITY ? exp(score - next_max) : 0.0f;
        running_sum = running_sum * correction + simd_sum(weight);
        running_max = next_max;
        if (lane < 16u) scores[uint(sg) * 16u + uint(lane)] = weight;
        for (uint i = uint(tid); i < 16u * D; i += 128u) {
            const uint j = i / D, t = kb + j;
            const int page = pages[j];
            panel[i] = page >= 0 && uint(page) < num_blocks
                ? V[(size_t(uint(page)) * block + t % block) * kv_dim + kv_head * D + i % D]
                : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint s = 0u; s < D / 32u; ++s) out_lane[s] *= correction;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const float w = scores[uint(sg) * 16u + j];
            if (w != 0.0f) {
                #pragma unroll
                for (uint s = 0u; s < D / 32u; ++s)
                    out_lane[s] += w * float(panel[j * D + uint(lane) + 32u * s]);
            }
        }
        // No next K/page producer may overwrite data still read by another SG.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    poisoned = poisoned || !isfinite(running_sum);
    #pragma unroll
    for (uint s = 0u; s < D / 32u; ++s) poisoned = poisoned || !isfinite(out_lane[s]);
    poisoned = simd_any(poisoned);
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    if (active) {
        #pragma unroll
        for (uint s = 0u; s < D / 32u; ++s) {
            const size_t at = (size_t(row) * heads + head) * D + uint(lane) + 32u * s;
            O[at] = poisoned ? half(NAN) : f16_sat(out_lane[s] * inv);
        }
    }
}
