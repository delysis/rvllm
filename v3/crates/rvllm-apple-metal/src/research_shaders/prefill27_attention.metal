// Q8/K32 matrix attention: four SGs first split the 32 score columns, then
// split the output D columns. Q stays in native-storage shared memory once.
// K/V are loaded into FP32 fragments directly; no D512 full-K/V shared tile.
// Both matrix products, probabilities, running max/denominator and output
// accumulators remain FP32. This is arithmetic-changing, NOT a bitwise claim.
template <uint D>
inline void pr27_attention(device const half *Q, device const half *K,
    device const half *V, device half *O, device const int *tables,
    device const int *contexts, device const int *cu, device const int *positions,
    uint M, uint batch, uint heads, uint kv_heads, uint dim, uint block,
    uint max_blocks, float scale, uint window, uint num_blocks,
    uint3 group, ushort tid, ushort sg, ushort lane, uint3 threads,
    threadgroup half *qt, threadgroup float *scores, threadgroup int *pages,
    threadgroup float *stats, threadgroup uint *poison, threadgroup uint *flags) {
    if (any(threads != uint3(128, 1, 1))) return;
    if (M < 6u || M > 2048u || batch != 1u || heads != 16u || dim != D || scale != 1.0f) return;
    if ((D == 256u && (kv_heads != 8u || window != 1024u)) ||
        (D == 512u && (kv_heads != 1u || window != 0u))) return;
    if (block == 0u || block > 4096u || max_blocks == 0u ||
        max_blocks > 4096u / block || num_blocks == 0u) return;
    if (group.z || group.x >= (M + 7u) / 8u || group.y >= heads) return;
    const uint first = group.x * 8u, head = group.y, capacity = block * max_blocks;
    bool bad = contexts[0] <= 0 || uint(contexts[0]) > capacity || cu[0] != 0 || cu[1] != int(M);
    for (uint r = 0u; r < 8u && first + r < M; ++r)
        bad = bad || positions[first + r] < 0 || positions[first + r] >= contexts[0];
    if (bad) {
        for (uint i = uint(tid); i < 8u * D; i += 128u)
            if (first + i / D < M)
                O[(size_t(first + i / D) * heads + head) * D + i % D] = half(NAN);
        return;
    }
    uint begin = capacity, end = 0u;
    for (uint r = 0u; r < 8u && first + r < M; ++r) {
        const uint e = uint(positions[first + r]) + 1u;
        begin = min(begin, window == 0u ? 0u : (e > window ? e - window : 0u));
        end = max(end, e);
    }
    const uint kvh = head / (heads / kv_heads), kvdim = kv_heads * D;
    for (uint i = uint(tid); i < 8u * D / 4u; i += 128u) {
        const uint r = i / (D / 4u), d = (i % (D / 4u)) * 4u;
        *((threadgroup vec<half, 4> *)(qt + r * D + d)) = first + r < M
            ? *((device const vec<half, 4> *)(Q + (size_t(first + r) * heads + head) * D + d))
            : vec<half, 4>(half(0.0f));
    }
    if (tid < 8u) {
        stats[tid] = -INFINITY; stats[8u + tid] = 0.0f;
        stats[16u + tid] = 0.0f; poison[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort2 xy = pr27_coord(lane);
    const uint rr = uint(xy.y), cc = uint(xy.x);
    simdgroup_float8x8 out[D / 32u];
    #pragma unroll
    for (uint j = 0u; j < D / 32u; ++j) out[j] = simdgroup_float8x8(0.0f);
    for (uint kb = (begin / 32u) * 32u; kb < end; kb += 32u) {
        if (tid < 32u) {
            const uint t = kb + uint(tid);
            pages[tid] = t >= begin && t < end ? tables[t / block] : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 score(0.0f);
        // Each SG owns eight keys and all eight queries; no per-key SIMD sum.
        for (uint dk = 0u; dk < D; dk += 8u) {
            simdgroup_float8x8 a, b;
            #pragma unroll
            for (uint e = 0u; e < 2u; ++e) {
                a.thread_elements()[e] = float(qt[rr * D + dk + cc + e]);
                const uint j = uint(sg) * 8u + cc + e, t = kb + j;
                const int page = pages[j];
                b.thread_elements()[e] = page >= 0 && uint(page) < num_blocks
                    ? float(K[(size_t(uint(page)) * block + t % block) * kvdim + kvh * D + dk + rr])
                    : 0.0f;
            }
            simdgroup_multiply_accumulate(score, a, b, score);
        }
        simdgroup_store(score, scores + uint(sg) * 8u, 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One SIMD group handles each of two rows, with 32 keys across lanes.
        #pragma unroll
        for (uint j = 0u; j < 2u; ++j) {
            const uint r = uint(sg) * 2u + j, row = first + r, t = kb + uint(lane);
            const uint own_end = row < M ? uint(positions[row]) + 1u : 0u;
            const uint own_begin = window == 0u ? 0u : (own_end > window ? own_end - window : 0u);
            const bool visible = row < M && t >= own_begin && t < own_end;
            const int page = pages[lane];
            const bool present = visible && page >= 0 && uint(page) < num_blocks;
            const float raw = scores[r * 32u + uint(lane)] * scale;
            const bool row_bad = simd_any((visible && page >= 0 && uint(page) >= num_blocks) ||
                                          (present && !isfinite(raw)));
            const float s = present && isfinite(raw) ? raw : -INFINITY;
            const float next_max = max(stats[r], simd_max(s));
            const float correction = stats[8u + r] > 0.0f ? exp(stats[r] - next_max) : 0.0f;
            const float p = s > -INFINITY ? exp(s - next_max) : 0.0f;
            const float total = stats[8u + r] * correction + simd_sum(p);
            scores[r * 32u + uint(lane)] = p;
            if (lane == 0u) {
                stats[r] = next_max; stats[8u + r] = total; stats[16u + r] = correction;
                poison[r] |= uint(row_bad || !isfinite(total));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0u; j < D / 32u; ++j) {
            #pragma unroll
            for (uint e = 0u; e < 2u; ++e)
                out[j].thread_elements()[e] *= stats[16u + rr];
        }
        uint bad_values = 0u;
        // Same four SGs now split D. P is FP32; V is widened exactly from BF16.
        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 p;
            simdgroup_load(p, scores + kk, 32u);
            #pragma unroll
            for (uint j = 0u; j < D / 32u; ++j) {
                simdgroup_float8x8 v;
                #pragma unroll
                for (uint e = 0u; e < 2u; ++e) {
                    const uint key = kk + rr, t = kb + key;
                    const int page = pages[key];
                    const uint d = uint(sg) * (D / 4u) + j * 8u + cc + e;
                    const float value = page >= 0 && uint(page) < num_blocks
                        ? float(V[(size_t(uint(page)) * block + t % block) * kvdim + kvh * D + d])
                        : 0.0f;
                    // 0*NaN in a matrix multiply would contaminate unrelated
                    // masked query rows. Sanitize, then poison ONLY rows with
                    // nonzero probability on the bad key, after an SG-wide OR.
                    if (!isfinite(value)) bad_values |= 1u << key;
                    v.thread_elements()[e] = isfinite(value) ? value : 0.0f;
                }
                simdgroup_multiply_accumulate(out[j], p, v, out[j]);
            }
        }
        const uint bad_keys = simd_or(bad_values);
        if (lane == 0u) flags[sg] = bad_keys;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint all_bad_keys = flags[0] | flags[1] | flags[2] | flags[3];
        #pragma unroll
        for (uint j = 0u; j < 2u; ++j) {
            const uint r = uint(sg) * 2u + j;
            const bool row_bad = simd_any(((all_bad_keys >> uint(lane)) & 1u) != 0u &&
                                         scores[r * 32u + uint(lane)] != 0.0f);
            if (lane == 0u) poison[r] |= uint(row_bad);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint bad_rows = 0u;
    #pragma unroll
    for (uint j = 0u; j < D / 32u; ++j) {
        #pragma unroll
        for (uint e = 0u; e < 2u; ++e)
            if (!isfinite(out[j].thread_elements()[e])) bad_rows |= 1u << rr;
    }
    const uint sg_bad_rows = simd_or(bad_rows);
    if (lane == 0u) flags[sg] = sg_bad_rows;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool row_bad = poison[rr] != 0u || (((flags[0] | flags[1] | flags[2] | flags[3]) >> rr) & 1u);
    const float inv = stats[8u + rr] > 0.0f ? 1.0f / stats[8u + rr] : 0.0f;
    if (first + rr < M) {
        #pragma unroll
        for (uint j = 0u; j < D / 32u; ++j) {
            #pragma unroll
            for (uint e = 0u; e < 2u; ++e) {
                const uint d = uint(sg) * (D / 4u) + j * 8u + cc + e;
                O[(size_t(first + rr) * heads + head) * D + d] =
                    row_bad ? half(NAN) : f16_sat(out[j].thread_elements()[e] * inv);
            }
        }
    }
}
