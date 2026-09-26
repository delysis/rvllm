// Bounded D512/GQA16 split-KV decode. This file follows global_decode_common.metal,
// which supplies GlobalDecodeParams, BF16 conversion, and fixed-tree reduction.
constant uint GLOBAL_SPLIT_STRIDE = 514u; // FP32 numerator[512], maximum, denominator.

template<uint R, uint S, uint P, uint T, uint PARTITIONS, bool DYNAMIC>
inline void global_decode_split_partial_body(
    device const ushort *q, device const ushort *k, device const ushort *v,
    device float *partials, device const int *table, device const int *contexts,
    device const int *positions, constant GlobalDecodeParams &p,
    uint3 group, uint tid, ushort sg, ushort lane, uint3 threads,
    threadgroup ushort *qt, threadgroup ushort *stage,
    threadgroup float *scores, threadgroup float *alpha, threadgroup float *weight,
    threadgroup int *pages) {
    constexpr uint NSG = T / 32u;
    constexpr uint OWNED = R / NSG;
    if (threads.x != T || threads.y != 1u || threads.z != 1u
        || group.x >= 16u / R || group.y >= PARTITIONS || group.z != 0u) return;

    // Every launched partition gets a deterministic identity before validation.
    for (uint r = 0; r < OWNED; ++r) {
        uint row = uint(sg) + r * NSG;
        uint head = group.x * R + row;
        size_t base = (size_t(group.y) * 16ul + head) * GLOBAL_SPLIT_STRIDE;
        for (uint d = uint(lane); d < 512u; d += 32u) partials[base + d] = 0.0f;
        if (lane == 0u) {
            partials[base + 512u] = -INFINITY;
            partials[base + 513u] = 0.0f;
        }
    }
    if (p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u || p.head_dim != 512u
        || p.window != 0u || p.scale != 1.0f || p.output_kind > 1u
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || size_t(p.max_blocks) * p.block_size > 4096ul) return;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return;
    uint end = uint(position) + 1u;
    uint visible_pages = (end - 1u) / p.block_size + 1u;
    for (uint b = 0; b < visible_pages; ++b) {
        int page = table[b];
        if (page >= 0 && uint(page) >= p.num_blocks) return;
    }
    uint begin = DYNAMIC
        ? uint((ulong(end) * ulong(group.y)) / ulong(PARTITIONS))
        : group.y * S;
    uint partition_end = DYNAMIC
        ? uint((ulong(end) * ulong(group.y + 1u)) / ulong(PARTITIONS))
        : min(begin + S, end);
    if (begin >= partition_end) return;

    float u[OWNED][16];
    float maxima[OWNED], denominators[OWNED];
    for (uint r = 0; r < OWNED; ++r) {
        maxima[r] = -INFINITY;
        denominators[r] = 0.0f;
        for (uint slot = 0; slot < 16u; ++slot) u[r][slot] = 0.0f;
    }
    for (uint i = tid; i < R * 512u; i += T)
        qt[i] = q[size_t(group.x) * R * 512u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first = begin; first < partition_end; first += 8u) {
        for (uint j = tid; j < 8u; j += T)
            pages[j] = first + j < partition_end ? table[(first + j) / p.block_size] : -1;
        for (uint i = tid; i < R * 8u; i += T) scores[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint panel = 0; panel < 512u; panel += P) {
            for (uint i = tid; i < 8u * P; i += T) {
                uint j = i / P, d = panel + i % P;
                stage[i] = pages[j] >= 0
                    ? k[(size_t(pages[j]) * p.block_size + (first + j) % p.block_size) * 512u + d]
                    : ushort(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < P; sub += 64u) {
                for (uint r = 0; r < OWNED; ++r) {
                    uint row = uint(sg) + r * NSG;
                    for (uint j = 0; j < 8u; ++j) {
                        if (pages[j] < 0) continue;
                        float dot = 0.0f;
                        for (uint d = uint(lane); d < 64u; d += 32u)
                            dot = fma(global_decode_widen(qt[row * 512u + panel + sub + d]),
                                global_decode_widen(stage[j * P + sub + d]), dot);
                        dot = global_decode_sum32(dot);
                        if (lane == 0u) scores[row * 8u + j] += dot;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            for (uint r = 0; r < OWNED; ++r) {
                uint row = uint(sg) + r * NSG;
                for (uint j = 0; j < 8u; ++j) {
                    if (pages[j] < 0) continue;
                    float score = scores[row * 8u + j];
                    float next = max(maxima[r], score);
                    float a = denominators[r] == 0.0f ? 0.0f
                        : (next == maxima[r] ? 1.0f : precise::exp(maxima[r] - next));
                    float w = score == next ? 1.0f : precise::exp(score - next);
                    alpha[row * 8u + j] = a;
                    weight[row * 8u + j] = w;
                    denominators[r] = fma(denominators[r], a, w);
                    maxima[r] = next;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint panel = 0; panel < 512u; panel += P) {
            for (uint i = tid; i < 8u * P; i += T) {
                uint j = i / P, d = panel + i % P;
                stage[i] = pages[j] >= 0
                    ? v[(size_t(pages[j]) * p.block_size + (first + j) % p.block_size) * 512u + d]
                    : ushort(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < OWNED; ++r) {
                uint row = uint(sg) + r * NSG;
                for (uint d = uint(lane); d < P; d += 32u) {
                    uint slot = (panel + d) / 32u;
                    for (uint j = 0; j < 8u; ++j) {
                        if (pages[j] < 0) continue;
                        u[r][slot] = fma(weight[row * 8u + j],
                            global_decode_widen(stage[j * P + d]), u[r][slot] * alpha[row * 8u + j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint r = 0; r < OWNED; ++r) {
        uint head = group.x * R + uint(sg) + r * NSG;
        size_t base = (size_t(group.y) * 16ul + head) * GLOBAL_SPLIT_STRIDE;
        for (uint slot = 0; slot < 16u; ++slot)
            partials[base + uint(lane) + slot * 32u] = u[r][slot];
        if (lane == 0u) {
            partials[base + 512u] = maxima[r];
            partials[base + 513u] = denominators[r];
        }
    }
}

template<uint PARTITIONS>
inline void global_decode_split_merge_body(
    device const float *partials, device uchar *output, device const int *table,
    device const int *contexts, device const int *positions,
    constant GlobalDecodeParams &p, uint head, ushort lane, uint3 threads) {
    if (threads.x != 32u || threads.y != 1u || threads.z != 1u || head >= 16u
        || p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u || p.head_dim != 512u
        || p.window != 0u || p.scale != 1.0f || p.output_kind > 1u
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || size_t(p.max_blocks) * p.block_size > 4096ul) return;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return;
    uint end = uint(position) + 1u;
    uint visible_pages = (end - 1u) / p.block_size + 1u;
    for (uint b = 0; b < visible_pages; ++b) {
        int page = table[b];
        if (page >= 0 && uint(page) >= p.num_blocks) return;
    }
    float numerator[16];
    for (uint slot = 0; slot < 16u; ++slot) numerator[slot] = 0.0f;
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint part = 0; part < PARTITIONS; ++part) {
        size_t base = (size_t(part) * 16ul + head) * GLOBAL_SPLIT_STRIDE;
        float other_l = partials[base + 513u];
        if (other_l == 0.0f) continue;
        float other_m = partials[base + 512u];
        if (denominator == 0.0f) {
            maximum = other_m;
            denominator = other_l;
            for (uint slot = 0; slot < 16u; ++slot)
                numerator[slot] = partials[base + uint(lane) + slot * 32u];
            continue;
        }
        float next = max(maximum, other_m);
        float a = next == maximum ? 1.0f : precise::exp(maximum - next);
        float b = next == other_m ? 1.0f : precise::exp(other_m - next);
        for (uint slot = 0; slot < 16u; ++slot)
            numerator[slot] = fma(partials[base + uint(lane) + slot * 32u], b,
                numerator[slot] * a);
        denominator = fma(denominator, a, other_l * b);
        maximum = next;
    }
    float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint slot = 0; slot < 16u; ++slot) {
        size_t index = size_t(head) * 512ul + uint(lane) + slot * 32u;
        float result = numerator[slot] * inv;
        if (p.output_kind == 1u) reinterpret_cast<device float *>(output)[index] = result;
        else reinterpret_cast<device ushort *>(output)[index] = global_decode_round(result);
    }
}
