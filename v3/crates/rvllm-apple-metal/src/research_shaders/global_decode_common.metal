// Global decode only: explicit BF16 bits, FP32 QK/state/PV, once-rounded output.
// Included only by an explicitly selected research candidate. No cache writes.
// Compile the candidate and independent oracle with identical math flags.
struct GlobalDecodeParams {
    uint sequences, heads, kv_heads, head_dim;
    uint block_size, max_blocks, num_blocks, window;
    float scale;
    uint output_kind; // 0: BF16, 1: FP32 oracle output; never both.
};

inline float global_decode_widen(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}
inline ushort global_decode_round(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort(bits >> 16u) | ushort(0x40u);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
inline float global_decode_sum32(float value) {
    // Fixed reduction, not an implementation-dependent simd_sum ordering.
    value += simd_shuffle_down(value, 16u);
    value += simd_shuffle_down(value, 8u);
    value += simd_shuffle_down(value, 4u);
    value += simd_shuffle_down(value, 2u);
    value += simd_shuffle_down(value, 1u);
    return value; // only lane zero consumes this result.
}

template<uint R, uint BK, uint P, uint T, bool PerTile>
inline void global_decode_body(
    device const ushort *q, device const ushort *k, device const ushort *v,
    device uchar *output, device const int *table, device const int *contexts,
    device const int *positions, constant GlobalDecodeParams &p,
    uint3 group, uint tid, ushort sg, ushort lane, uint3 threads,
    threadgroup ushort *qt, threadgroup ushort *stage,
    threadgroup float *scores, threadgroup float *alpha, threadgroup float *weight,
    threadgroup int *pages) {
    // All exits/continues enclosing barriers are threadgroup-uniform. Host also
    // validates full model identity, dtype, spans, aliasing and queried PSO limits.
    if (p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u || p.head_dim != 512u
        || p.window != 0u || p.scale != 1.0f || p.output_kind > 1u
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || size_t(p.max_blocks) * p.block_size > 2147483647ul
        || threads.x != T || threads.y != 1u || threads.z != 1u
        || group.x >= 16u / R || group.y != 0u || group.z != 0u) return;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return;
    uint end = uint(position) + 1u;
    // A corrupt positive page cannot cause an OOB read or a partial output.
    // Negative pages are holes. Scan only visible pages, never a future suffix.
    uint visible_pages = (end - 1u) / p.block_size + 1u;
    for (uint b = 0; b < visible_pages; ++b) {
        int page = table[b];
        if (page >= 0 && uint(page) >= p.num_blocks) return;
    }
    constexpr uint NSG = T / 32u;
    constexpr uint OWNED = R / NSG;
    float u[OWNED][16];
    float maxima[OWNED], denominators[OWNED];
    for (uint r = 0; r < OWNED; ++r) {
        maxima[r] = -INFINITY;
        denominators[r] = 0.0f;
        for (uint slot = 0; slot < 16u; ++slot) u[r][slot] = 0.0f;
    }
    for (uint i = tid; i < R * 512u; i += T) {
        qt[i] = q[size_t(group.x) * R * 512u + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first = 0; first < end; first += BK) {
        for (uint j = tid; j < BK; j += T) {
            pages[j] = first + j < end ? table[(first + j) / p.block_size] : -1;
        }
        for (uint i = tid; i < R * BK; i += T) {
            scores[i] = 0.0f;
            weight[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // COMPLETE all D panels before any normalization. Each staged K panel
        // is consumed by every packed head; no repeated per-head device KV load.
        for (uint panel = 0; panel < 512u; panel += P) {
            for (uint i = tid; i < BK * P; i += T) {
                uint j = i / P, d = panel + i % P;
                stage[i] = pages[j] >= 0
                    ? k[(size_t(pages[j]) * p.block_size + (first + j) % p.block_size) * 512u + d]
                    : ushort(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Fixed 64-coordinate reduction leaves across the entire family.
            // P changes staging/barrier cost, NEVER FP32 association order.
            for (uint sub = 0; sub < P; sub += 64u) {
                for (uint r = 0; r < OWNED; ++r) {
                    uint row = uint(sg) + r * NSG;
                    for (uint j = 0; j < BK; ++j) {
                        if (pages[j] < 0) continue;
                        float partial = 0.0f;
                        for (uint d = uint(lane); d < 64u; d += 32u) {
                            partial = fma(global_decode_widen(qt[row * 512u + panel + sub + d]),
                                global_decode_widen(stage[j * P + sub + d]), partial);
                        }
                        partial = global_decode_sum32(partial);
                        if (lane == 0) scores[row * BK + j] += partial;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Only SIMD leaders execute exponentials. Every key's FP32 sufficient
        // statistics are retained until the corresponding V panel is consumed.
        if (lane == 0) {
            for (uint r = 0; r < OWNED; ++r) {
                uint row = uint(sg) + r * NSG;
                if (PerTile) {
                    // An all-hole tile is an identity; do not reuse a prior
                    // tile's output rescale factor.
                    alpha[row * BK] = 1.0f;
                    float next = maxima[r];
                    bool valid = false;
                    for (uint j = 0; j < BK; ++j) if (pages[j] >= 0) {
                        next = max(next, scores[row * BK + j]);
                        valid = true;
                    }
                    if (valid) {
                        float a = denominators[r] == 0.0f ? 0.0f
                            : (next == maxima[r] ? 1.0f : precise::exp(maxima[r] - next));
                        float sum = 0.0f;
                        for (uint j = 0; j < BK; ++j) if (pages[j] >= 0) {
                            float score = scores[row * BK + j];
                            float w = score == next ? 1.0f : precise::exp(score - next);
                            weight[row * BK + j] = w;
                            sum += w;
                        }
                        alpha[row * BK] = a;
                        denominators[r] = fma(denominators[r], a, sum);
                        maxima[r] = next;
                    }
                } else {
                    for (uint j = 0; j < BK; ++j) {
                        if (pages[j] < 0) continue;
                        float score = scores[row * BK + j];
                        float next = max(maxima[r], score);
                        float a = denominators[r] == 0.0f ? 0.0f
                            : (next == maxima[r] ? 1.0f : precise::exp(maxima[r] - next));
                        float w = score == next ? 1.0f : precise::exp(score - next);
                        alpha[row * BK + j] = a;
                        weight[row * BK + j] = w;
                        denominators[r] = fma(denominators[r], a, w);
                        maxima[r] = next;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Reuse K staging for V. No BF16 probabilities or normalized partial
        // outputs. The output state stays in FP32, distributed across lanes.
        for (uint panel = 0; panel < 512u; panel += P) {
            for (uint i = tid; i < BK * P; i += T) {
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
                    if (PerTile) u[r][slot] *= alpha[row * BK];
                    for (uint j = 0; j < BK; ++j) {
                        if (pages[j] < 0) continue;
                        float a = PerTile ? 1.0f : alpha[row * BK + j];
                        u[r][slot] = fma(weight[row * BK + j], global_decode_widen(stage[j * P + d]),
                            u[r][slot] * a);
                    }
                }
            }
            // All consumers finish before staging is reused by ANY SIMD group.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint r = 0; r < OWNED; ++r) {
        float l = simd_broadcast(denominators[r], 0u);
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        uint head = group.x * R + uint(sg) + r * NSG;
        for (uint slot = 0; slot < 16u; ++slot) {
            size_t index = size_t(head) * 512u + uint(lane) + slot * 32u;
            float result = u[r][slot] * inv;
            if (p.output_kind == 1u) reinterpret_cast<device float *>(output)[index] = result;
            else reinterpret_cast<device ushort *>(output)[index] = global_decode_round(result);
        }
    }
}
