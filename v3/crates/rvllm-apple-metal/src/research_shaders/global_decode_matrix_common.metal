// Atlas-derived FP32 SIMD-matrix QK/PV schedule for the current global D512 ABI.
// BF16 cache storage, FP32 matrix operands/state, once-rounded output.
template<uint R, uint BK, uint P, uint T>
inline void global_decode_matrix_body(
    device const ushort *q, device const ushort *k, device const ushort *v,
    device uchar *output, device const int *table, device const int *contexts,
    device const int *positions, constant GlobalDecodeParams &p,
    uint3 group, uint tid, ushort sg, ushort lane, uint3 threads,
    threadgroup float *qt, threadgroup float *stage,
    threadgroup float *scores, threadgroup float *weights,
    threadgroup float *state, threadgroup int *pages) {
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
    uint visible_pages = (end - 1u) / p.block_size + 1u;
    for (uint b = 0u; b < visible_pages; ++b) {
        int page = table[b];
        if (page >= 0 && uint(page) >= p.num_blocks) return;
    }
    constexpr uint NSG = T / 32u;
    constexpr uint OWN = (R + NSG - 1u) / NSG;
    constexpr uint RM = (R + NSG * 8u - 1u) / (NSG * 8u);
    float u[OWN][16];
    for (uint r = 0u; r < OWN; ++r)
        for (uint d = 0u; d < 16u; ++d) u[r][d] = 0.0f;
    for (uint r = tid; r < R; r += T) {
        state[r] = -INFINITY;
        state[R + r] = 0.0f;
        state[2u * R + r] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first = 0u; first < end; first += BK) {
        for (uint j = tid; j < BK; j += T)
            pages[j] = first + j < end ? table[(first + j) / p.block_size] : -1;
        for (uint i = tid; i < R * BK; i += T) {
            scores[i] = 0.0f;
            weights[i] = 0.0f;
        }
        for (uint r = tid; r < R; r += T) state[2u * R + r] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 accum[RM][BK / 8u];
        for (uint m = 0u; m < RM; ++m)
            for (uint n = 0u; n < BK / 8u; ++n) accum[m][n] = simdgroup_float8x8(0.0f);
        for (uint panel = 0u; panel < 512u; panel += P) {
            for (uint i = tid; i < R * P; i += T)
                qt[i] = global_decode_widen(q[(size_t(group.x) * R + i / P) * 512u + panel + i % P]);
            for (uint i = tid; i < BK * P; i += T) {
                uint j = i / P;
                size_t token = pages[j] >= 0
                    ? size_t(pages[j]) * p.block_size + (first + j) % p.block_size : 0ul;
                stage[i] = pages[j] >= 0
                    ? global_decode_widen(k[token * 512u + panel + i % P]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m = 0u; m < RM; ++m) {
                uint row = (uint(sg) + m * NSG) * 8u;
                if (row >= R) continue;
                for (uint d = 0u; d < P; d += 8u) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, qt + row * P + d, P);
                    for (uint n = 0u; n < BK / 8u; ++n) {
                        simdgroup_float8x8 b;
                        simdgroup_load(b, stage + n * 8u * P + d, P, ulong2(0), true);
                        simdgroup_multiply_accumulate(accum[m][n], a, b, accum[m][n]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint m = 0u; m < RM; ++m) {
            uint row = (uint(sg) + m * NSG) * 8u;
            if (row >= R) continue;
            for (uint n = 0u; n < BK / 8u; ++n)
                simdgroup_store(accum[m][n], scores + row * BK + n * 8u, BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) for (uint r = 0u; r < OWN; ++r) {
            uint row = uint(sg) + r * NSG;
            float next = state[row];
            bool valid = false;
            for (uint j = 0u; j < BK; ++j) if (pages[j] >= 0) {
                next = max(next, scores[row * BK + j]);
                valid = true;
            }
            if (!valid) continue;
            float old_l = state[R + row], old_m = state[row];
            float alpha = old_l == 0.0f ? 0.0f
                : (old_m == next ? 1.0f : precise::exp(old_m - next));
            float sum = 0.0f;
            for (uint j = 0u; j < BK; ++j) if (pages[j] >= 0) {
                float score = scores[row * BK + j];
                float weight = score == next ? 1.0f : precise::exp(score - next);
                weights[row * BK + j] = weight;
                sum += weight;
            }
            state[row] = next;
            state[R + row] = fma(old_l, alpha, sum);
            state[2u * R + row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint panel = 0u; panel < 512u; panel += P) {
            for (uint i = tid; i < BK * P; i += T) {
                uint j = i / P;
                size_t token = pages[j] >= 0
                    ? size_t(pages[j]) * p.block_size + (first + j) % p.block_size : 0ul;
                stage[i] = pages[j] >= 0
                    ? global_decode_widen(v[token * 512u + panel + i % P]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m = 0u; m < RM; ++m) {
                uint row = (uint(sg) + m * NSG) * 8u;
                if (row >= R) continue;
                for (uint d = 0u; d < P; d += 8u) {
                    simdgroup_float8x8 pv(0.0f);
                    for (uint j = 0u; j < BK; j += 8u) {
                        simdgroup_float8x8 a, b;
                        simdgroup_load(a, weights + row * BK + j, BK);
                        simdgroup_load(b, stage + j * P + d, P);
                        simdgroup_multiply_accumulate(pv, a, b, pv);
                    }
                    simdgroup_store(pv, qt + row * P + d, P);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < OWN; ++r) {
                uint row = uint(sg) + r * NSG;
                for (uint d = uint(lane); d < P; d += 32u) {
                    uint slot = (panel + d) / 32u;
                    u[r][slot] = fma(u[r][slot], state[2u * R + row], qt[row * P + d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint r = 0u; r < OWN; ++r) {
        float denom = simd_broadcast(state[R + uint(sg) + r * NSG], 0u);
        float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        uint head = group.x * R + uint(sg) + r * NSG;
        for (uint slot = 0u; slot < 16u; ++slot) {
            size_t index = size_t(head) * 512u + uint(lane) + slot * 32u;
            float result = u[r][slot] * inv;
            if (p.output_kind == 1u) reinterpret_cast<device float *>(output)[index] = result;
            else reinterpret_cast<device ushort *>(output)[index] = global_decode_round(result);
        }
    }
}

#define GLOBAL_DECODE_MATRIX_ENTRY(NAME, R, BK, P, T) \
kernel void NAME( \
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]], \
    device const ushort *v [[buffer(2)]], device uchar *output [[buffer(3)]], \
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]], \
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], \
    uint3 threads [[threads_per_threadgroup]]) { \
    threadgroup float qt[R * P], stage[BK * P], scores[R * BK], weights[R * BK], state[3 * R]; \
    threadgroup int pages[BK]; \
    global_decode_matrix_body<R, BK, P, T>(q, k, v, output, table, contexts, positions, p, \
        group, tid, sg, lane, threads, qt, stage, scores, weights, state, pages); \
}
