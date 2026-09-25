// Copyright 2026. Experimental attention-atlas ABI v1. No shipping selector.
// BF16 storage is explicit ushort bits; all scores, P, state and PV are FP32.
// Metal 3.1, -fno-fast-math. FMA/reduction association is explicit in the SIMD
// family. The matrix family has a DISTINCT, device-qualified numerical order.
struct AtlasParams {
    uint abi, queries, live_keys, kv_heads;
    uint dim, window, page_size, max_blocks;
    uint physical_blocks, splits, output_kind, cache_kind;
    uint rows, keys, panel, threads;
};
inline float atlas_widen(ushort x) { return as_type<float>(uint(x) << 16u); }
inline ushort atlas_round(float x) {
    uint b = as_type<uint>(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return ushort(b >> 16u) | ushort(0x40u);
    return ushort((b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u);
}
inline float atlas_sum32(float x) {
    x += simd_shuffle_down(x, 16u); x += simd_shuffle_down(x, 8u);
    x += simd_shuffle_down(x, 4u); x += simd_shuffle_down(x, 2u);
    x += simd_shuffle_down(x, 1u); return x; // only lane zero consumes
}
inline uint atlas_start(uint pos, uint window) {
    return window == 0u ? 0u : (pos + 1u > window ? pos + 1u - window : 0u);
}
inline bool atlas_visible(uint pos, uint key, uint window) {
    return key <= pos && key >= atlas_start(pos, window);
}
inline bool atlas_shape_ok(constant AtlasParams &p) {
    return p.abi == 0x41540001u && p.queries > 0u && p.queries <= 4096u
        && p.live_keys >= p.queries && p.live_keys <= 262144u
        && ((p.kv_heads == 8u && p.dim == 256u && p.window == 1024u)
            || (p.kv_heads == 1u && p.dim == 512u && p.window == 0u))
        && p.page_size > 0u && p.page_size <= 262144u && p.max_blocks > 0u
        && p.physical_blocks > 0u && size_t(p.max_blocks) * p.page_size <= 2147483647ul
        && p.live_keys <= size_t(p.max_blocks) * p.page_size
        && (p.splits == 1u || p.splits == 2u || p.splits == 4u || p.splits == 8u
            || p.splits == 16u || p.splits == 32u)
        && (p.splits == 1u || p.queries <= 8u) && p.output_kind <= 1u
        && (p.cache_kind == 0u || (p.cache_kind == 1u && p.dim == 512u));
}

// Serialized metadata prepass, with a dependent encoder before the main pass.
// On refusal only status changes. Neither outputs nor partial states are touched.
kernel void atlas_validate(device const int *pages [[buffer(3)]],
    device const int *positions [[buffer(4)]], device uint *status [[buffer(7)]],
    constant AtlasParams &p [[buffer(12)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    status[0] = 1u;
    if (!atlas_shape_ok(p)) return;
    status[0] = 2u;
    int first = positions[0];
    if (first < 0) return;
    for (uint i = 0u; i < p.queries; ++i) {
        int pos = positions[i];
        if (pos < 0 || uint(pos) >= p.live_keys || size_t(first) + i != size_t(pos)) return;
    }
    uint lo = atlas_start(uint(first), p.window) / p.page_size;
    uint hi = uint(positions[p.queries - 1u]) / p.page_size;
    status[0] = 3u;
    for (uint b = lo; b <= hi; ++b) {
        int page = pages[b];
        if (page >= 0 && uint(page) >= p.physical_blocks) return;
    }
    status[0] = 0u;
}

// Separate experimental producer contract. Do NOT feed this FP32 fused
// projection values narrowed to BF16 and call it a layout-only optimization.
// n = widen(z)*r, nK = BF16(n*gamma), V = BF16(n).
// RoPE uses compact FP32 coefficients and exactly two rounded nK operands;
// the rotation's second multiply-add is explicitly fused, defining this ABI.
inline ushort atlas_k(device const ushort *k, device const float *factor,
    device const ushort *gamma, device const float *cs, device const float *sn,
    size_t physical_token, uint logical_token, uint head, uint d, constant AtlasParams &p) {
    size_t base = (physical_token * p.kv_heads + head) * p.dim;
    if (p.cache_kind == 0u) return k[base + d];
    float norm = atlas_widen(k[base + d]) * factor[physical_token];
    ushort scaled = atlas_round(norm * atlas_widen(gamma[d]));
    bool first = d < 64u;
    bool second = d >= 256u && d < 320u;
    if (!first && !second) return scaled;
    uint pair = first ? d : d - 256u;
    float x0 = atlas_widen(atlas_round((atlas_widen(k[base + pair]) * factor[physical_token])
        * atlas_widen(gamma[pair])));
    float x1 = atlas_widen(atlas_round((atlas_widen(k[base + pair + 256u]) * factor[physical_token])
        * atlas_widen(gamma[pair + 256u])));
    float c = cs[size_t(logical_token) * 64u + pair];
    float s = sn[size_t(logical_token) * 64u + pair];
    return atlas_round(first ? fma(-x1, s, x0 * c) : fma(x0, s, x1 * c));
}
inline ushort atlas_v(device const ushort *k, device const ushort *v,
    device const float *factor, size_t physical_token, uint head, uint d,
    constant AtlasParams &p) {
    size_t index = (physical_token * p.kv_heads + head) * p.dim + d;
    return p.cache_kind == 0u ? v[index] : atlas_round(atlas_widen(k[index]) * factor[physical_token]);
}
inline void atlas_store(device uchar *out, size_t index, float x, uint kind) {
    if (kind == 0u) reinterpret_cast<device ushort *>(out)[index] = atlas_round(x);
    else reinterpret_cast<device float *>(out)[index] = x;
}

// Shared signature keeps source and Rust binding indices in one stable ABI.
#define ATLAS_ARGUMENTS \
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]], \
    device const ushort *v [[buffer(2)]], device const int *table [[buffer(3)]], \
    device const int *positions [[buffer(4)]], device uchar *out [[buffer(5)]], \
    device float *partial [[buffer(6)]], device const uint *status [[buffer(7)]], \
    device const float *factor [[buffer(8)]], device const ushort *gamma [[buffer(9)]], \
    device const float *cs [[buffer(10)]], device const float *sn [[buffer(11)]], \
    constant AtlasParams &p [[buffer(12)]], uint3 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]

#define ATLAS_PASS q,k,v,table,positions,out,partial,status,factor,gamma,cs,sn,p,group,tid,sg,lane,threads

// Tiled, GQA-packed, D-paneled output-stationary SIMD schedule.
// PerKey preserves fixed-64-leaf QK and per-key FP32 updates. PerTile reduces
// output rescaling; it deliberately changes association and is a separate arm.
template<uint D, uint R, uint BK, uint P, uint T, bool PerKey>
inline void atlas_coop_body(device const ushort *q, device const ushort *k,
    device const ushort *v, device const int *table, device const int *positions,
    device uchar *out, device float *partial, device const uint *status,
    device const float *factor, device const ushort *gamma, device const float *cs,
    device const float *sn, constant AtlasParams &p, uint3 group, uint tid,
    ushort sg, ushort lane, uint3 threads, threadgroup ushort *qt,
    threadgroup ushort *stage, threadgroup float *scores, threadgroup float *alpha,
    threadgroup float *weights, threadgroup float *ml, threadgroup int *pages) {
    if (status[0] != 0u || !atlas_shape_ok(p) || p.dim != D || p.rows != R
        || p.keys != BK || p.panel != P || p.threads != T
        || threads.x != T || threads.y != 1u || threads.z != 1u
        || group.y >= p.kv_heads || group.z >= p.splits) return;
    uint gqa = 16u / p.kv_heads;
    uint packed0 = group.x * R;
    if (packed0 >= p.queries * gqa) return;
    constexpr uint NSG = T / 32u;
    constexpr uint OWN = (R + NSG - 1u) / NSG;
    float u[OWN][D / 32u];
    for (uint r = 0u; r < OWN; ++r)
        for (uint d = 0u; d < D / 32u; ++d) u[r][d] = 0.0f;
    for (uint r = tid; r < R; r += T) { ml[r] = -INFINITY; ml[R + r] = 0.0f; }
    uint first_query = packed0 / gqa;
    uint last_query = min(packed0 + R, p.queries * gqa) - 1u;
    last_query /= gqa;
    uint start = atlas_start(uint(positions[first_query]), p.window);
    uint end = uint(positions[last_query]) + 1u;
    size_t extent = end - start;
    uint stop = start + uint(extent * (group.z + 1u) / p.splits);
    start += uint(extent * group.z / p.splits);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first = start; first < stop; first += BK) {
        for (uint j = tid; j < BK; j += T)
            pages[j] = first + j < stop ? table[(first + j) / p.page_size] : -1;
        for (uint i = tid; i < R * BK; i += T) {
            scores[i] = 0.0f; weights[i] = 0.0f; alpha[i] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool any_page = false;
        for (uint j = 0u; j < BK; ++j) any_page |= pages[j] >= 0;
        // Retire every read before the next iteration may overwrite pages.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!any_page) continue;
        for (uint panel = 0u; panel < D; panel += P) {
            for (uint i = tid; i < R * P; i += T) {
                uint row = i / P, packed = packed0 + row;
                uint token = packed / gqa, head = group.y * gqa + packed % gqa;
                qt[i] = token < p.queries ? q[(size_t(token) * 16u + head) * D + panel + i % P] : ushort(0);
            }
            for (uint i = tid; i < BK * P; i += T) {
                uint j = i / P;
                size_t pt = pages[j] >= 0 ? size_t(pages[j]) * p.page_size + (first + j) % p.page_size : 0ul;
                stage[i] = pages[j] >= 0 ? atlas_k(k,factor,gamma,cs,sn,pt,first+j,group.y,panel+i%P,p) : ushort(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Complete ALL D leaves before softmax. P only changes staging.
            for (uint sub = 0u; sub < P; sub += 64u) {
                for (uint r = 0u; r < OWN; ++r) {
                    uint row = uint(sg) + r * NSG;
                    uint token = (packed0 + row) / gqa;
                    if (row >= R || token >= p.queries) continue;
                    uint pos = uint(positions[token]);
                    bool all_visible = first >= atlas_start(pos,p.window) && first + BK <= pos + 1u;
                    for (uint j = 0u; j < BK; ++j) {
                        if (pages[j] < 0 || (!all_visible && !atlas_visible(pos,first+j,p.window))) continue;
                        float dot = 0.0f;
                        for (uint d = uint(lane); d < 64u; d += 32u)
                            dot = fma(atlas_widen(qt[row*P+sub+d]), atlas_widen(stage[j*P+sub+d]), dot);
                        dot = atlas_sum32(dot);
                        if (lane == 0) scores[row*BK+j] += dot;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (uint r = 0u; r < OWN; ++r) {
                uint row = uint(sg) + r * NSG, token = (packed0 + row) / gqa;
                if (row >= R || token >= p.queries) continue;
                uint pos = uint(positions[token]);
                float m = ml[row], l = ml[R+row];
                if (PerKey) {
                    for (uint j = 0u; j < BK; ++j) {
                        if (pages[j] < 0 || !atlas_visible(pos,first+j,p.window)) continue;
                        float s = scores[row*BK+j], next = max(m,s);
                        float a = l == 0.0f ? 0.0f : (m == next ? 1.0f : precise::exp(m-next));
                        float w = s == next ? 1.0f : precise::exp(s-next);
                        alpha[row*BK+j] = a; weights[row*BK+j] = w;
                        l = fma(l,a,w); m = next;
                    }
                } else {
                    float next = m;
                    bool valid = false;
                    for (uint j = 0u; j < BK; ++j) {
                        if (pages[j] >= 0 && atlas_visible(pos,first+j,p.window)) {
                            next = max(next,scores[row*BK+j]); valid = true;
                        }
                    }
                    if (valid) {
                        float a = l == 0.0f ? 0.0f : (m == next ? 1.0f : precise::exp(m-next));
                        float sum = 0.0f;
                        for (uint j = 0u; j < BK; ++j) {
                            if (pages[j] < 0 || !atlas_visible(pos,first+j,p.window)) continue;
                            float s = scores[row*BK+j];
                            float w = s == next ? 1.0f : precise::exp(s-next);
                            weights[row*BK+j] = w; sum += w;
                        }
                        alpha[row*BK] = a; l = fma(l,a,sum); m = next;
                    }
                }
                ml[row] = m; ml[R+row] = l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint panel = 0u; panel < D; panel += P) {
            for (uint i = tid; i < BK * P; i += T) {
                uint j = i / P;
                size_t pt = pages[j] >= 0 ? size_t(pages[j]) * p.page_size + (first+j)%p.page_size : 0ul;
                stage[i] = pages[j] >= 0 ? atlas_v(k,v,factor,pt,group.y,panel+i%P,p) : ushort(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < OWN; ++r) {
                uint row = uint(sg) + r * NSG;
                if (row >= R) continue;
                for (uint d = uint(lane); d < P; d += 32u) {
                    uint slot = (panel + d) / 32u;
                    for (uint j = 0u; j < BK; ++j)
                        u[r][slot] = fma(weights[row*BK+j],atlas_widen(stage[j*P+d]),u[r][slot]*alpha[row*BK+j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint r = 0u; r < OWN; ++r) {
        uint row = uint(sg) + r * NSG, packed = packed0 + row;
        uint token = packed / gqa, head = group.y * gqa + packed % gqa;
        if (row >= R || token >= p.queries) continue;
        size_t qr = size_t(token) * 16u + head;
        float l = ml[R+row], m = ml[row];
        float inv = l > 0.0f ? 1.0f/l : 0.0f;
        size_t state = (qr*p.splits+group.z)*(D+2u);
        if (p.splits > 1u && lane == 0) { partial[state] = m; partial[state+1u] = l; }
        for (uint slot = 0u; slot < D/32u; ++slot) {
            uint d = uint(lane) + slot*32u;
            if (p.splits > 1u) partial[state+2u+d] = u[r][slot];
            else atlas_store(out,qr*D+d,u[r][slot]*inv,p.output_kind);
        }
    }
}

#define ATLAS_COOP_ENTRY(NAME,D,R,BK,P,T,KEY,SPLITS) \
    kernel void NAME(ATLAS_ARGUMENTS) { \
        if (p.splits != SPLITS) return; \
        threadgroup ushort qt[R*P], stage[BK*P]; \
        threadgroup float scores[R*BK], alpha[R*BK], weights[R*BK], ml[2*R]; \
        threadgroup int pages[BK]; \
        atlas_coop_body<D,R,BK,P,T,KEY>(ATLAS_PASS,qt,stage,scores,alpha,weights,ml,pages); \
    }

// Deterministic common-max normalization plus balanced binary reductions.
// NOT an average of independently normalized partition outputs. Empty leaves
// are identities; no subtraction of two negative infinities is evaluated.
kernel void atlas_merge(device uchar *out [[buffer(5)]], device const float *partial [[buffer(6)]],
    device const uint *status [[buffer(7)]], constant AtlasParams &p [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (status[0] != 0u || !atlas_shape_ok(p) || p.splits == 1u
        || group.x >= p.queries * 16u || group.y != 0u || group.z != 0u
        || threads.x != 32u || threads.y != 1u || threads.z != 1u) return;
    threadgroup float maxima[32], den[32], weights[32];
    size_t base = size_t(group.x)*p.splits*(p.dim+2u);
    float l = uint(lane) < p.splits ? partial[base+uint(lane)*(p.dim+2u)+1u] : 0.0f;
    maxima[lane] = l > 0.0f ? partial[base+uint(lane)*(p.dim+2u)] : -INFINITY;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=16u; step>0u; step>>=1u) {
        if (uint(lane)<step) maxima[lane]=max(maxima[lane],maxima[uint(lane)+step]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = maxima[0];
    float w = l > 0.0f ? precise::exp(partial[base+uint(lane)*(p.dim+2u)]-m) : 0.0f;
    weights[lane]=w; den[lane]=w*l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=16u; step>0u; step>>=1u) {
        if (uint(lane)<step) den[lane]+=den[uint(lane)+step];
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = den[0]>0.0f ? 1.0f/den[0] : 0.0f;
    for (uint d=uint(lane); d<p.dim; d+=32u) {
        float values[32];
        for (uint i=0u;i<32u;++i)
            values[i]=i<p.splits ? partial[base+i*(p.dim+2u)+2u+d]*weights[i] : 0.0f;
        for (uint step=16u;step>0u;step>>=1u)
            for (uint i=0u;i<step;++i) values[i]+=values[i+step];
        atlas_store(out,size_t(group.x)*p.dim+d,values[0]*inv,p.output_kind);
    }
}
