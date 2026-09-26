#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Short-context global decode candidate for exact Gemma 4 geometry:
// Hq=16, Hkv=1, D=512, scale=1.0, one sequence.
// Intended selector: live KV <= 512. It deliberately uses NO split-KV and NO
// merge kernel. One 128-thread TG handles four query heads (one/SIMDgroup),
// staging each K/V vector once into threadgroup memory so four heads reuse it.
// Four TGs cover the 16 global heads.
//
// Simplified contiguous-KV research ABI; integration must map the same body to
// rvLLM's paged-cache ownership, holes, newest-K/V and rollback semantics.

static inline float bf16_to_f32(ushort x) { return as_type<float>(uint(x) << 16); }
static inline ushort f32_to_bf16_rne(float x) {
    uint u = as_type<uint>(x); uint lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb; return ushort(u >> 16);
}

kernel void gemma4_global_d512_g16_short_unsplit(
    device const ushort *q [[buffer(0)]],        // [16,512]
    device const ushort *k [[buffer(1)]],        // [L,512]
    device const ushort *v [[buffer(2)]],        // [L,512]
    device ushort *out [[buffer(3)]],            // [16,512]
    constant uint &L [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (threads != 128 || L == 0 || L > 512) return;
    const uint head = tg * 4u + uint(sg);
    if (head >= 16u) return;

    threadgroup ushort kt[512];
    threadgroup ushort vt[512];

    float qreg[16];
    float oreg[16];
    #pragma unroll
    for (uint s = 0; s < 16; ++s) {
        const uint d = uint(lane) + s * 32u;
        qreg[s] = bf16_to_f32(q[head * 512u + d]);
        oreg[s] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    for (uint t = 0; t < L; ++t) {
        // Cooperative staging: 128 threads load four dimensions each.
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            const uint d = tid + i * 128u;
            kt[d] = k[ulong(t) * 512ul + d];
            vt[d] = v[ulong(t) * 512ul + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) {
            const uint d = uint(lane) + s * 32u;
            partial += qreg[s] * bf16_to_f32(kt[d]);
        }
        const float score = simd_sum(partial); // Gemma 4 scale == 1.0
        const float nm = max(m, score);
        const float a = exp(m - nm);
        const float b = exp(score - nm);
        l = l * a + b;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) {
            const uint d = uint(lane) + s * 32u;
            oreg[s] = oreg[s] * a + b * bf16_to_f32(vt[d]);
        }
        m = nm;

        // All four SIMDgroups must finish before the next token overwrites kt/vt.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    #pragma unroll
    for (uint s = 0; s < 16; ++s) {
        const uint d = uint(lane) + s * 32u;
        out[head * 512u + d] = f32_to_bf16_rne(oreg[s] * inv);
    }
}
