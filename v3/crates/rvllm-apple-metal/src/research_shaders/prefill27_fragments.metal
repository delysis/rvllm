// The 8x8 fragment coordinate map below is adapted from MLX Steel BaseMMAFrag
// (Copyright (c) 2024 Apple Inc., MIT). See tools/prefill-next/MLX-NOTICE.txt.
// Pinned source: a2a09fd56ccf064121f489528d339de40d2ba8b3,
// mlx/backend/metal/kernels/steel/gemm/mma.h. This is an implementation detail,
// not an Apple-family-independent ABI. The device layout probe is mandatory.
inline ushort2 pr27_coord(ushort lane) {
    const ushort qid = lane / 4u;
    return ushort2((qid & 2u) * 2u + (lane % 2u) * 2u,
                   (qid & 4u) + ((lane / 2u) % 4u));
}
inline void pr27_store(device float *out, size_t i, float x) { out[i] = x; }
inline void pr27_store(device half *out, size_t i, float x) { out[i] = f16_sat(x); }

// Diagnostic only, never a layer-forward dispatch. It verifies the mapping
// against Metal's own matrix load and a nontrivial matrix product. The host
// compares all 128 FP32 values and guard bytes before any new operator screen.
kernel void pr27_fragment_layout_probe(
    device float *out [[buffer(0)]], ushort lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]]) {
    if (any(group != uint3(0)) || any(threads != uint3(32, 1, 1))) return;
    threadgroup float a[64], b[64];
    for (uint e = 0u; e < 2u; ++e) {
        const uint i = uint(lane) * 2u + e;
        a[i] = float(i + 1u);
        b[i] = float((i / 8u + 1u) * (i % 8u + 2u)) / 16.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 am, bm, product(0.0f);
    simdgroup_load(am, a, 8u);
    simdgroup_load(bm, b, 8u);
    simdgroup_multiply_accumulate(product, am, bm, product);
    const ushort2 xy = pr27_coord(lane);
    for (uint e = 0u; e < 2u; ++e) {
        const uint i = uint(xy.y) * 8u + uint(xy.x) + e;
        out[i] = am.thread_elements()[e];
        out[64u + i] = product.thread_elements()[e];
    }
}
