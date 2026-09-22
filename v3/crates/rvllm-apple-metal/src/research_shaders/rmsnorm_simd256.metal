// metal-rmsnorm-simd256: only standalone post-projection RMS norms.
// FP32 statistics and real gamma; only the reduction tree changes.
kernel void wave2_rmsnorm_simd256(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    device const half *gamma [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    // Uniform refusal before any barrier. The host separately requires SIMD32.
    if (threads.x != 256u || threads.y != 1u || threads.z != 1u || hidden != 3840u) return;
    size_t base = size_t(group.x) * hidden;
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += 256u) {
        float v = float(input[base + i]);
        partial += v * v;
    }
    threadgroup float sums[8];
    float subtotal = simd_sum(partial);
    if (lane == 0) sums[sg] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    float rms = rsqrt(total / float(hidden) + eps);
    // Exact in-place input/output is safe: stats have completed, each thread
    // reads/writes its own channels, and gamma is disjoint from every output.
    for (uint i = tid; i < hidden; i += 256u) {
        float v = float(input[base + i]);
        output[base + i] = f16_sat(v * rms * float(gamma[i]));
    }
}
