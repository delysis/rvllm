// metal-rms-simd32: post-projection RMS statistics in FP32, real gamma.
// A separately named reduction-order experiment. No storage rounding removed.
kernel void research_rms_simd32(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    device const half *gamma [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (hidden != 3840 || tokens < 6 || tokens > 1024 || eps != 1.0e-6f ||
        group.x >= tokens || group.y != 0 || group.z != 0 || threads.x != 32 ||
        threads.y != 1 || threads.z != 1) return;
    const ulong base = ulong(group.x) * hidden;
    float sum = 0.0f;
    for (uint d = lane; d < hidden; d += 32) {
        const float value = float(input[base + d]);
        sum += value * value;
    }
    const float inverse = rsqrt(simd_sum(sum) / float(hidden) + eps);
    // Each lane reads/writes only its own elements. Exact in-place use is
    // legal; partial overlap and overlap with gamma are rejected by the host.
    // No threadgroup scratch, cross-lane memory communication, or blanket
    // assumption that SIMD lockstep is a replacement for a memory barrier.
    for (uint d = lane; d < hidden; d += 32) {
        const float value = float(input[base + d]);
        output[base + d] = f16_sat(value * inverse * float(gamma[d]));
    }
}
