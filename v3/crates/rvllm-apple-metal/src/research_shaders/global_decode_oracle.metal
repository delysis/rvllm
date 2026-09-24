// Test-only, independent serial GPU oracle. Append to the SAME generated core
// source; never included by a normal family export or timing library. The CPU
// oracle separately checks this FP32 reduction schedule against FP64 dots.
kernel void global_decode_serial_oracle(
    device const ushort *q [[buffer(0)]], device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]], device float *output [[buffer(3)]],
    device const int *table [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *positions [[buffer(6)]], constant GlobalDecodeParams &p [[buffer(7)]],
    constant uint &panel_size [[buffer(8)]], device float *sampled_dots [[buffer(9)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= 16u || p.sequences != 1u || p.heads != 16u || p.kv_heads != 1u
        || p.head_dim != 512u || p.window != 0u || p.scale != 1.0f
        || p.block_size == 0u || p.max_blocks == 0u || p.num_blocks == 0u
        || (panel_size != 64u && panel_size != 128u)) return;
    int context = contexts[0], position = positions[0];
    if (context <= 0 || size_t(context) > size_t(p.max_blocks) * p.block_size
        || position < 0 || position >= context) return;
    uint end = uint(position) + 1u;
    for (uint b = 0; b <= (end - 1u) / p.block_size; ++b) {
        if (table[b] >= 0 && uint(table[b]) >= p.num_blocks) return;
    }
    float u[512];
    for (uint d = 0; d < 512u; ++d) u[d] = 0.0f;
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint t = 0; t < end; ++t) {
        int page = table[t / p.block_size];
        if (page < 0) continue;
        size_t base = (size_t(page) * p.block_size + t % p.block_size) * 512u;
        float score = 0.0f;
        for (uint panel = 0; panel < 512u; panel += 64u) {
            float parts[32];
            for (uint lane = 0; lane < 32u; ++lane) {
                float part = 0.0f;
                for (uint d = lane; d < 64u; d += 32u) {
                    part = fma(as_type<float>(uint(q[head * 512u + panel + d]) << 16u),
                        as_type<float>(uint(k[base + panel + d]) << 16u), part);
                }
                parts[lane] = part;
            }
            for (uint delta = 16u; delta > 0u; delta /= 2u) {
                for (uint lane = 0; lane < delta; ++lane) parts[lane] += parts[lane + delta];
            }
            score += parts[0];
        }
        // Diagnostic-only raw QK values. No sampled-dot buffer or branch exists
        // in the actual candidate or timing library.
        if (head == 0u || head == 7u || head == 15u) {
            uint h = head == 0u ? 0u : (head == 7u ? 1u : 2u);
            uint tokens[3] = {0u, end / 2u, end - 1u};
            for (uint j = 0; j < 3u; ++j) {
                if (t == tokens[j]) sampled_dots[h * 3u + j] = score;
            }
        }
        float next = max(maximum, score);
        float correction = denominator == 0.0f ? 0.0f
            : (maximum == next ? 1.0f : precise::exp(maximum - next));
        float weight = score == next ? 1.0f : precise::exp(score - next);
        denominator = fma(denominator, correction, weight);
        for (uint d = 0; d < 512u; ++d) {
            u[d] = fma(weight, as_type<float>(uint(v[base + d]) << 16u), u[d] * correction);
        }
        maximum = next;
    }
    float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint d = 0; d < 512u; ++d) output[head * 512u + d] = u[d] * inverse;
}
