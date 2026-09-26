// Explicit storage ABI: do not rewrite half to bfloat in this source.
inline float round_decode_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}
inline ushort round_decode_rne(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort(bits >> 16u) | ushort(0x40u);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
// Fixed tree, no dependence on implementation-specific simd_sum association.
inline float round_decode_sum32(float x) {
    for (ushort delta = 16; delta > 0; delta /= 2) x += simd_shuffle_down(x, delta);
    return simd_broadcast(x, 0);
}
inline float round_decode_gelu(float x) {
    if (x >= 5.0f) return x;
    if (x <= -5.0f) return 0.0f;
    return 0.5f * x * (1.0f + tanh(0.7978845608f *
        (x + 0.044715f * x * x * x)));
}
