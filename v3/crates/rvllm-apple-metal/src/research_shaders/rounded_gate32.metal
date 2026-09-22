// metal-rounded-gate32: shared A tile for separate gate/up accumulators.
// Storage roundings BEFORE GELU and AFTER GELU*up are intentional contracts.
kernel void research_rounded_gate32(
    device const half *A [[buffer(0)]], device const half *W [[buffer(1)]],
    device half *activated [[buffer(2)]], device half *gate_up [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &H [[buffer(5)]],
    constant uint &I [[buffer(6)]], constant uint &write_gate_up [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[32 * 32];
    threadgroup half gt[32 * 32];
    threadgroup half ut[32 * 32];
    threadgroup float cg[32 * 32];
    threadgroup float cu[32 * 32];
    uint mr = group.x * 32u, nc = group.y * 32u;
    uint sm = uint(sg / 2u) * 16u, sn = uint(sg % 2u) * 16u;
    simdgroup_float8x8 g00(0.0f), g01(0.0f), g10(0.0f), g11(0.0f);
    simdgroup_float8x8 u00(0.0f), u01(0.0f), u10(0.0f), u11(0.0f);
    for (uint kb = 0; kb < H; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u, k = kb + index % 32u;
            at[index] = mr + row < M && k < H ? A[size_t(mr + row) * H + k] : half(0.0f);
            gt[index] = nc + row < I && k < H ? W[size_t(nc + row) * H + k] : half(0.0f);
            ut[index] = nc + row < I && k < H ? W[size_t(I + nc + row) * H + k] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a0, a1, g0, g1, u0, u1;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(g0, gt + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(g1, gt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(u0, ut + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(u1, ut + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(g00, a0, g0, g00);
            simdgroup_multiply_accumulate(g01, a0, g1, g01);
            simdgroup_multiply_accumulate(g10, a1, g0, g10);
            simdgroup_multiply_accumulate(g11, a1, g1, g11);
            simdgroup_multiply_accumulate(u00, a0, u0, u00);
            simdgroup_multiply_accumulate(u01, a0, u1, u01);
            simdgroup_multiply_accumulate(u10, a1, u0, u10);
            simdgroup_multiply_accumulate(u11, a1, u1, u11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(g00, cg + sm * 32u + sn, 32);
    simdgroup_store(g01, cg + sm * 32u + sn + 8u, 32);
    simdgroup_store(g10, cg + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(g11, cg + (sm + 8u) * 32u + sn + 8u, 32);
    simdgroup_store(u00, cu + sm * 32u + sn, 32);
    simdgroup_store(u01, cu + sm * 32u + sn + 8u, 32);
    simdgroup_store(u10, cu + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(u11, cu + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u, col = nc + index % 32u;
        if (row < M && col < I) {
            // +0 matches the alpha=1,beta=0 standalone GEMM epilogue,
            // including its signed-zero boundary. Do not move either cast.
            half rounded_gate = f16_sat(cg[index] + 0.0f);
            half rounded_up = f16_sat(cu[index] + 0.0f);
            if (write_gate_up != 0u) {
                gate_up[size_t(row) * (2u * I) + col] = rounded_gate;
                gate_up[size_t(row) * (2u * I) + I + col] = rounded_up;
            }
            activated[size_t(row) * I + col] =
                f16_sat(gelu_tanh(float(rounded_gate)) * float(rounded_up));
        }
    }
}
