# Gemma 4 MLX-shaped Metal decode candidates — 2026-09-25

Status: **default-off research code; not performance-qualified and not a shipping selector**.

Branch: `chat/gemma4-mlxlike-gemv-qmv-20260925`

## Why this round is different

The current campaign has largely closed the isolated long-context attention gap while full-route decode remains far behind MLX. The remaining hot path is dominated by dense projections / FFN and by dispatch boundaries. This round therefore stops spending the primary optimization budget on attention tiles and attacks the M=1 projection geometry directly.

The key reference is the current MLX Metal implementation at commit
`073d2252c96754e9e57ed6633be3f8ecb7058548`:

- `mlx/backend/metal/kernels/gemv.h`: the Gemma 4 BF16 decode trace selects a geometry equivalent to four SIMD groups per threadgroup, four output rows per SIMD group, and four contiguous K values per lane.
- `mlx/backend/metal/kernels/quantized.h::qmv_fast_impl`: two SIMD groups per threadgroup, four output rows per SIMD group, and two packed words per lane. That is 16 W4 values or 8 W8 values per lane.

rvLLM's incumbent dense `gemm_f16_vec8` instead uses 256 lanes, an 8×256 FP32 threadgroup reduction buffer, and repeated threadgroup barriers. The existing low-bit N4/N8/vector families use only one SIMD group per output tile and do substantially more scalar scale/weight work.

## Candidate A — native-BF16 `research_decode_gemv_mlx16`

Source: `crates/rvllm-apple-metal/src/research_shaders/decode_gemv_mlx16.metal`

Geometry:

- M=1 only.
- 128 threads = four SIMD groups.
- four output rows per SIMD group = 16 outputs per threadgroup.
- four contiguous K values per lane = 128 K values per SIMD iteration.
- FP32 accumulation and SIMD-only reduction.
- no threadgroup scratch and no threadgroup barriers.
- one BF16 rounding boundary at output.

Admission is intentionally exact and covers only the Gemma 4 12B dense decode shapes:

- QKV family: `N=8192|9216, K=3840`
- gate/up combined route: `N=30720, K=3840`
- output/down family: `N=3840, K=4096|8192|15360`

It is wired through the existing `MetalResearchCandidate` / normal projection dispatch path as
`metal-decode-gemv-mlx16`. Missing PSO, wrong storage, wrong scale, FP32-output QKV-prefill, near-miss shape, or resource failure falls back to the incumbent route.

## Candidate B/C — group-32 W4/W8 `mlx-qmv`

Source: `crates/rvllm-apple-metal/src/research_shaders/low_bit_qmv_mlx.metal`

These retain rvLLM's existing package format exactly: symmetric group-32 quantization, FP16 scales, BF16 activation/output, FP32 accumulation. They do **not** silently adopt MLX's affine group-64 format, so any speed difference is attributable to execution geometry rather than a different quantizer.

Shared geometry:

- 64 threads = two SIMD groups.
- four output rows per SIMD group = eight outputs per threadgroup.
- activation mini-vector loaded once and reused across four output rows.
- SIMD-only reduction; no threadgroup scratch/barrier tree.

Format-specific inner loop:

- W4: 16 K values per lane, 512 K values per SIMD iteration.
- W8: 8 K values per lane, 256 K values per SIMD iteration.
- one scale load per output row / lane / iteration, reused across the mini-vector.

The real-checkpoint referee exposes this explicitly as `--candidate mlx-qmv`; existing adaptive/production selectors are unchanged.

## Required gates

Do not promote any of these from source inspection.

1. Linux/general compile and unit tests.
2. macOS MSL compile + queried PSO limits.
3. Component correctness against independent CPU/reference output, including repeatability and guard bytes.
4. Real-checkpoint M=1 screening for q/k/v/o/gate/up/down roles.
5. Paired ABBA/BAAB timing against both the incumbent candidate and the current role winner.
6. Normal-route dependent-token correctness and timing for `metal-decode-gemv-mlx16`.
7. Full-route comparison with matched token count/context. Only after that should a selector change be proposed.

## Commands

From `v3/`:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test -p rvllm-apple-metal --lib
cargo test -p rvllm-apple-metal --features low-bit-real-weight-research
cargo build -p rvllm-apple-metal --release --features low-bit-real-weight-research --bin rvllm-low-bit-real-weight
```

Example real-weight screen (use the campaign's sealed model path and exact tensor names):

```bash
target/release/rvllm-low-bit-real-weight \
  --model-dir "$MODEL_DIR" \
  --tensor language_model.model.layers.0.mlp.down_proj.weight \
  --m 1 \
  --format both \
  --samples 9 \
  --candidate mlx-qmv \
  --order abba
```

Repeat with the independently counterbalanced order and with each supported projection role. Preserve raw receipts; do not infer model quality from operator timing.

For the dense candidate, run the existing normal-route research harness with:

```bash
RVLLM_METAL_RESEARCH=metal-decode-gemv-mlx16 ...
```

The first meaningful comparison should be M=1 projection/operator timing versus `gemm_f16_vec8`, followed immediately by dependent-token normal-route timing. There is little value in another long attention tournament before these projection candidates are adjudicated.
