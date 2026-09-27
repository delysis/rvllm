# Donor SG8 component and prefill screen 01

This is a **single-pass diagnostic screen**, not paired qualification or
production promotion. The existing serial experiment queue ran all 16
real-weight BF16 Gemma 4 12B jobs successfully with observed condition
strata eligible and `stable_seconds: 0`. The 256-token prompt, two generated
tokens, model path, SG8 metallib, and inference executable were held fixed;
the candidate selector, diagnostic component mask, and explicit prefill
options are in each queued manifest in the evidence archive. All arms returned generated IDs
`[236770, 236770]`, reported zero inference compiles, and completed without
an output failure. This does **not** establish 64-step numerical equivalence
or checkpoint-quality acceptance.

The queue-owned executable SHA-256 is
`bf149a5bfd2835fe8668008988f24b69f0331526750cff5712340112e9ec3dc5`;
the SG8 metallib SHA-256 is
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
The manifests pin the prompt file and model `config.json`, but do not by
themselves seal every checkpoint weight shard. This binary was built before
the subsequent diagnostic-mask numeric-ABI fingerprint repair and must not
qualify imported/shared KV handoff. These jobs used direct sessions only.

## Decode: component selection actually dispatched

| Arm | Two-token decode ms | Donor physical dispatches (global/local/gate/projection) |
| --- | ---: | ---: |
| all | 139.68 | 16 / 80 / 96 / 288 |
| none | 615.64 | 0 / 0 / 0 / 0 |
| projection only | 518.05 | 0 / 0 / 0 / 384 |
| gate only | 567.85 | 0 / 0 / 96 / 0 |
| local attention only | 523.93 | 0 / 80 / 0 / 0 |
| global attention only | 333.72 | 16 / 0 / 0 / 0 |
| all except projection | 199.34 | 16 / 80 / 96 / 0 |
| all except gate | 145.71 | 16 / 80 / 0 / 384 |
| all except local attention | 226.84 | 16 / 0 / 96 / 288 |
| all except global attention | 416.42 | 0 / 80 / 96 / 288 |
| none repeated | 748.02 | 0 / 0 / 0 / 0 |
| all repeated | 145.07 | 16 / 80 / 96 / 288 |
| selector off | 703.09 | 0 / 0 / 0 / 0 |

The global D512 attention change is the leading explanation: removing it
from all-on increased observed decode 139.68 → 416.42 ms, and adding it
alone reduced 615.64 → 333.72 ms against the earlier none arm. The source
contrast is direct: selector-off D512 does not meet the incumbent
`head_dim <= 256` online predicate, so `attention_decode_f16` launches **one
thread per head** and serially scans KV tokens and 512 elements. SG8 launches
eight SIMD groups/256 threads per head for strided online-softmax scans and
a within-group merge. This screen confirms actual SG8 global dispatches;
it does not measure their isolated GPU durations or a stable causal fraction.

The component effects are not additive. Disabling the fused donor gate causes
96 additional native-projection dispatches (384 versus 288), and controls
drifted materially: none was 615.64 then 748.02 ms, while all was 139.68
then 145.07 ms. Thus a single-arm subtraction cannot be promoted to a
component timing claim. `none repeated` and true `selector off` were
reasonably close at the tail, but have not been proved numerically identical
at all internal boundaries. Next use paired ABBA/BAAB blocks at 256 and
2048 contexts for global attention, then local/projection/gate as warranted;
retain complete source, work-count, and dispatch evidence. Long-context
global attention should be tested on identical Q/K/V and full-route states.

## Prefill: a large opt-in lead, not yet per-kernel attribution

| Arm | Prefill ms | Two-token decode ms |
| --- | ---: | ---: |
| all-on SG8, default prefill options | 11,564.86 | 139.68 |
| all-on SG8, repeated | 12,826.67 | 145.07 |
| selector off | 16,233.44 | 703.09 |
| SG8 + `prefill_mma32` | 3,133.33 | 144.14 |
| SG8 + `prefill_simd_attention` | 14,255.41 | 144.94 |
| SG8 + both prefill options | **2,464.17** | 140.76 |

Unlike `rvllm_disaggregated_infer`, the `rvllm_metal_infer` binary used in
the earlier donor screens leaves both prefill options off unless explicitly
selected. The generated BF16 shader contains native `simdgroup_matrix<bfloat>`
MMA32 code with FP32 accumulators; this is not a BF16→FP16 adaptation. The
large total-prefill reduction when MMA32 is enabled, and the near-unchanged
decode, make dense projection/GEMM routing the strongest *current*
prefill-bottleneck hypothesis. The existing route policy makes eligible
256-row Gemma 4 projections use MMA32 when selected, but this receipt has no
per-kernel dispatch ledger or valid GPU timestamps. It cannot allocate the
saved milliseconds among QKV, FFN, attention, host overhead, or interactions.

For orientation, the older non-interleaved MLX BF16 baseline reports
128.774 prompt tokens/s at 256 tokens, about 1.99 seconds for 256 tokens.
The 2.46-second combined arm is therefore within an exploratory ~1.24x of
that baseline versus the earlier ~5.8–8.2x gap. Model-file identity,
prompt-token identity, host conditions, and operation-level work were not
matched across frameworks; this is **not MLX parity**. A queue-owned
512-token, two-token ABBA prefill-off/on/on/off screen has been submitted
separately, followed by longer numerical and per-kernel checks if warranted.

The 16 raw queue receipts, each retaining input hashes, conditions,
stdout/stderr, and backend report, are in `queue-results.tar.gz` (SHA-256
`04cf463650179bb0b2ceabfea528591f619e4784776ca9b9f947b4a4f90c9f52`).
No thermal-stability wait, retry, measurement pruning, or automatic
promotion was performed.
