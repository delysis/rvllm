# Gemma 4 Metal next packet: integration and first device adjudication

Date: 2026-09-25
Status: integrated as default-off research candidates; device-correct at the stated boundaries; not production-routed or promoted.

## Provenance

- Input archive: `/Users/george/Downloads/gemma4-metal-next.zip`
- Input archive SHA-256: `83a82f9c134f5d7060fa34460131e540dd05806b7f8ff014ad59c0128debbca3`
- Original per-file hashes are preserved by the unchanged documentation and manifest plus the source deltas below.
- The three supplied shaders did not compile. Each mixed a scalar `threadgroup_position_in_grid` input with a vector `threads_per_threadgroup` input, which Metal rejects. The only source repair changes `threads_per_threadgroup` from `uint3` to `uint` and preserves the exact 1-D launch checks.
- Repaired source SHA-256:
  - QMV: `4a303839c1ff3367c31499e041abadf20fd500442f2dd88d211010b4b10b625c`
  - fused FFN: `c30a109a5969632adf60744579f7ceb52294d961c360946e2a5e7930091a5bde`
  - short attention: `32c3d3afcacf36b377e81f454320c717ff95c21fc1fa89531187b753655320c0`
- Combined metallib SHA-256: `d6d42dec984fe8bc37075520bad0e1218d74b8e04059dc4336e442e71f73f743`

## Device evidence

All five entry points compiled with Metal 3.1, linked, loaded, and formed compute pipeline states on the Apple M4 Max. Public pipeline properties and raw compiler artifacts are sealed under `evidence/`.

| Candidate | Device oracle | Guards | Repeatability | Disposition |
|---|---|---|---|---|
| Q4 affine group-64 QMV | independent nonzero CPU reference, N=17 K=64 | pass | bit-exact | viable scheduling/storage prototype; needs group-32 inner-loop port or an explicit authenticated repacker |
| Q8 affine group-64 QMV | independent nonzero CPU reference, N=17 K=64 | pass | bit-exact | same boundary as Q4 |
| Q4 fused gate+up→GELU | exact production shape M=1 K=3840 I=15360, sparse nonzero independent reference | pass | bit-exact | viable fusion prototype; checkpoint-quality and real-weight gates still required |
| BF16 fused gate+up→GELU | exact production shape M=1 K=3840 I=15360, sparse nonzero independent reference | pass | bit-exact | strongest immediate integration candidate because it changes no checkpoint precision |
| global D512/GQA16 short unsplit | independent softmax/PV reference at L=3 | pass | bit-exact | algorithmically correct for its contiguous-KV ABI; blocked from production routing until ported to paged-cache semantics |

The milliseconds in `oracle.json` are diagnostic single-dispatch observations, not benchmark results. They were not collected with ABBA/BAAB, sufficient warmup, incumbent pairing, or a real checkpoint and must not be used as speedup claims.

### Persistent-queue replay

On 2026-09-26 the same five-case live-device referee was submitted three times, sequentially, through the existing persistent experiment queue. All three jobs terminated `succeeded` with exit code 0, unchanged declared inputs, no violations, and all five cases `passed` on every replay. The queue sampled and retained the ambient host state rather than waiting for a pristine machine; the SAM audio-restoration process was active during these runs. These are independent process-level correctness/repeatability replays, not timing qualification.

| Queue job | Report SHA-256 | Trial stdout SHA-256 | Result |
|---|---|---|---|
| `g4-metal-next-device-correctness-r1-20260926` | `e03ae95255fc81fc8fb3e59d6dec8565d4bbb619672f34c3a67c182d2f1876ee` | `d2fab24fc557f106213d32145c2bbce8550948d5e6476d2155251d75b1b7b676` | succeeded; 5/5 passed |
| `g4-metal-next-device-correctness-r2-20260926` | `74061418bfc56feb0012abbabcfecdd4c99d2ec3d69de875e1dc22fbe3274c98` | `087f25edf2b595450dc5708f624274003ce1982f3b58d038ec2c6bc428850c63` | succeeded; 5/5 passed |
| `g4-metal-next-device-correctness-r3-20260926` | `0d85c3beec646be3e73ddebe4275ba89e88e8823ecdd6f8ff9274fe67f45adec` | `c3b5afd3afee37ce855b48f0b8beb3f0d6116dfaec9d705f52c0d31030aedc6b` | succeeded; 5/5 passed |

The submitted manifests and complete nonempty queue outputs are sealed under `jobs/` and `queue-results/`. Each source-to-copy SHA-256 was checked after capture. `trial.stderr` was empty for all three jobs and is intentionally not represented as a nonempty artifact.

## What is and is not integrated

Integrated:

- repaired and identity-sealed source packet;
- offline AIR/metallib and raw public-tool evidence;
- a safe-Rust live-device referee that exercises every kernel, full production FFN dimensions, nonzero arithmetic, repeated execution, and trailing guards;
- default-off research status.

Not integrated:

- no selector or production default changed;
- no new checkpoint weight format was silently introduced;
- no contiguous-KV attention shader was wired into the paged cache;
- no real-weight, logit/perplexity, ABBA/BAAB, full-route, or MLX comparison claim is made.

## Next tournament arms

1. Port the r8/sg2 QMV mapping onto rvLLM's already-qualified group-32 W4/W8 sidecar ABI. Compare it directly with the current role-specific winners for W4 down and W8 output.
2. Wire the BF16 fused FFN control as a default-off exact-shape research route and run real layer-0 correctness, then ABBA/BAAB against separate gate/up plus GELU.
3. Only if BF16 fusion wins, port the fused kernel to the existing group-32 low-bit package. Keep affine group-64 as a separate repacker experiment.
4. Port short unsplit attention onto the existing paged-cache/newest-KV/hole/rollback contract before any timing campaign. Screen at L=256, then advance only if it beats the stable route.

## Reproduction

```sh
cargo build -p rvllm-apple-metal --bin rvllm-metal-next-oracle --release
target/release/rvllm-metal-next-oracle \
  reports/gemma4-metal-next-20260925/compile/gemma4-metal-next.metallib
```

The referee's result is checked in as `oracle.json`; compiler and pipeline receipts are under `evidence/`.
