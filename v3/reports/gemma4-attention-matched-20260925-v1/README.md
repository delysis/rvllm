# Matched D512 global-decode tournament

This campaign compares `atlas-coop-key-r8-k8-p64-t128-s32` against the exact
two-dispatch `metal-global-d512-split-mma_r8k32s256t128` family at 256, 512,
1,024, and 2,048 live keys.

Each queue job is one sealed `trial`: it compiles the frozen source, runs the
independent full FP64/F32/BF16/repeat/guard/negative-metadata oracle for both
arms, then runs nine ABBA blocks. Candidate and control each include their
partial and merge dispatch in one command-buffer GPU interval. No source
compilation occurs during timed samples. The queue records power, thermal, and
listed process observations but has `stable_seconds=0` and no thermal,
low-power, power-mode, quiet-process, or idle-server gate.

The MLX file is deliberately a separate-process evidence join. It does not
describe MLX as an ABBA arm and cannot establish a promotion ratio.

Reproduction commands from the v3 root:

```sh
cargo test -p rvllm-apple-metal --features attention-atlas-research attention_atlas --lib
cargo build --release -p rvllm-apple-metal --features attention-atlas-research --bin rvllm-attention-atlas
target/release/rvllm-attention-atlas prepare reports/gemma4-attention-matched-20260925-v1/specs/split32-vs-split-matrix-l256.json "$PWD/reports/gemma4-attention-matched-20260925-v1/prepared-l256"
target/release/rvllm-attention-atlas queue reports/gemma4-attention-matched-20260925-v1/requests/trial-l256.json reports/gemma4-attention-matched-20260925-v1/jobs/trial-l256.json
target/release/rvllm_experiment_queue submit "$PWD/reports/gemma4-global-decode-local-20260924/queue" "$PWD/reports/gemma4-attention-matched-20260925-v1/jobs/trial-l256.json"
```

Repeat the final three commands for 512, 1024, and 2048. Generated prepared
directories and queue results are write-once; existing receipts are never
overwritten.

## First completed queue pass

These historical receipts are bound to runner SHA-256
`17040e8ad1fae2678c7d2ef9f987713117121f8f406db270d0addce8a73a4467`
and generated Metal source SHA-256
`072efb760d38b88756981d122b0456832639c668ff48454a047f06b38cab56b1`.
The later referee strengthening changed the runner; it is not evidence about
what this earlier executable checked.

All four jobs compiled the frozen source, passed the common independent oracle,
and retained all 36 timed samples (nine ABBA blocks). Both arms report three
encoded dispatches per sample: metadata validation, partial attention, and
merge. The 5% outer-control drift gate admitted only L1024; the other scores are
retained inconclusive observations rather than discarded or retried samples.

| Live keys | Split-matrix control | Cooperative split-32 | Control / candidate | Drift | Disposition |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 3.427 ms | **0.704 ms** | 4.87x | 11.77% | inconclusive drift |
| 512 | 2.046 ms | **0.517 ms** | 3.96x | 5.20% | inconclusive drift |
| 1,024 | 1.830 ms | **0.748 ms** | **2.45x** | 0.34% | admitted |
| 2,048 | 1.948 ms | **1.297 ms** | 1.50x | 6.47% | inconclusive drift |

Every point estimate favors cooperative split-32. Relative to the pinned,
separate-process MLX BF16 observations, split-32 is approximately 2.17x slower
at L256 and 2.15x slower at L512, while its L2048 point estimate is 1.15x
faster. Those are engineering estimates only. The MLX L1024 value is a retained
40.05 ms interference outlier and is not used to claim a win.

The first-pass executable checked full FP32/F64 and once-rounded BF16
correctness, repeatability, guards, and negative metadata for the cooperative
candidate and BF16 accuracy for the matched control. The control family also
has its earlier standalone full split-matrix oracle. Current uncommitted referee
source additionally repeats the control's FP32, repeatability, once-rounding,
and negative-metadata checks in the same trial. No receipt in this directory
claims to have exercised those later checks.
