# Gemma 4 Metal prefill online-softmax candidate

This packet contains sealed `rvllm.experiment_job.v1` device-referee jobs. They
have not been submitted or run, and no shared queue state was changed. The
candidate is default-off and is not connected to the shipping selector.

The conventional arm has explicit external BF16 QKV and O boundaries, FP32
online-softmax state, 64-key panels, causal/window/page-hole handling, and a
source identity contract. The independent Rust FP64 oracle covers 1 and
63/64/65 and 255/256/257 key boundaries, first/middle/last holes, malformed
shapes, untouched guards, and repeat equality.

TensorOps is explicitly deferred: this checkout has no stable queried Metal
TensorOps ABI. GPU-family inference is intentionally insufficient and there is
no ANE claim or silent fallback.

The 256-token manifest pins a release test executable and strict release
validator by SHA-256. It JIT-compiles and dispatches the candidate and incumbent
SIMD control, then validates the independent FP64 comparisons, tails, holes,
guards, repeatability, source/executable identity and public PSO resource
fields. The 512, 1024, and 2048 jobs form a strict predecessor chain.

## L256 screen result

The L256 job passed its correctness and receipt validator, but the candidate did
not pass the speed gate. It was slower than the existing SIMD control in 12 of
14 cases. The exact 256-token boundary was 1.138x faster, but its adjacent
255- and 257-token cases were only 0.914x and 0.985x as fast as the control.
Longer built-in screens were also losses: 652-token sliding was 0.910x and
652-token global was 0.804x. The 512/1024/2048 jobs are therefore not submitted.
This conventional tiled arm is rejected as a general prefill candidate; the
receipt is retained as a correctness-qualified negative result.
