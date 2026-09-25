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
