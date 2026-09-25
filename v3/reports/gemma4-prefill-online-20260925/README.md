# Gemma 4 Metal prefill online-softmax candidate

This packet is preparation only. No accelerator command was run and no shared
queue state was changed. The candidate is default-off and is not connected to
`layer_forward`.

The conventional arm has explicit external BF16 QKV and O boundaries, FP32
online-softmax state, 64-key panels, causal/window/page-hole handling, and a
source identity contract. The independent Rust FP64 oracle covers 1 and
63/64/65 and 255/256/257 key boundaries, first/middle/last holes, malformed
shapes, untouched guards, and repeat equality.

TensorOps is explicitly deferred: this checkout has no stable queried Metal
TensorOps ABI. GPU-family inference is intentionally insufficient and there is
no ANE claim or silent fallback.

Run the 256-token manifest only after compiling the exact source and device
referee. Advance in order to 512, 1024, and 2048 only if the preceding receipt
passes correctness, untouched guards, bitwise repeat, generated-code identity,
and the declared initial speed gate. QKV, attention, and O timings must be
reported separately.
