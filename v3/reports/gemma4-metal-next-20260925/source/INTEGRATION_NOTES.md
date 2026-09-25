# Integration notes for current rvLLM campaign

## Priority order

1. **Port the r8/sg2 row mapping into the existing winning W4-down and W8-output storage format.** This avoids introducing a new quantizer merely to test scheduling. Keep the existing arithmetic oracle unchanged.
2. **Add the BF16 fused gate+up->GELU control.** It changes no checkpoint precision and tells us whether the materialization/launch boundary is worth attacking before quantization is involved.
3. If fusion wins, create the same fused kernel over the existing W4/W8 format. Only then consider affine group-64 repacking.
4. Add short-context global-unsplit as an explicit selector candidate, not a replacement for split-32.
5. Stop broadening the attention search until normal-route evidence says attention has become dominant again.

## Why the existing prefill kernel family is unlikely to close 10-19x alone

The older rvLLM MMA candidate is already a respectable 32x32x32 simdgroup-matrix implementation with register prefetch. A 10-19x end-to-end deficit is too large to plausibly explain as one tile-size mistake. The system still needs fewer materializations, better low-bit matrix paths, and larger resident/fused regions. Treat Steel-style matrix machinery and future TensorOps as a separate prefill backend rather than mutating decode QMV into a fake universal kernel.

## Suggested new candidate IDs

- `metal-qmv-r8sg2-w4-down-m1`
- `metal-qmv-r8sg2-w8-output-m1`
- `metal-ffn-gateup-gelu-bf16-m1`
- `metal-ffn-gateup-gelu-w4-m1`
- `metal-global-d512-g16-unsplit-short`

Keep candidate identity tied to role + shape + storage, not just shader entrypoint.
