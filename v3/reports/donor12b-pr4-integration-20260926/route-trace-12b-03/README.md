# BF16 SG8 global-attention route observation at decode position 512

This is a diagnostic comparison of stored tensors, **not** an independent
numerical oracle, timing measurement, model-quality qualification, or promotion.
The route-observation feature is default-off. Capturing synchronizes after each
selected layer and may change scheduling, although it supplies no trace scratch
to kernel selection. The two runs used the same 12B-it checkpoint, 512-token
prompt case, executable, metallib, and trace positions. Both serial-queue jobs
exited 0, sampled conditions were eligible, and each reported zero inference
library/pipeline compiles. Both generated `[236770, 236770]`.

The on route recorded 16 `research_donor12b_sg8_global_attention` dispatches;
the off route recorded none. Both recorded 80 local-attention, 96 native-gate,
and 288 native-projection donor dispatches. These are complete-case counts, not
per-layer GPU durations.

The earlier `route-trace-12b-01` and `-02` captures correctly hashed raw u16
buffers but incorrectly decoded BF16 values as F16 in printed statistics. They
remain preserved; do not use their printed numerical values. This pair explicitly
records `float_type: bf16`, decodes the sampled values as BF16, and seals full
raw bits for attention output and final residual at the selected decode token.
`rvllm_metal_trace_compare` verifies raw-bit SHA-256 against each trace before
comparing.

| Layer | On/off stored-buffer observation |
| --- | --- |
| 4 | All captured field digests equal. |
| 5, first selected global layer | Q/K/V and physical-prefix KV-cache digests equal. Attention output differs at 1 of 8,192 BF16 elements: index 2,428 is `b755` on versus `b754` off, or −0.000012695789337158203 versus −0.000012636184692382812 (absolute difference 5.960464477539063e−8). The complete 3,840-element stored residual after this layer is bitwise equal. |
| 6 | Physical-prefix KV and FFN digests differ. The attention output differs at 358 of 4,096 elements and the stored residual at 1,730 of 3,840; maximum finite residual difference is 0.046875. |

Thus the **first observed stored-field difference** is one BF16 attention-output
element in layer 5. It does not prove the first arithmetic difference, a defect,
or that this bit caused the later layer-6 divergence: the trace is post-layer,
not a complete dataflow record, and no independent FP64/reference attention
result was compared for that exact real activation. The layer-5 residual's
bitwise equality makes a direct causal story especially premature. A repeat
capture, complete consumed-state inspection/teacher-forced logits, and a
same-input independent reference are required before interpreting quality or
changing the production selector.

The source gives a bounded list of **candidate arithmetic sites**, not an
attribution. The selector-off D512 fallback in `kernels.rs` accumulates each
512-element Q·K dot in one thread in dimension order, scans visible KV tokens
serially with online softmax and `exp`, then stores the weighted sum using a
reciprocal multiply. `donor12b_common.metal` partitions each Q·K across a
32-lane SIMD group with `simd_sum`; eight groups scan strided KV positions,
each maintaining online statistics with `precise::exp`, then merge those
statistics in group order and store `value / z`. BF16 compilation changes the
incumbent storage type and final store, but does not make these FP32 reduction
orders identical. Any one or several of dot order, KV scan/merge order,
exponential implementation, and final divide-versus-reciprocal rounding could
produce a one-ULP BF16 difference. The retained trace lacks full Q and active
K/V values, so it cannot test these candidates against an independent
same-input reference. Do not alter donor arithmetic based on this source list
or treat bitwise equality of the stored layer-5 residual as proof that the
subsequent layer-6 state is equal.

The route trace's KV summaries hash a contiguous physical prefix of each
cache buffer (`kv_cache_rows * kv_dim` from its base offset), not a logical
token sequence gathered through the block table. Therefore the layer-6 KV
digest difference alone does **not** establish that an attention kernel read
different logical history. Layer-6 Q/K/V values immediately after projection
and normalization have matching digests; its attention output differs. A
same-input reference needs the page table, valid logical length and the
actual visible K/V entries, not just these physical-prefix hashes.

There is an existing safe-Rust route for the **KV portion** of that next
experiment. `KvPageIo::capture_page` on `ModelMetalBackend` returns exact native
two-byte K/V payloads for all layers in a physical page, in layer/K/V order.
It refuses access while a submitted Metal ticket still owns the arena; the
direct session can call it after `collect`, before launching the next decode
step. The direct one-sequence session in this trace constructs handoffs with
empty block tables, for which `materialize_block_tables` assigns identity
physical pages in logical order. A future capture must nevertheless record
the *materialized* table and context length, select only live logical rows,
and use the attention source layer's K/V for any shared-KV layer. The existing
ANE prefill export also calls `capture_page`, but converts BF16 to F16, so its
converted values are not an exact BF16 referee input. Neither safe page API
captures the active Q tensor or an intermediate layer boundary; a complete
same-input attention reference still needs a separately justified,
route-preserving Q producer. This is a source-level implementation path, not
new device evidence or an arithmetic attribution.

Jobs: `g4-donor-route-trace-12b-bf16-on-03` and
`g4-donor-route-trace-12b-bf16-off-03`. The trace executable SHA-256 is
`eec866e924c5e3f053b2b6374e67f9156c0ac187a3697b68d4f6153ad4e8f12b`;
the comparator executable SHA-256 is
`af5ab58166d4e744f43f298f3d3c2f7810a11b346fd7411641d0f596ef481e57`.
The complete small queue archive (`queue-results.tar.gz`) contains manifests,
condition samples, stdout/stderr, and reports; SHA-256:
`10130f958a01002af92c89c5efd0603e06a0fec8a4d5388499ebd385b611ded8`.
The layer-5 trace SHA-256 values are `13b4f9adfba2a0aadafa5588bcd2a26e30a992da41e10793027cd208f163864b`
for on and `7e4f40a077ade60e2f7bd0d02a7e8a6e78b41a5a07036873913d1c0613ec8764`
for off. All six trace JSONs and both inference reports are retained here.
