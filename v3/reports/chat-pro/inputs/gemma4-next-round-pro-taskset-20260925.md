# Gemma 4 Apple kernel program: next implementation round

Remain in ordinary Chat; do not switch to Work.

Work directly from PR #4, branch `astra/gemma4-load4-tiles-20260923`, at or
after commit `149f0cbd`. Produce reviewable source patches or a complete source
archive, not merely recommendations. Do not claim Apple-device performance you
cannot measure. Preserve safe Rust, exact dispatch evidence, independent
correctness oracles, and the existing kernel-game admission boundaries.

## Current evidence

- Global D512 BF16 decode attention has moved from a scalar 396.938 ms at
  2,048 live keys to a correctness-qualified split-matrix 2.196 ms. The closest
  measured MLX BF16 operator observation is 1.489 ms. A cooperative split-32
  candidate has observations as low as 0.983 ms, but material run-to-run
  variance prevents promotion.
- A same-referee tournament is now being run at 256/512/1024/2048 between
  cooperative split-32 and split-matrix, including complete partial-plus-merge
  time. Treat its eventual receipts as authoritative; do not invent a winner.
- Native-BF16 experimental Metal projection entry points now exist for
  W4ABF16 and W8ABF16. They retain BF16 activations/output, FP16 group-32
  scales, and FP32 accumulation. Real layer-0 Q projection correctness passes
  at M=1 and M=4, but timing is highly variable. A seven-role Q/K/V/O/gate/up/
  down campaign is being staged.
- The ANE baseline INT8 and stacked-FFN routes have exact cached programs,
  zero-compile inference, and exact 10-token correctness. A 12-arm
  ABBA/BAAB/ABBA timing sequence is running. Early pairs are directionally
  favorable to stacked FFN but are not a final result.
- MLX isolated-stage evidence says FFN dominates PP4096, while FFN, attention,
  QKV and O projection are all material in long-context decode. The captured
  MLX trace and generated Metal evidence are diagnostic, not a speed oracle.

## Deliverable A: production-quality Metal low-bit kernels

Implement materially faster native-BF16 W4 and W8 projection candidates for
all seven Gemma 4 dense roles, supporting decode M=1 and bounded prefill tiles.
The current one-output-per-SIMD implementation is only a correctness baseline.
Design proper tiled kernels with cooperative loading, vectorized packed-weight
decode, scale reuse, multiple outputs per SIMD group, and explicit tail
handling. Keep W4 and W8 separate where their optimal instruction/memory
schedule differs. Maintain FP32 accumulation and one BF16 storage rounding
boundary.

Provide at least two credible tile/schedule candidates per bit depth, with
resource calculations, exact dispatch geometry, rejected-shape behavior,
guard tests, repeated-use tests, generated MSL identity, and source-level
tests proving BF16 and FP16 ABIs cannot alias. Integrate candidates into the
research catalog and queue without changing production defaults.

## Deliverable B: decode attention selector and overhead reduction

Audit the cooperative split-32 and split-matrix implementations. Explain the
short-context launch/merge overhead from the actual code. Implement a bounded
selector hypothesis with explicit crossover inputs, but do not select it in
production without receipts. Seek a low-overhead single-command-buffer path at
256/512 and preserve the high-throughput cooperative/split path at 1024/2048.

Address newest-K/V visibility without CPU serialization. The oracle must cover
lengths around partition boundaries, first/middle/last holes, rejected
dispatches, untouched buffers, guards, and bitwise repeated execution. Include
complete split-plus-merge time in every comparison.

## Deliverable C: separate prefill work

Implement a distinct Gemma 4 prefill attention experiment. Do not extrapolate
the decode fusion policy to prefill. Compare a conventional tiled online
softmax arm with a hardware-dependent TensorOps arm when supported. Make
unsupported hardware an explicit deferred result, not an ANE claim or silent
fallback. Include QKV/O boundaries so fusion claims are measurable.

## Deliverable D: device-resident autoregressive loop

Design and implement the smallest vertical slice that keeps decode dependency
state, newest K/V, sampling state, and bounded token batches device-resident.
Expose command-buffer token batching as an explicit bounded policy. Instrument
submission, GPU wait, attention, projections, FFN, vocabulary and host work
outside the hot path. Existing output must remain unchanged when
instrumentation is disabled. Provide tests proving disabled instrumentation
does not add per-layer allocations, synchronization or logging.

## Deliverable E: ANE 4-bit/8-bit/16-bit framework

Audit the current static INT8, stacked-FFN, LUT4 and BF16/Metal boundaries.
Return concrete safe-Rust patches for the next ANE candidates, prioritizing
graph fusion and launch reduction rather than speculative private APIs. Make
compile-cache identity, zero-compile inference, graph availability, exact
route, evaluation count and fallback absence first-class receipt fields.
Separate 4-bit storage/quality experiments from claims of native 4-bit ANE
arithmetic.

## Deliverable F: checkpoint-quality gates

Add a checkpoint-specific quality referee for each W4/W8 packaging contract.
It must bind checkpoint/config/tensor/quantizer identities; test representative
logits or perplexity against BF16; retain all failures; and distinguish
operator numerical agreement from model-quality acceptance. Choose thresholds
from a documented calibration process rather than convenience.

## Required output

Return:

1. a concise findings and risk report;
2. a complete patch series or source archive based on the stated commit;
3. a manifest listing every changed file and its purpose;
4. exact host tests and Apple-device commands Codex should run;
5. queue manifests for every new candidate, with short-context screening and
   advancement to 512/1024/2048 only after correctness and initial speed gates;
6. explicit statements of what remains unmeasured or unsupported.

Do not weaken strict JSON parsing, identity sealing, independent confirmation,
zero-compile requirements, route qualification, guard checks, or failed-receipt
retention. Do not modify shipping defaults. Do not fabricate benchmark results.

