# Source audit: MLX versus rvLLM timing boundaries

This audit narrows the interpretation of the existing same-checkpoint,
exact-token MLX comparison. It does not add a timing sample or alter any
completed queue job. The prior M=304 standalone rvLLM ABBA screen failed
its predeclared 5% drift gate; its descriptive ratios remain unaccepted.

The pinned MLX-LM `mlx_lm/generate.py` has SHA-256
`4a3bf57f5679dac73791f069268866ab5b9336e58582d436b31133984e4b6971`.
In `generate_step` (lines 456–469), MLX processes the remaining prompt,
forms and evaluates the first sampled token before yielding it. It also
enqueues a following step before that first yield, though asynchronous
execution prevents assigning its device work to a phase from these source
lines alone. In `stream_generate` (lines 722–727), the prompt timer ends
when that first token is yielded. Its `prompt_tps` therefore includes the
first sampled-token boundary, whereas rvLLM's `prefill_ms` timer in
`rvllm_metal_infer.rs` (lines 1907–1926) ends after collecting a prefill
operation that returns **no** sampled token. rvLLM's first sampled token is
obtained in a separate decode step (lines 1958–2000).

MLX resets its generation timer at the first token (line 727) but reports
`generation_tps = (n + 1) / elapsed` at later yields (lines 743 and 757).
Thus the 64-token reported rate uses a count including the first token,
whose completion preceded that timer. In contrast, rvLLM's `decode_ms`
spans all 64 collected decode steps. The existing Python wrapper
`v3/tools/mlx_gemma4_exact_prompt_bench.py` (SHA-256
`d65d434aea7cfff954fe5fca45581dd59f51be3afc027053064146898708bee8`)
derives MLX prompt nanoseconds from `prompt_tps` and retains
`generation_tps`; it does not reconcile either boundary.

These are **definite source-level accounting differences**, not a measured
correction factor. Asynchronous MLX work, detokenization, and process
boundaries preclude simply moving one token's time between columns. The
reported M=304 ~3.226× prompt and ~6.727× decode MLX advantages remain
planning-grade orientation, further limited by noninterleaving, the rvLLM
drift failure and MLX's queue timing-ineligible power sample. Matching
checkpoint and generated token IDs does not make the two phase timers
equivalent.

A future strict cross-framework experiment should prospectively specify
one shared observable start/end boundary—preferably warm, complete
fixed-length request wall time—then interleave fresh independent processes
at several prompt lengths with exact pinned model, library, executable,
prompt IDs, generated-token count and outer queue conditions. Its separate
prefill/decode phase claims must wait for equally defined synchronization
boundaries in both implementations. Use new immutable job IDs and preserve
all failed receipts; do not replay the completed M=304 runs or use this
source audit as a speed verdict.
