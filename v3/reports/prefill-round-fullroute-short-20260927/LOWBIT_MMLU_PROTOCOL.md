# Predeclared one-layer W4/W8 sidecar diagnostic

The existing authenticated original-12B-it BF16 packages replace only
`model.language_model.layers.0.mlp.down_proj.weight` with a group-32 W4 or
W8 sidecar. Their manifest SHA-256 values are respectively
`1b9c43598644619432589b66836ed22e52d1e48c968e4b93134f21bd5e324949`
and `1cb72e93d101f23f969b422586bb63058bca0bb23aa80c27e87accff1d162a69`.
The packages are local ignored artifacts in the donor12b PR6 worktree, not
portable files checked into Git. The package opener checks every manifest
asset's checksum, model semantics and sidecar contents before inference.
The W4 packed values/scales hashes are
`5f25d4d760c4adf0a1e6581a024e3eea2e4b729f85144162c9998c27dbdee00e`/
`95b97ae914375a61e1e2b4848da79af4f7b6446bf6e0962981cd16da993713c5`;
the W8 hashes are
`b885c80ac1f7152998d3e29917f1fd060348d3fd6ab5b1bbf448867a8d5d0842`/
`1e4bd6dace99ed4ab6cca745ba42f3a64ac662e1a143eea8b71c9c3f14d41175`.
Both package BF16 libraries match the native donor SG8 library SHA-256
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.

For each of the two previously frozen MMLU prompts and target sequences in
`MMLU_NATURAL_PROTOCOL.md`, run the same pinned direct inference executable
SHA-256 `23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463`
with donor SG8 native BF16, one-layer W4 and one-layer W8, serially in that
order: `logic-native`, `logic-w4`, `logic-w8`, `cs-native`, `cs-w4`, `cs-w8`.
Each arm uses the exact original tokenizer, prompt IDs and forced target IDs;
the native route uses the same donor SG8 BF16 Metal library as each package.
The safe-Rust `rvllm_gemma4_lowbit_teacher_job_gen` validates both frozen
MMLU template hashes, the executable, source library and both full Apple
packages before emitting immutable six-job manifests into a new directory.
Refuse an existing output directory. Submit via the existing serial queue
with zero thermal-stability dwell; record conditions and retain failures.

Require actual W4/W8 sidecar dispatch, not a native-weight fallback, over
prefill and forced decode. The donor SG8 selector may decline a low-bit
prefill shape and use the existing BF16 low-bit schedule; record the named
research and per-role low-bit counts separately instead of calling that a
native fallback. Preserve every target NLL, rank and sampled greedy ID;
`generated_token_ids` are supplied targets, **not** sampled agreement.
Compare both signs of all per-position changes to the native donor SG8 route.
The fail-closed `rvllm_gemma4_lowbit_teacher_summary` checks the exact six
frozen manifests against the queue copies, clean terminal receipts, named
SG8 low-bit decode dispatch, exact source-tokenized prompts and targets, and
all finite step scores. The frozen inference executable does not expose the
generic low-bit prefill dispatch count separately; this experiment must not
claim that count or prefill kernel selection from its named decode ledger.
Do not use teacher-readback timing. A one-layer quantization perturbation,
even if numerically benign, does **not** qualify a whole-W4/W8 checkpoint,
establish an independent numerical oracle or authorize promotion. The two
MMLU question targets are subject to the Codex-authored wrapper and unknown
training contamination stated in `MMLU_NATURAL_PROTOCOL.md`; this is not an
MMLU score. Full-checkpoint sidecars, calibrated quality criteria, independent
same-boundary reference and paired speed evidence remain separate gates.
