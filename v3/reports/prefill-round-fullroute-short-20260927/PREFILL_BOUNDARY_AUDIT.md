# M304 prefill-final versus decode-replay boundary audit

This is a numerical diagnostic for the original BF16 `google/gemma-4-12B-it`
checkpoint, not a speed test or a production promotion. Both immutable queue
jobs are terminal, and their complete compact receipts are retained below.

## Why the boundaries differ

The inference driver sends all 304 prompt tokens through the ordinary prefill
route. It then sends the final prompt token again through the first decode
route at position 303. The prefill route does not normally project final
logits; the first decode route does. The default-off
`--teacher-prefill-last-logits` research probe projects all 304 final prefill
rows and scores only row 303 before the decode replay. This adds an LM-head
command buffer and readback; none of its timing fields are performance
evidence. The source for these steps is
`v3/crates/rvllm-runtime/src/bin/rvllm_metal_infer.rs` and
`v3/crates/rvllm-runtime/src/apple_metal_backend.rs`.

The two logits readbacks use the same `metal_finalize_logits_blocking` entry
point, but with shapes `M=304` and `M=1`. The Metal final-logits path applies
final RMSNorm, an LM-head GEMM, and softcap/argmax. Its GEMM selector depends
on M: a 304-row vocabulary projection does not qualify for the small-M
cooperative-vector or batch-eight predicates, while the one-row projection
does qualify for the vector predicate unless another earlier selector accepts
it. Thus a prefill-final versus post-decode difference could involve the
underlying residual/attention route, the LM-head arithmetic/rounding path, or
both. This source audit does **not** isolate a first arithmetic error.

## Comparison plan

The pinned independent CPU/Hugging Face one-step reference evaluates the exact
304 prompt IDs and target token 107 on the original checkpoint. Its target
NLL is 0.4660174578126135. The CPU environment uses Transformers 5.14.1,
whereas the checkpoint config identifies 5.10.0.dev0; numerical-boundary
equivalence is not established.

The two new queue jobs are `prefill26-boundary-m304-off-20260927` and
`prefill26-boundary-m304-combined-20260927`. Both pin the separate research
executable, CLI source, 304-token input, original safetensor, CPU reference,
and their exact Metal libraries. Compare the prefill-final row, subsequent
decode-replay row, and each earlier **unprobed** teacher readback. Require
terminal queue success, unchanged inputs, actual candidate dispatch and full
condition receipts before making a numerical finding. Equal post-decode
results would show that this particular probe did not visibly change that
one-step result; it would not make the prefill-final probe a production route
or an independent tensor reference.

## Terminal result

Both immutable jobs succeeded with exit code zero, unchanged pinned inputs,
eligible sampled conditions and no violations. Their sampled stratum was AC
power, power mode 2, thermal state 0, low-power mode off. No stability wait was
used. Both used exactly the same 304 prompt IDs, original model file, research
executable and target 107. The control's research dispatch was empty. The
combined candidate actually dispatched 48 GEMMs, 48 QKV projections, 96 raw
projections, 96 raw normalization kernels, 40 D256 and eight D512 Q4K16
attention kernels. Both generated token 107 with target rank one.

| Boundary / target 107 | Independent CPU/HF | Metal control | Metal combined |
| --- | ---: | ---: | ---: |
| Full-prompt final row: target logit | 14.9375 | 14.5625 | 14.9375 |
| Full-prompt final row: target NLL | 0.46601746 | 0.69456873 | 0.63686904 |
| First decode replay: target logit | not separately measured | 14.625 | 14.8125 |
| First decode replay: target NLL | not separately measured | 0.68074801 | 0.65161510 |

The candidate's prefill-final target logit equals the CPU target logit exactly
at BF16 reporting precision, but its target NLL is still higher by 0.17085158.
The control's corresponding excess is 0.22855127. One selected logit cannot
explain a full-vocabulary distribution, so this is not a numerical parity
finding. The candidate-minus-control NLL difference is -0.05769969 at the
prefill-final boundary and -0.02913292 after decode replay. Within the
control, replay lowers NLL by 0.01382072; within the candidate, replay raises
it by 0.01474606. The opposite directions are diagnostic, not proof that
either attention, projection or LM-head arithmetic is wrong.

The new probe's **entire first post-decode teacher step** is identical to the
respective earlier unprobed teacher job, for both control and candidate;
the prompt IDs and first generated token also match. Thus this specific
one-step post-decode observation was not visibly perturbed by the inserted
prefill-final readback. The result does not prove that later continuation,
scratch contents, timing or internal tensors are unperturbed.

The two job manifests pin CLI source SHA-256
`abc49131e49e2dff5f1e6aa6c83e4542d375dd79f906cf2bc57eecaf0cf9309d`,
separate research executable
`a228847fe667e5aad50de1b8c36b96744cb4636809ac687fa2fc9fe055f6b705`,
model safetensor
`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`,
and normal/combined metallibs respectively
`bb88c9667b4b7ac3758760b602abe40cbe72cc907bf3c40e2f2a2d33b0bd8df2`
and `8a1234a3ac14cb1c55c6c4c601c4fb2647defbc955b14be3a2c23414552fbfcb`.
The complete ten-file queue archive `boundary-m304-queue-results.tar.gz` has
SHA-256 `ec4a981d87706c8b2a9348d32360041956ff56ea81c9a528ddabc74b815f3ea8`.
The separate CPU/HF reference and its earlier failure receipts are retained
in this report directory; they were not rerun.

The remaining experiment is to isolate whether the shared CPU-versus-Metal
distribution gap first appears in final hidden state, final normalization,
LM-head projection or softcap, while keeping the selected production route
unchanged. A direct same-residual M304-versus-M1 LM-head check would also
separate shape-dependent projection rounding from the prefill/decode model
path. None of these data establish checkpoint-wide quality or speed.

## Predeclared same-residual LM-head shape trial

The next default-off Rust diagnostic requests logits in a fixed
`M=1 → M=304 → M=1` sequence **after the ordinary prefill and before decode**.
It compares row zero from all three passes. Row zero reads the same residual
and model weights; the intervening full-row pass changes the LM-head GEMM
shape. The repeated one-row pass checks whether the full-row readback visibly
changes the later one-row result. The program records bitwise-equal and
different vocabulary counts, nonfinite mismatches, maximum and mean finite
absolute logit differences, and the token ID with maximum finite difference.
It also retains the prior row-303 prefill-final target score and one-step
decode score. Every timing field in these jobs is invalid as speed evidence.

This can establish whether an M-dependent LM-head result exists on row zero;
it cannot automatically transfer that finding to row 303 or identify the
first differing hidden-state operation. A positive difference might involve
the M-dependent LM-head arithmetic or an unobserved readback side effect;
the short/full/short repeat helps check the latter but does not prove all
scratch state unchanged. A negative row-zero difference does not exonerate
tail-row arithmetic.

The two immutable queue IDs are
`prefill26-lm-head-shape-m304-off-20260927` and dependent
`prefill26-lm-head-shape-m304-combined-20260927`. They pin the original M304
prompt, model and metallibs, new CLI source SHA-256
`c091c7069ad563271ae2d46361190c0f8f451f521abfe41d7d30b749c4fbdf04`,
and separate release executable SHA-256
`23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463`.
Feature host tests passed 17/17 and default host tests 11/11; six ignored
feature device tests were not run. The manifests are
`lm-head-shape-m304-off-job.json` and
`lm-head-shape-m304-combined-job.json`. No device result existed when this
trial was specified. Retain failed and successful receipts without replay.
