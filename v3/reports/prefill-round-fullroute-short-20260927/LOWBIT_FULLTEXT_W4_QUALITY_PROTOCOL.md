# Prospective full-text-projection W4 continuation diagnostic

This protocol is sealed while the one-package W4 preparation job
`prefill26-lowbit-fulltext-w4-package-build-v1-20260927` is running. No
inference job has been submitted. Package success and identity must be
verified before creating any immutable inference manifest. Do not use this
protocol to replay a prior job or to reinterpret the one-layer W4 trial.

The two source cases in `lowbit-fulltext-w4-quality-source-v1.json` are
**Codex-authored synthetic diagnostics**, not a human-authored or independent
held-out corpus. Its SHA-256 is
`ad075af3147a6c242aed94f0c2f5ac40845c09b38c2929983be5dbd234d5b80a`.
The unchanged safe-Rust tokenizer fixture builder source SHA-256 is
`bcfcc431e39aa9939d07741fe67335cf9c7a8af3fb0f8f398e4594c36200cdbf`.
It verified exact prefix tokenization with the original tokenizer SHA-256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The sealed `lowbit-fulltext-w4-quality-tokens-v1.json` SHA-256 is
`f4899669393b9726953a698eaa3049819de2d83822e67416d891b511e1cff65c`.

| Case | Prompt tokens | Forced targets | First target |
| --- | ---: | ---: | ---: |
| `estuary-sensor-v1` | 218 | 21 | 107 |
| `textile-catalog-v1` | 220 | 22 | 107 |

The first target in each case is a newline token. All 43 targets, not just
the first, must be retained. Supplied targets in `generated_token_ids` are
transport for teacher forcing, **not** sampled model agreement. Compare
`teacher_forced.steps[].sampled_token_id`, target rank and target NLL at
every position, plus per-case NLL sum/mean and candidate-minus-control
differences with both signs. Report the first sampled-greedy difference and
all rank changes. No favorable-position selection or new threshold after
results are visible.

If the W4 package is validated, generate four fresh dependent jobs through
the existing serial queue: native BF16 donor and authenticated full-text W4
for the estuary case, then native and W4 for the textile case. Pin the exact
original model/tokenizer/source/fixture, package manifest and assets,
frozen inference executable/source, metallibs, prompt and target IDs. The
native control must use the same donor SG8 BF16 library as the package.
Require a clean terminal result, unchanged pins, exact prompt/target IDs and
actual named W4 low-bit **decode** dispatch for each candidate, with no
unexplained native fallback. The frozen CLI does not separately count generic
low-bit prefill kernels, so neither prefill kernel attribution nor speed
follows from its named decode ledger. Per-step logit readback invalidates
all trial timing.

Before submission, implement and test a separate fail-closed safe-Rust
offline referee for these four manifests. It must verify package identity,
queue conditions, all steps and aggregates, and preserve failed or
unfavorable receipts. The prospective gate is diagnostic only: all-text
projection W4 leaves embeddings and multimodal tensors BF16, two synthetic
cases are not a checkpoint-wide corpus, and the BF16 donor is not an
independent numerical oracle. Token agreement or lower NLL alone does not
qualify quality, speed or promotion.
