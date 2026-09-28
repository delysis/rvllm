# Prospective full-text-projection W4 continuation diagnostic

This protocol was sealed before the one-package W4 preparation job
`prefill26-lowbit-fulltext-w4-package-build-v1-20260927` completed. That job
subsequently succeeded with clean queue conditions; the authenticated package
manifest SHA-256 is
`abcd5b043322efe1756b6ac81e805f0aee9da5b04c9e10a77ae391532f026ce3`.
The builder's separate read-only validation also succeeded, but is not an
independent validator implementation. No inference job has yet been submitted.
Do not use this protocol to replay a prior job or reinterpret the one-layer
W4 trial.

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
actual named W4 low-bit **decode** dispatch for projection, gate/up and QKV
(`research_donor12b_sg8_w4`, `research_donor12b_sg8_gate_w4`, and
`research_donor12b_sg8_qkv_w4`) for each candidate. Require the donor SG8
attention dispatch and reject any W8 dispatch. Retain all native-dispatch
counts; their interpretation requires a source-bound explanation rather than
an assumption that every native count is fallback. The frozen CLI does not
separately count generic low-bit prefill kernels, so neither prefill kernel
attribution nor speed follows from its named decode ledger. Per-step logit
readback invalidates all trial timing.

Before submission, implement and test a separate fail-closed safe-Rust
offline referee for these four manifests. It must verify package identity,
queue conditions, all steps and aggregates, and preserve failed or
unfavorable receipts. The prospective gate is diagnostic only: all-text
projection W4 leaves embeddings and multimodal tensors BF16, two synthetic
cases are not a checkpoint-wide corpus, and the BF16 donor is not an
independent numerical oracle. Token agreement or lower NLL alone does not
qualify quality, speed or promotion.

The separate safe-Rust generator source SHA-256 is
`2bb3ef19e24e129103fd70fdcf26560bfebc18e39dd0eb851f74dc1fadc24a74`.
It authenticated the exact package and sealed four fresh dependent jobs and
two prompt JSONLs in `lowbit-fulltext-w4-quality-v1-queue/`. The immutable
manifest SHA-256s in order estuary-native, estuary-W4, textile-native,
textile-W4 are `ecee5e07697c9797c55510f8aec3084538a2c55ddd707a2c2ffd4d18e21e3f66`,
`048907938601caa94b0655fe6d2833be0ad45b074bd0656ecdc6b4415e5a0e61`,
`3aaa30f95d374ca1cce6e0c8af340bea687097a1859b6a1f247ae139db354643`,
and `4a7d23367f99addcd342cf8dd81cc6a48a27efefd48320d28fbb0625cd4ce0f4`.
W4 manifests pin all 675 package files, including 656 low-bit sidecar files,
plus source and experiment inputs. The frozen referee source SHA-256 is
`b2ff574ef41b6959057605bf9f488bd1c2b12b05584e0f873a7d3aae9fefb738`;
its five focused tests include all four tracked manifest hashes and negative
dispatch, score, and seal cases. Generation and tests are host-side only;
submission and all numerical observations remain pending.
