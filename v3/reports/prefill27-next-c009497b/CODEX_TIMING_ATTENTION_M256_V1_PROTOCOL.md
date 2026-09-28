# PR27 M=256 local-attention operator timing protocol

This protocol was written before any timing job or output for this comparison.
It does not reuse the failed M=256 down-norm timing jobs. The two component
screens named below already passed as part of the immutable correctness ladder;
both passed the independent FP64 oracle, the fragment-layout/product probe,
the masked-NaN-V and negative fixtures, and full output checks. Their shared
M=256 fixture has matching complete BF16 output hashes. Higher-M one-BF16-ULP
differences remain recorded in the ladder reports and are not erased by this
M=256 timing trial.

The sealed config is `codex-timing-attention-m256-v1-config.json`, SHA-256
`5bfad8bdbb1ed5c9116615ec6ff8c1a303d65127742c15c1ca4a7196cb2c3636`.
A is `metal-prefill-q4k16`; B is the proposed `metal-prefill-mma8k32`;
role is `attention-local`; M is exactly 256. The retained screens are
`prefill27-first256-20260927-s02-m256` and
`prefill27-first256-20260927-s03-m256`. The config pins the already built
strict Metal libraries and supplied test harness, and the isolated safe-Rust
timing referee executable. No new Swift driver will be authored.

The frozen timing referee SHA-256 is
`7803ce924420b729a03f2a6878a8dd71ad5e0a531fe181aac5530cd35516f116`;
the supplied test harness SHA-256 is
`eb9cbdabac712c0006b5b096e3163272d1467ee44f3a20cd99d4c88bd8080fba`.
The two retained screen JSON SHA-256s, A then B, are
`71e16b7b58a0b1930747f7c4021170d4ed8b74f16ef798228a38b0b35439ccf9`
and `aefffc3f895fed04bb30bc3f25c3e79d2aa291d63ac8fc4f37ff6a72cf5044cd`.
The generator will additionally pin the compiled libraries and all retained
screen cells before any queue submission.

The unchanged frozen safe-Rust generator accepted that config and created
`timing-attention-m256-v1-20260928/` without invoking Metal. Its adjudication
plan SHA-256 is
`afaad2f6ea88d06239f7d68bca3eb2a8cd1dc6c944b7bb9e364b876e2e94b85d`.
The eight create-new job manifest SHA-256s, in ABBA-BAAB order, are:

| Job suffix | Arm | Manifest SHA-256 |
| --- | --- | --- |
| t00 | A | `24e0c29e9a70183890aff6928d57121bb45a61bd2fd2792e866479b0323a0650` |
| t01 | B | `61641944db2daf367bc54a738205e29a166a0b7273c3e62550cab06a7445a836` |
| t02 | B | `dcf5a626b67c35277382158295739227a566d3dc867bf2090ffd3fea34fb95ab` |
| t03 | A | `44f22dda97290970184e230e2bb674c45792e81c99a3e6ae35d9a9ab13b9f0d9` |
| t04 | B | `cb644c3222c7479148b1defcaa3cd1f2666dddb2a837acda6e3c2f3ea8f55a02` |
| t05 | A | `a3fc58e97389fa518aa232be2cd0646d24e550696ce600d4fd4c2d3dbe2e41b8` |
| t06 | A | `bfa427065648806a997aef44d8fcdeaf979046af1fda2df3e2dc7dc35c0a297e` |
| t07 | B | `316a413198e4362c8aa5c88237980e5c0454d5bec76038d255514c652bbdb98a` |

All eight have `purpose: exploratory_timing`, `stable_seconds: 0`, a
64-GiB free-space guard, a 3600-second run limit and 56 pinned inputs
including both complete retained screen JSONs/cells. Their sample output
directories were unused at generation. No job had been submitted when these
identities and the decision rules were sealed.

At approximately 2026-09-28 02:33 UTC, after the independent W4 quality
chain had finished and the serial queue was waiting, all eight exact job
manifests were accepted once by the existing queue in order. The measured
result remains pending; an accepted submission is not an operator speed
result. Do not submit any of these IDs again.

Only after the existing full-text W4 serial quality chain is terminal, and
after verifying the output directory is absent and there is at least 64 GiB
free, generate and review eight **fresh** serial queue manifests. Use the
config's unique `prefill27-attn256-abba-v1-20260928` campaign and immutable
ABBA-BAAB order A,B,B,A,B,A,A,B. Each queue job must retain one warmup
process's 20 warmups and nine measured whole-operator GPU samples, exact
source/library/driver/referee and retained-screen pins, all raw output and
oracle checks, sampled conditions, and a unique output directory. Use zero
thermal-stability dwell; record conditions without waiting for a desired
thermal state. Never replay a terminal ID or overwrite a receipt.

The frozen safe-Rust adjudicator must first revalidate all retained numerical
fixtures, raw output hashes and complete sample receipts, then retain all 72
GPU observations, eight process medians, both order-specific ratios and both
within-arm drift definitions. Nonfinite/nonpositive timers, changed inputs,
missing dispatch, incomplete oracle or fragment checks fail closed. Both
inter-process median drift and all-sample drift must be at most 5% for both
arms; all eight outer queue receipts must separately be terminal successful,
unchanged, eligible and violation-free before any qualified operator timing
description. A performance lead additionally requires old/new ratio >1.05
in **both** order blocks and a separately predeclared confirmation. A stable
slower candidate can be described as such, but it is not a lead. If the gate
rejects, retain the failed adjudication and every sample, without selecting
a favorable subset or replaying the same jobs.

This is one synthetic **operator** at M=256, not full-route or MLX timing.
It cannot establish production speed, checkpoint quality, the first SG8
arithmetic cause, or promotion. The previously failed down-norm timing result
remains failed and independent.
