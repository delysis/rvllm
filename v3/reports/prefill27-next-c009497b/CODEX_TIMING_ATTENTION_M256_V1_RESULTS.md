# PR27 M=256 local-attention paired operator timing

**Rejected by the prospectively declared 5% all-sample drift gate. No speed
result is accepted.** The eight fresh dependent ABBA–BAAB queue jobs
`prefill27-attn256-abba-v1-20260928-t00-m256` through `t07-m256` each
terminated `succeeded`, exit 0, with unchanged inputs, sampled-condition
eligibility and no violations or overrun. Independent review of all eight
outer condition journals found every sample ready, on AC power, `pmset`
mode 2, low-power mode off, nominal thermal state 0, and no sampled
competitor. The frozen safe-Rust adjudicator revalidated the retained
correctness screens, complete outputs, independent numerical oracle,
fragment probe, raw hashes and all 72 measured GPU samples, then returned
`timing-screen-rejected-drift` (exit 1). No job was replayed, omitted or
overwritten.

A is incumbent `metal-prefill-q4k16`; B is proposed
`metal-prefill-mma8k32`, both for `attention-local` at exactly M=256. The
retained correctness screens passed and produced the same complete BF16
output hash on the shared M=256 structured fixture. The process medians
below are **descriptive only**, in milliseconds, in sealed run order:

| Job | Arm | Median GPU ms |
| --- | --- | ---: |
| t00 | A | 1.722708 |
| t01 | B | 0.693250 |
| t02 | B | 0.684542 |
| t03 | A | 1.706458 |
| t04 | B | 0.694333 |
| t05 | A | 1.719000 |
| t06 | A | 1.713000 |
| t07 | B | 0.681625 |

The descriptive A/B ratio is 2.490943 overall, 2.488886 in ABBA and
2.494262 in BAAB. This is **not a qualified speed comparison**. The
between-process median drift was 0.9523% for A and 1.8644% for B, within
the 5% limit, but all-sample drift was **15.8853% for A and 13.9185% for
B**, exceeding it. The complete 72 observations—not a favorable subset—are
retained in the failed adjudication. The apparent operator advantage cannot
advance to the separately required confirmation or production selection.

The exact prospective protocol is
`CODEX_TIMING_ATTENTION_M256_V1_PROTOCOL.md`; sealed config SHA-256
`5bfad8bdbb1ed5c9116615ec6ff8c1a303d65127742c15c1ca4a7196cb2c3636`;
generated adjudication plan SHA-256
`afaad2f6ea88d06239f7d68bca3eb2a8cd1dc6c944b7bb9e364b876e2e94b85d`;
frozen timing referee SHA-256
`7803ce924420b729a03f2a6878a8dd71ad5e0a531fe181aac5530cd35516f116`.
The failed adjudication is
`timing-attention-m256-v1-20260928/timing-attention-m256-v1-adjudicated-01.json`,
SHA-256
`3c26a966c39317d67b88687260e085cd3bd459c1cdd08f89423ede39be44a33d`.
The complete 40-file outer queue archive is
`prefill27-attn256-abba-v1-queue-results.tar.gz`, SHA-256
`b3a19c42c8381e9785544697e68d6dfb7cddd64b62de7c1938f2af9193cd6137`.
The complete 32-file compact sample JSON archive is
`prefill27-attn256-abba-v1-sample-json.tar.gz`, SHA-256
`ec3f233d33ff31e965738a7948051d293b69a510b454eb541a5462df24e8e873`.
Original binary readbacks and stdout/stderr remain untouched in the
individual `timing-attention-m256-v1-20260928/samples/` directories; they
are not included in the compact JSON archive.

This is one synthetic **operator** fixture, not full-route or MLX timing.
It does not establish a thermal cause for drift, checkpoint quality,
production speed or promotion. The separately rejected M=256 down-norm
timing run remains rejected. PR27 attention's one-BF16-ULP differences at
larger M remain preserved in their own correctness reports; this M=256
timing does not resolve their first arithmetic cause.
