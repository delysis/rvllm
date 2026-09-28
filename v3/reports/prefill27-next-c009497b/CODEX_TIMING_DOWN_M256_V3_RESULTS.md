# PR27 M=256 down-norm paired operator timing (2026-09-28)

**Rejected by the predeclared 5% drift gate. No speed result is accepted.** The eight fresh dependent ABBA–BAAB queue jobs `prefill27-down256-abba-v3-20260928-t00-m256` through `t07-m256` each terminated `succeeded`, exit 0, `files_unchanged: true`, `sampled_conditions_eligible: true`, `overdue: false`, and no violations. The frozen safe-Rust referee revalidated the retained correctness fixtures and all 72 measured GPU samples, then returned `timing-screen-rejected-drift` (exit 1). It wrote the complete failed adjudication to `timing-down-m256-v3-20260928/timing-down-m256-v3-adjudicated-01.json`; no job was retried or omitted.

A is old `metal-prefill-pipeline32x64`; B is new `metal-prefill-wide64`. Both produce the same complete output and FP32 raw-intermediate SHA-256 on the shared M=256 fixture. The process medians below are descriptive only, in milliseconds, in the predeclared run order:

| Job | Arm | Median GPU ms |
| --- | --- | ---: |
| t00 | A | 2.507500 |
| t01 | B | 3.220625 |
| t02 | B | 3.195667 |
| t03 | A | 2.445875 |
| t04 | B | 3.164833 |
| t05 | A | 2.401875 |
| t06 | A | 2.431000 |
| t07 | B | 3.387750 |

The descriptive A/B ratio is 0.760077 overall, 0.772000 in ABBA and 0.737553 in BAAB. These ratios point to the proposed B operator being slower on this one fixture, but **are not a qualified speed comparison**. The 5% stability rule failed for B's inter-process median drift (7.0436%) and all-sample drift (22.0617%); A's all-sample drift also failed (6.1179%), though A's inter-process median drift was 4.3976%. In particular the last B process increased within its nine samples; it is retained, not discarded. All 72 observations and all eight medians are in the failed adjudication and the compact sample archive.

The exact protocol is `CODEX_TIMING_DOWN_M256_V3_PROTOCOL.md`; base `c009497bee8125d456e4aee26f0b35836d79c231`; config SHA-256 `2590974603a1394961505e57b1c4b1be2572602647fa73e1810c2811d085995d`; adjudication plan SHA-256 `1788facb27b5114e65052ad83d6b8ae9971a76e3930489fffe24e9cfe362c8c6`; frozen timing referee SHA-256 `7803ce924420b729a03f2a6878a8dd71ad5e0a531fe181aac5530cd35516f116`. Failed adjudication SHA-256 `7dec9eb0f97920df9bd8ce0888fafc84e463475d0431945fd7f8ee09454a86ca`. The complete 40-file outer queue receipt archive is `prefill27-down256-abba-v3-queue-results.tar.gz`, SHA-256 `0e70627024e8043725cb5d7bb85bc82f8405f1e1c033202ee0a8a5e9ac52de13`; the 32-file per-sample JSON archive is `prefill27-down256-abba-v3-sample-json.tar.gz`, SHA-256 `1a258160eaac16caa89c291d53f5e8a4c43fd6ed9a0c1f03104c35ac1acd4157`. The original binary readbacks remain unmodified under `timing-down-m256-v3-20260928/samples/` and are not included in those compact archives.

This is one synthetic down-norm **operator** fixture. It does not establish full-model performance, a thermal cause for the drift, production speed, or promotion. The predeclared failed timing run is preserved; there is no same-ID retry or favorable subset verdict.
