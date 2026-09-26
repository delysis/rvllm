# Donor12B SG8/SG4 exact-base screen — 2026-09-26

## Identity and scope

The supplied `rvllm-593e1d6f-donor12b.patch` (SHA-256
`62158cd1dacc4495204eab9c65fc4fc3aa8a6eccb84db6c0a1ee3b8123b214f9`)
applied cleanly to `593e1d6fb25088608198f2248dcac038d19fe761`. The
formatted, source-only validation commit is `7de19128444a07a52546b1c3edc63f9f37121a3d`.
This is **not** the PR #4 integration tree. The two selectors remain default-off.

Apple M4 Max (Apple9), Xcode Metal 3.1, `-fno-fast-math`. Both strict library
builds passed. SG8 source/metallib/build-receipt SHA-256:
`49f5fb616a5c8db9689896b2e97c8636c67eee5227703245498a0f6ed291b13e` /
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06` /
`cfa9bf75ed315887887e45594453315b0fe006693b8c5083c8a0ff9dab9b6a62`.
SG4: `c9ee195a84710489e662f7fedd9996154359493b92290ac51faca471b46ec219` /
`94372f6c3af50e537d30758fc0d6742363174718dc3cd541a7f2aa61ee7298b0` /
`8c4f3ce0083a7a5b628b924a1d0e8ff734d59ea7d44b0f339239c80d91cef8e5`.
The pinned native test executable SHA-256 is
`f33ed1371cffef919f9b41cc989dfd79701f2012e61dcf5de708374ea10c39c4`.

## Gates completed

- Offline locked `rvllm-apple-metal --all-targets` check and Apple-feature
  runtime library check passed; 173 Metal library tests passed, 29 ignored.
  Donor/alias tests passed after formatting as well. Whole-workspace rustfmt
  check remains red from unrelated baseline files; patch-owned Rust was
  formatted in isolation.
- Queue-executed native operator oracle passed **21/21 cases per selector**.
  It exercised all twelve entry points for each family on synthetic full-shape
  tensors, FP64 sampled references, finite/padding/guard checks, repeated use,
  attention modes and queried pipeline limits. This is not a real checkpoint.
- Queue-executed ABBA/BAAB projection trials passed as processes. The trial
  reports retain all samples and their own five-percent drift/order gates.
  Conditions were observed, never used as a stability wait. Both queue jobs
  reported sampled conditions eligible (AC, low-power false, thermal state 0).

| Cell | SG8 incumbent / candidate µs | SG8 eligible ratio | SG4 incumbent / candidate µs | SG4 eligible ratio |
| --- | ---: | ---: | ---: | ---: |
| W4 down, M1, K15360 | 237.28 / 112.33 | 2.112× | 236.67 / 112.34 | 2.107× |
| W8 global O, M1, K8192 | 147.79 / 59.30 | 2.492× | 149.07 / 59.95 | 2.487× |
| W8 local O, M1, K4096 | 66.55 / 30.15 | 2.208× | 72.22 / 33.65 | **ineligible** |
| W8 local O, M9, K4096 | 318.14 / 144.60 | 2.200× | 313.53 / 110.86 | 2.828× |

The control is the existing native-BF16 `experimental_projection_*_bf16_n4`
on the **same synthetic quantized weight data**. Times are median GPU
per-dispatch estimates in a hot-cache operator-only loop; they are not
end-to-end, real-weight, MLX, ANE, checkpoint quality, or promotion results.
The SG4 local M1 cell exceeded the 5% stability gate (incumbent drift 23.96%,
candidate drift 30.90%); its apparent numerical ratio must not be counted.
All reports explicitly have `automatic_promotion: false` and
`production_promotion: false`.

## Evidence and next gates

`sg{8,4}-oracle/donor12b-oracle.json` and
`sg{8,4}-abba/donor12b-projection-abba.json` are the raw native receipts.
`queue-results/` contains the exact queue jobs, reports, stdout/stderr and
sampled conditions. `jobs/` contains the submitted job manifests. The ABBA
receipts have SHA-256 `9b1e2ab9775b550e40b1e4354f08418c2954021445b64ed0c4549db28e17d1fd`
(SG8) and `34d3360e4ef4f6ebeaf7500f76101c6033ad424fe43e1f4b7cc7d3c0c83e7ea5`
(SG4).

Next: append these candidates after the occupied PR #4 registry slots; rerun
catalog/build/native gates on that **integrated** tree. Then run checkpoint
sidecar authentication, real-weight full-route correctness and route dispatch
accounting, longer-context attention and paired whole-model timing. No source
or numerical operator screen establishes those properties.
