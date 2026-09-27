# Gemma 4 BF16 prefill phase-boundary diagnostic — 2026-09-26

## Result and claim boundary

The existing serial experiment queue ran two real-weight Gemma 4 12B IT BF16
512-token prefill + two-token continuation jobs. Both jobs succeeded, retained
all condition samples, were sampled-condition eligible in the same reported
stratum (AC, power mode 2, no low-power mode, thermal state 0), and had no
violations. No stable-thermal gate was requested. This is a two-arm exploratory
screen, **not** ABBA/BAAB confirmation or a promotion decision.

| Arm | Prefill wall ms | Command-buffer wait ms | CPU encode ms | Prefill encoders | Generated IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| Default prefill; donor decode selector on | 22,086.449 | 22,067.362 | 19.000 | 481 | `236770, 236770` |
| MMA32 + SIMD prefill opt-ins; same donor decode selector | 3,937.257 | 3,926.482 | 10.759 | 625 | `236770, 236770` |

The opt-in arm was 5.610x faster on prefill wall time in this sequential pair.
The command-buffer waits account for 99.91% and 99.73% of their respective
prefill wall intervals. This rules out host encoding as the primary explanation
for the roughly 18.15-second difference in these runs; it does **not** split
device time among projection, FFN, attention, queue scheduling, or individual
shaders. More encoders in the faster arm also show why encoder count alone is
not a cost proxy.

The new `cases[0].prefill_phase` receipt is sampled before prefill launch and
after its completed collect, so subsequent decode work cannot contaminate its
counter deltas. Both arms report one prefill step, zero decode steps, one
command buffer, zero library/pipeline compiles in this bounded phase, and an
empty **research-candidate** dispatch ledger. The full-case ledger contains
donor decode dispatches in both arms. An empty research ledger is not evidence
that no ordinary prefill kernels ran: this ledger does not cover ordinary GEMM,
FFN, or attention entry points. Neither opt-in's selected ordinary kernel name
nor per-kernel GPU duration has yet been captured on this route.

The older same-host MLX BF16 512-token reference was 173.051 prompt tok/s;
512/3.937257 gives this opt-in rvLLM arm approximately 130.0 prompt tok/s,
or a directional 1.33x MLX lead. Those jobs did not share exact prompt tokens,
model files, continuation semantics, or a paired timing block, so this is only
orientation. The earlier default-route continuation experiment first differed
from its reference at generated index 5 with MMA alone, index 57 with SIMD
attention alone, and index 9 with both enabled. Two matching IDs here therefore
do **not** establish the opt-in route's numerical quality or authorize promotion.

## Exact evidence and next discriminator

- Source commit: `be7c5b6775258dfc43eab255381cbb2e6dc5ab75` on PR #6.
- Executable SHA-256: `a03c68c8f201685e267b17b0dd4e140ba47d2a084de944c266da6bd963f66494`.
- Queue manifests: `off.json`, `on.json`; complete immutable queue receipts,
  stdout/stderr, and power/process observations: `queue-results.tar.gz`
  (SHA-256 `0ead4bd28c09a1271fb937ae0fc7900f6884d49669a718526d0199be8bed0d94`).
- Direct normal-route reports: `off-report.json` (SHA-256
  `e3b67e086ac213299c52cbd3d57d91105e779bc3f0eeb9c4c15920614070e757`)
  and `on-report.json` (SHA-256
  `50970bbfcb6f7909c07d033570e4a225e5c0c7342f4e7157f50332b1b021f5eb`).
- The referenced Google BF16 checkpoint config, prompt JSONL, wrapper, and
  metallib are individually pinned in each queue manifest. No weights or
  checkpoint data are copied into this report.

Next: capture actual ordinary prefill function names and complete per-role
projection/FFN/attention time via a validated Metal trace or bounded stage
isolation, then run matched prompt/model MLX and route-preserving long
continuation/logit checks. Isolate MMA and attention opt-ins separately; do not
attribute the combined speedup to either option from this pair alone.
