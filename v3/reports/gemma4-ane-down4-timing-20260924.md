# Gemma 4 ANE Down4 timing screen — 2026-09-24

## Decision

`static-int8-down4-ffn-cached` is **correctness-qualified but not a performance
nominee**. Keep ordinary `static-int8-ffn-cached` as the ANE INT8 incumbent.

Five usable counterbalanced pairs contain 405 decode steps per arm. The
geometric mean of the per-step ordinary-INT8/Down4 wall-time ratios is
`0.982632`, so Down4 is about 1.77% slower. Down4 was faster on 192 steps and
slower on 213. The pooled arithmetic means are 321.423 ms/token for ordinary
INT8 and 331.815 ms/token for Down4, a 3.23% Down4 regression. The results vary
substantially by pair and do not support promotion.

This is an exploratory timing rejection, not pristine promotion evidence. The
strict nominal comparator rejected the comparisons when observations were
missing, stale, or changing. The checked-in referee's
`--stable-fair-exploratory` mode accepted pairs 2–6 within the same observed
Fair stratum. Conditions were logged rather than awaited; no stable-condition
admission delay was used.

## Correctness and route prerequisites

Before timing, Down4 passed all of the following:

- Three real layer-0 FFN activations from the 21-, 84-, and 652-token routes
  were finite, deterministic, and bit-exact to the ordinary INT8 control.
- A normal full-model route matched the reference tokens, used 162 ANE
  programs, executed on ANE with no Metal decode steps, and compiled nothing.
- The exact timing workload produced the ten expected tokens, executed nine
  ANE decode steps with no Metal steps, loaded 162 programs, and compiled
  nothing.

Qualification pins:

- component events:
  `5440ad3cac754191dd6b2929a77690d575a6e72fcfe441db76cd3853c9db5e82`
- exact timing-workload route report:
  `5c9573356967b13f0710f2d9d1c96cbb5cbac769257cea694cc74a0a1cc6afb6`
- full-route qualification README:
  `a95e5dc495c1f1c296c2306b6b2d074aad3cd1a85a4630a0b7bed6be68845fdd`

## Pair results

Each arm used nine identical 84-token prompts, generated ten tokens, and
contributed 81 measured decode steps. A ratio above 1 favors Down4.

| Pair | Execution order | Geometric ordinary/Down4 | Arithmetic ordinary/Down4 | Down4 faster/slower steps | Verdict |
|---:|---|---:|---:|---:|---|
| 1 | A1, B1 | not admitted | not admitted | not admitted | exclude: B1 overlapped a release Cargo build and lacks intact comparable observations |
| 2 | B2, A2 | 0.927685 | 0.952373 | 31 / 50 | Down4 slower |
| 3 | B3, A3 | 0.987948 | 1.011105 | 46 / 35 | effectively tied, geometric loss |
| 4 | A4, B4 | 1.061913 | 1.087122 | 48 / 33 | Down4 faster in this pair |
| 5 | A5, B5 | 1.004975 | 1.016465 | 48 / 33 | effectively tied |
| 6 | B6, A6 | 0.936647 | 0.944976 | 19 / 62 | Down4 slower |
| **2–6 pooled** | counterbalanced | **0.982632** | **1.002408** | **192 / 213** | **do not promote** |

The arithmetic mean of individual ratios is close to parity because a small
number of large favorable ratios pull it upward; the geometric ratio and the
direct pooled wall-time means both reject a speedup. Pair 6 is especially
informative because it reverses the pair-5 order and removes the apparent
small advantage.

## Phase means for usable pairs

These are unweighted arithmetic means over the 405 steps in pairs 2–6. Only
the FFN layout differs by design; movement in other phases is evidence of
whole-run variance, not a claim that Down4 changed those kernels.

| Phase | Ordinary INT8 ms/token | Down4 ms/token |
|---|---:|---:|
| total | 321.423 | 331.815 |
| FFN | 144.574 | 148.903 |
| attention | 34.270 | 35.352 |
| QKV | 55.679 | 57.757 |
| output projection | 38.966 | 41.547 |
| host | 17.095 | 17.270 |

## Arm receipts

The hashes below pin the self-contained inference report followed by the outer
queue report. Raw reports, power observations, and logs remain in the retained
queue result directories.

| Arm | Mean total ms/token | Mean FFN ms/token | Inference report SHA-256 | Queue report SHA-256 |
|---|---:|---:|---|---|
| A1 control | 318.674 | 134.471 | `7ada60dfed28010ce646a7e6d08f60d1a1899f46f913a2d5a24389e55b70e509` | `e4d3169cee09847ef706753f957051bd9e0ffafa653ac244b4c31ed17674d3b4` |
| B1 Down4 | 386.872 | 168.123 | `15a3a9164086c886c4e69534427b1b752143c35b22bfd885645d86bba6a4b529` | `8ed41c8a3aeaeb9a20381fb2b3e6379c8eaa04ab019c983e2bd0bfd3ebb9fd14` |
| B2 Down4 | 364.038 | 154.769 | `45a75c4cc1bf953f02546e0d54432ecf077acfa647b53b1079f431a9cdd449fc` | `f15b49e27044e51f006f93d61f081fea4a5ba0ee23c540c21fd1e391f6c610b3` |
| A2 control | 323.822 | 145.638 | `cc482b8ac5b3fb32d7e9e7cbe8512ca0df5376e67ff1b81b5f7c922611d6ece9` | `e4214c6d55a0322ab3e0f8b82cefdc77c7b914e19ba46de39036012ce465f6cd` |
| B3 Down4 | 327.000 | 144.532 | `ab062d3301a1dbdbbbd98f77edfc1e41330146a226b148ad1bc2eafc623c29e9` | `3c119eee3bf993ea693d5e16ce0836d469e295a103bee715393115be25d133a4` |
| A3 control | 318.721 | 142.849 | `07caeba482145004527338c69b084baf6c3997d1651daa2b1bc115dac994f843` | `c0dbaa1bb80d8c381ae6846286655e6b38f38141e8e0e4a012387189de6c6004` |
| A4 control | 310.408 | 140.135 | `e75ad13d899e871ce9caf5e741e71d57a0b964cfed2c3460917837bdfc7d6f3b` | `8fa3bf36e5519c63209a71cb30ce5b05253f4f73b25596627ac26baea3485973` |
| B4 Down4 | 292.884 | 131.641 | `2135642ac2c69fb1b8ddb409fe76c4b35c7c4353f24c4a2ee9c045bf64c021f6` | `394a1d7c60f32c81e97b3f73d1b1481b120111bcea5418f24cc856c86e05485e` |
| A5 control | 351.654 | 162.426 | `84902c416876355cf24cc08f0b553e7e9effd575d039bfbb5fdfd9db2e5dba6d` | `b36bc774c976608ba91efc6aaec63bd284b06d92e0798d5a9a1e1832dc96375f` |
| B5 Down4 | 350.919 | 161.410 | `ea10ade1a94d0bdc81214abc39423e7b642c99f6eb46d3d7903304558c47e283` | `d15b7139e77bdfa136dc878a3556ae6f96e84ed04da6c7420dc2ec5da0bce5e4` |
| B6 Down4 | 324.235 | 152.161 | `aa47a03fc757a3b0039db8b589c73a28e60b017e2f7ee411b8b2ed13bb3a3510` | `bc02a4dfcd9005bae157dbd9763c2d7a6d0f89177072df3e156f62a17057f5c6` |
| A6 control | 302.511 | 131.823 | `df7b6aa7a088c36bfe8a9a29be99199db89b2e27f5fff582393e97b6e3681a5c` | `46a2a58718d36ce868536f815f530d5fe87fdc4458c415357af58a6c0f261da3` |

Fair-comparator receipt hashes for pairs 2–6, respectively:

- `9ceea4c7704f536266139a22de2f55c91ebe4e52bb1526e6b178c030fede7536`
- `3275fa07af6799bbe3725851da689784d3ddc7b63dfa470fb862fab113de16dd`
- `b5996d3ab530eaa4c21b2c36351ccd445219f549198f9008dce5b4f149e5ec51`
- `be82023defb19b243030c8979bf9350f463145d1d698a8c39ad4d72d8dfc02e2`
- `15fad0dada7d9d69149c9b638c8ccb4c8770fd7ac4e69d8a96ca33b3fb837077`

## Next design constraint

Down4 changes tiling but does not reduce the number of ANE program or
convolution boundaries. Together with the slower interleaved candidate and the
numerically rejected Chunk4 candidate, this points the next round toward fewer
program/conv boundaries or a genuinely fused layout—not another cosmetic
tiling variant. Any successor must first pass the same real-activation oracle
and exact-route identity gates before entering timing.
