# Gemma 4 D512 headwise-SIMD control

Date: 2026-09-24

## Decision

`metal-global-d512-r1p128t32` is correctness-qualified as a research control,
not a promotion nominee. It materially changes the interpretation of the
earlier global-attention tournament:

- at short context, ordinary head-level GPU occupancy is more valuable than
  sharing K/V across all 16 query heads;
- between 512 and 1024 tokens the existing shared-K/V `r8p64t128` schedule
  crosses over and becomes faster;
- at 2048, the headwise control is almost twice as slow as `r8p64t128`;
- both schedules remain far behind the comparable MLX D512 attention result at
  2048, so the next algorithmic trial remains occupancy-driven split-KV.

This result disproves the idea that one unsplit ownership pattern should serve
all context buckets. It supports an autotuning boundary between a short-context
headwise route and a longer-context shared-K/V or split-KV route.

## Candidate

The control launches one 32-lane SIMD group per query head:

- grid: `[16, 1, 1]`;
- threads: `[32, 1, 1]`;
- D panel: 128;
- FP32 output state per lane: 16 values;
- source and queried static threadgroup memory: 3,200 bytes;
- no global scratch;
- deliberately rereads the shared K/V stream for all 16 heads.

It is not the final GQA16 design. Its purpose is to distinguish gains from
ordinary GPU parallelism from gains due to shared K/V reuse.

## Correctness

The Apple M4 Max / Apple9 native oracle passed. It covered the same adversarial
paged-cache, prefix/rollback, hole-page, rejected-dispatch, guard-byte,
repeated-use, independent sampled FP64-dot, exact serial-GPU FP32, and exact
once-rounded BF16 contract as the unsplit family.

Key sealed identities:

| Identity | Value |
| --- | --- |
| Candidate source SHA-256 | `d3f019aade6a429979d5b1778c376ac5daaaf59b9a21f95f85261fed293d46f3` |
| Oracle metallib SHA-256 | `cdd11a62d7131ee0314252455427f04617b5ebfcf57fca3bc0883e88dcfd5722` |
| Test executable SHA-256 | `b6d785badcd80c0a09f5bc7dff812119de4a5b51455309263d09dac1ef7d61dc` |
| Oracle receipt SHA-256 | `a6710614d0be53fdc36630db601e37544e401d74d83a965bae6672a5cdcb656a` |

## Exploratory ABBA timing

Every cell used five warmups per arm, five ABBA blocks, 100 dispatches per
sample, zero source compiles during samples, and retained all observations.
The baseline is the untouched scalar `attention_decode_f16` production
fallback. Values are arithmetic means over the ten samples for each arm.

| Context | Scalar baseline | Headwise control | Baseline/control | Time reduction | 5% control-drift gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 15.884638 ms | 3.059039 ms | 5.1927x | 80.74% | failed; retained |
| 512 | 34.666162 ms | 8.243714 ms | 4.2052x | 76.22% | failed; retained |
| 1024 | 70.050860 ms | 17.738599 ms | 3.9491x | 74.68% | passed |
| 2048 | 267.458398 ms | 45.155368 ms | 5.9231x | 83.12% | failed; retained |

Receipt SHA-256 values, in context order 256/512/1024/2048:

- `3a9650c0391bfb879b8608c71f3f44c51d4301fb1df5dfc90d086b96a1fe27cf`
- `119c5d3a54e35047339a103a0403ccc71dd27880b05960243735bea7a45e11fa`
- `b24fd04d89fb2b67360e33025b1a3ca3e33f268cd54ed3df63616620146172ff`
- `22ff5d8e3804d59996b284c519e8e4ff1d319c467a70d61db0a1c6836bcd0e85`

The failed drift cells are exploratory evidence, not promotion evidence. They
are intentionally retained because the campaign does not wait for idealized
stable conditions.

## Comparison with the existing shared-K/V leader

The older cells were collected under separate host strata, so these are
directional comparisons rather than matched ABBA between the two candidates.

| Context | Headwise control | `r8p64t128` | Direction |
| ---: | ---: | ---: | --- |
| 256 | 3.059 ms | 4.918 ms | headwise about 37.8% faster |
| 512 | 8.244 ms | 8.293 ms | effectively tied |
| 1024 | 17.739 ms | 16.473 ms | shared-K/V about 7.1% faster |
| 2048 | 45.155 ms | 23.399 ms | shared-K/V about 48.2% faster |

The main conclusion is the crossover, not the last decimal place.

## Comparison boundary with MLX

The comparable retained MLX BF16 isolated full-attention decode result at
context 2048 is 1.489122 ms. Therefore:

- headwise control: approximately 30.3 times slower;
- current rvLLM `r8p64t128` leader: approximately 15.7 times slower.

The BF16 MLX context-1024 stage job was condition-ineligible and reported a
large outlier, so it is not used to claim a ratio. More broadly, rvLLM still
lacks completed normal-route Metal stage timing for QKV, local attention, O,
FFN, norms, embedding, and LM head. The existing MLX 4/8/16-bit stage matrix is
measured, but model-wide rvLLM Q4/Q8 routes do not yet exist. Missing and
incomparable cells must remain explicit rather than being inferred from the
ANE mixed-precision or BF16 load4 campaigns.

## Next trial

Implement the first bounded split-KV vertical slice:

`metal-global-d512-split-r8s256t128`

It should reuse the proven rows-8 / D-panel-64 / 128-thread inner schedule,
emit FP32 `(maximum, denominator, D512 numerator)` state for each 256-token
partition, and merge partitions in order with identity handling for empty or
fully masked partitions. Cap the initial admitted cache capacity at 4096 and
time the first genuinely split workload at context 512. Partial, merge, and
total synchronized GPU time must all be recorded; only total time advances.

No production default changes on this evidence.
