# Gemma 4 Metal next-round kernel packet

Status: **source candidates only; not compiled, correctness-qualified, or benchmarked on Apple hardware in this session.**

## Diagnosis

The current campaign is spending too much optimization budget on attention after the long-context global operator has mostly caught MLX. The large remaining end-to-end gap is more consistent with projection/FFN bandwidth, repeated materialization, and launch/runtime boundaries. The next useful unit of work is therefore not another broad attention sweep; it is a deliberately tiny family of decode-first projection/fusion kernels plus a short-context attention route that avoids paying split-KV costs where they do not amortize.

## What is new here

### 1. `gemma4_qmv_g64.metal`

Two affine group-64 decode QMV kernels, Q4 and Q8. The important choices are:

- dequantize *inside* the dot product; never materialize dense weights;
- one lane pair spans exactly one 64-value quantization group;
- eight output rows per SIMDgroup, two SIMDgroups per threadgroup;
- one activation pair and its `sum(x)` are reused across all eight rows;
- FP32 accumulation; BF16 output rounded once.

This is deliberately modeled on the high-level strategy visible in current MLX: shape-selected qmv/qmm families and fused dequantized arithmetic. The 8-row mapping also follows the direction that has paid off in recent llama.cpp Metal decode tuning: more output rows per SIMDgroup can improve register/dispatch utilization on Apple GPUs.

Initial roles should be **W4 down projection** and **W8 output head**, because those are the roles for which the rvLLM campaign already has positive operator evidence. Do not blanket-apply Q4/Q8 to Q/K/V; existing evidence says W8 K lost and V is unresolved.

### 2. `gemma4_ffn_decode_fused.metal`

The high-value candidate is `gemma4_ffn_gateup_gelu_q4_g64_r4_sg2` for exact M=1, K=3840, I=15360 decode.

Instead of:

1. project `[3840] -> [30720]` gate+up,
2. write 30,720 values,
3. launch GELU*mul,
4. read 30,720 values,
5. write 15,360 activated values,

it computes gate and up together while traversing K once, immediately applies GELU*up, and emits only the 15,360 activated values needed by the down projection. This attacks both memory traffic and a dispatch boundary.

A BF16 version is included as a control so the value of **fusion** can be measured separately from the value of Q4 storage.

### 3. `gemma4_global_short_decode.metal`

This is intentionally *not* another long-context split-KV kernel. It is a 128-thread, unsplit D512/GQA16 kernel aimed only at L<=512, where current split-32 measurements show poor amortization relative to MLX. Four SIMDgroups cover all 16 query heads; each K/V token is conceptually reused across four heads within each SIMDgroup.

The file uses contiguous K/V as a compact algorithmic prototype. It is **not ready for rvLLM routing** until ported onto the existing paged-cache semantics, newest-K/V rules, holes, restored prefixes, rollback, and guard-byte oracle.

## Selector policy worth testing

Do not seek a single global winner. Start with:

- global attention L<=512: short unsplit candidate vs incumbent stable route;
- global attention L>=1024: current cooperative split-32 / split-matrix campaign remains the relevant family;
- decode FFN gate+up: fused gateup->GELU BF16 control first, then Q4 only after checkpoint quality qualification;
- decode FFN down: Q4 group64 candidate;
- output head: Q8 group64 candidate;
- Q/K/V: retain checkpoint-native/BF16 initially; reopen role-specific low-bit only where quality + ABBA both pass;
- prefill: do not reuse these M=1 kernels. Continue a separate matrix/tensor path.

## Exact gates

For each candidate:

1. compile and seal source/metallib/PSO identity;
2. exact shape/refusal tests and guard bytes;
3. deterministic CPU oracle (including affine `scale*q + bias` group semantics);
4. repeated-use bit stability;
5. whole-tensor role correctness against the actual checkpoint;
6. checkpoint logit/perplexity gate for Q4/Q8;
7. ABBA and reverse-order timing against the *current role-specific incumbent*;
8. only then normal-route token timing;
9. only after route qualification, matched 64-token MLX comparison.

## Integration caveat

The attached status report indicates rvLLM's current quant package differs from MLX's affine group-64 format. These kernels therefore intentionally define a clean group-64 ABI rather than pretending to be drop-in compatible with an unseen current branch. A real integration needs either:

- a one-time checkpoint repacker into this format, or
- adaptation of the inner loop to the current rvLLM packed format while retaining the row/SIMDgroup/fusion strategy.

The latter is likely the smaller near-term experiment if the current W4/W8 package is already checkpoint-qualified.
