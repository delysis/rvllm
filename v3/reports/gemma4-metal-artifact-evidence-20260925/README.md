# Matched-attention Metal artifact evidence

This directory contains public-tool compiler and pipeline evidence for the
matched split-32 versus split-matrix attention source used at 1,024 live keys.
The strict receipt is `evidence.json`, SHA-256
`6305e1f06c47be36e8ee1dc5e492d41cf4999ffbe0eb93906dc6760e88cd3960`.

The captured public `MTLComputePipelineState` properties are:

| Kernel | SIMD width | Maximum threads | Static threadgroup memory |
| --- | ---: | ---: | ---: |
| cooperative split-32 partial | 32 | 1,024 | 2,912 B |
| cooperative split-32 merge | 32 | 1,024 | 384 B |
| split-matrix partial | 32 | 1,024 | 12,512 B |
| split-matrix merge | 32 | 1,024 | 0 B |

The directory retains the compiled AIR, metallib, compiler logs, build tables,
and raw public `metal-objdump` output. Every artifact is bound by size and
SHA-256 in the receipt. Use `rvllm_metal_artifact_evidence_verify` to reject
changed, missing, duplicate-key, unknown-field, or semantically invalid
evidence.

Apple's supported public interfaces do not expose stable register count,
register residency, or occupancy data. Raw AIR/objdump output is not treated as
a semantic guarantee of executed machine SIMD-matrix or low-bit-unpack
lowering. The receipt records those claims as unavailable or unverified rather
than inferring them.

This is generated-code/resource evidence only. It is not correctness, dispatch,
timing, MLX-comparison, or promotion evidence.
