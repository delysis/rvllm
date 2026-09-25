# Gemma 4 native-BF16 low-bit projection screen

Date: 2026-09-25

This is real-checkpoint operator evidence for the experimental Metal
`W4ABF16` and `W8ABF16` projection ABIs. It is not full-route, model-quality,
or promotion evidence.

The source tensor is Gemma 4 12B layer-0 Q projection BF16 `[4096, 3840]`.
Activations and outputs remain BF16, group-32 quantization scales are FP16,
and accumulation is FP32. The entry points are intentionally distinct from
the existing W4A16/W8A16 FP16 ABI.

The checked-in receipt is `q-projection-m1-m4.json`, SHA-256
`8deb00a77d4fe2d4403134585c88319f71e3d5e8304aa551b98523395048a870`.
It seals generated MSL
`a9aa9ea629a5bda62f8d0292e841b3e62a0868b4dac71324759a4dc7fd4557f9`
and executable
`f0e44fa198634f865fb3ab72ddb8fb27c9c05e246ecc629cc918998408cb1171`.

All four cases passed exact-dispatch accounting, two correctness dispatches,
bitwise repeated output, output guards, and the independent CPU low-bit
reference. Relative-L2 error was `0.00079` through `0.00091`; maximum absolute
error was at most one BF16 step (`0.0078125`).

| ABI | M | candidate median | dense-BF16 median | observed speedup |
| --- | ---: | ---: | ---: | ---: |
| W4ABF16 | 1 | 0.198 ms | 0.357 ms | 1.80x |
| W4ABF16 | 4 | 0.603 ms | 2.433 ms | 4.03x |
| W8ABF16 | 1 | 1.761 ms | 1.034 ms | 0.59x |
| W8ABF16 | 4 | 2.026 ms | 2.322 ms | 1.15x |

These point estimates are screening observations only. Two earlier identical
runs changed by multiples and sometimes reversed the winner, so no speed claim
is promotable. The next gate is sealed repeated ABBA execution through the
kernel-game queue, followed by all seven projection roles and checkpoint-level
quality evaluation.

Reproduction command:

```sh
cargo run -p rvllm-apple-metal \
  --features low-bit-real-weight-research \
  --bin rvllm-low-bit-real-weight -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --tensor model.language_model.layers.0.self_attn.q_proj.weight \
  --m 1,4 --samples 5
```

