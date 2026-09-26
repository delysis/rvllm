# MLX Gemma 4 stage microbenchmark

`tools/mlx_gemma4_stage_bench.py` measures isolated operators from an actually
loaded Gemma 4 MLX-LM model. It covers embedding; sliding and full-attention
QKV, SDPA, and output projection; FFN gate/up plus activation; FFN down;
RMSNorm plus residual; and the tied LM head. The canonical matrix contains
prefill and one-token decode shapes at contexts 256, 512, 1024, 2048, and 4096
for each supplied 4-bit, 8-bit, or BF16 model.

The operator timing protocol is sealed to core MLX commit
`c215b6f88cf0fee0b0895623e4046cda797ef397`, specifically
`benchmarks/python/time_utils.py`; its source root and file SHA-256 are recorded
in every receipt. Model implementation provenance is separately sealed to MLX-LM commit
`87b7b583a697537aa68f47130b40884700b5f55f`. Every receipt includes that commit,
the absolute source path, and SHA-256 hashes of both the pinned Gemma 4 model
implementation (`mlx_lm/models/gemma4_text.py`) and the upstream end-to-end
runner (`mlx_lm/benchmark.py`). Model config and weight-index hashes are also recorded.
The harness refuses a source-commit mismatch or a requested weight width that
the model config does not expose.
Quantized-path enforcement applies to the QKV, output, and FFN linear modules.
The tied embedding/LM-head storage is reported exactly as loaded rather than
being assumed quantized; MLX-LM may leave that shared table unquantized.

Each case uses exactly five warmups followed by 100 timed iterations. Every
warmup and iteration constructs the operator graph and synchronizes it with
`mx.eval`; inputs and prerequisite stage outputs are materialized before the
warmups. One enclosing timer covers the 100 synchronized iterations, matching
the upstream timing-loop shape without adding a Python timer call to each
operator execution. JSON output includes total and per-iteration mean duration.
JSON serialization rejects NaN and infinity.

These measurements are diagnostic isolated-operator timings. They are not
end-to-end normal-route prefill or decode timings, do not include surrounding
model stages, and cannot alone qualify a performance claim or promotion.

Inspect the complete matrix without loading weights or running Metal work:

```sh
PYTHONPATH=/Users/george/.cache/rvllm-mlx-lm-87b7b583 \
python tools/mlx_gemma4_stage_bench.py \
  --mlx-source /path/to/mlx-at-c215b6f88cf0 \
  --mlx-lm-source /Users/george/.cache/rvllm-mlx-lm-87b7b583 \
  --model /path/to/model \
  --weight-bits 4 \
  --plan-only
```

Run each model representation in a separate kernel-game queue job without
`--plan-only`. The production matrix must use the sealed BF16, affine Q8 g64,
and affine Q4 g64 model identities already recorded by the campaign; it must
not run the three resident models concurrently. Queue receipts remain the
authority for host state, eligibility, and isolation.
