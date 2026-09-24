# MLX Gemma 4 Q4/256 generated-Metal capture

This lane captures exactly one MLX-LM normal-route request: affine Q4 g64,
256 pseudo-random prompt tokens, batch one, and 64 generated tokens. It uses
upstream `mx.metal.start_capture(path)` and `stop_capture()` and produces an
Xcode `.gputrace` package. Model loading and parameter materialization occur
before capture begins.

The installed MLX 0.30.5 wheel is not acceptable for this evidence because its
build does not establish `MLX_METAL_DEBUG=ON`. The original v1 attempt built
core commit `c215b6f88cf0fee0b0895623e4046cda797ef397`, but capture failed before
GPU work because pinned MLX-LM commit
`87b7b583a697537aa68f47130b40884700b5f55f` declares `mlx>=0.32.2` and calls
`new_thread_local_stream`, which that older core does not expose. Preserve the
v1 failure receipt as incompatibility evidence.

The v2 queue job instead builds official MLX tag `v0.32.2`, commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`, into a fresh dedicated
import-precedence target with `CMAKE_ARGS=-DMLX_METAL_DEBUG=ON`. Upstream says
that flag records Metal source and labels Metal objects. The build tool emits a
receipt binding the expected source commit, CMake/setup inputs, build flag,
interpreter, and resulting `mlx.core` binary hash.

The build and capture are intentionally separate jobs. After the build job
succeeds, materialize the capture job with the actual debug runtime and
build-receipt hashes; never substitute placeholders or submit it early. Run the
capture with `MTL_CAPTURE_ENABLED=1`. The capture path must not exist. Both
tools reserve stdout for one JSON receipt.

The `.gputrace` is generated-source and dispatch-inspection evidence. It is not
an isolated-stage timing, an end-to-end benchmark, a correctness oracle, or a
speed/promotion result. Any timing displayed by Xcode belongs to this captured
diagnostic run and must not be combined with the isolated stage matrix as if
they shared a timing protocol.

Lightweight inspection only:

```sh
/Users/george/christian_mystics/venv/bin/python \
  tools/build_mlx_metal_debug.py \
  --mlx-source /Users/george/.cache/rvllm-mlx-source-v0.32.2 \
  --expected-commit 1f8e74e3f12f31365464a6867c6579f0e9b29d85 \
  --target /Users/george/.cache/rvllm-mlx-metal-debug-v0.32.2 \
  --receipt /Users/george/.cache/rvllm-mlx-metal-debug-v0.32.2-build.json \
  --plan-only

/Users/george/christian_mystics/venv/bin/python \
  tools/mlx_gemma4_gputrace_capture.py \
  --mlx-source /Users/george/.cache/rvllm-mlx-source-v0.32.2 \
  --mlx-lm-source /Users/george/.cache/rvllm-mlx-lm-87b7b583 \
  --model /Users/george/.cache/rvllm-mlx-gemma4-12b-affine4-g64-20260924 \
  --capture /Users/george/.cache/rvllm-mlx-gemma4-q4-pp256-20260924.gputrace \
  --plan-only
```
