# MLX Gemma 4 Q4/256 generated-Metal capture

This lane captures exactly one MLX-LM normal-route request: affine Q4 g64,
256 pseudo-random prompt tokens, batch one, and 64 generated tokens. It uses
upstream `mx.metal.start_capture(path)` and `stop_capture()` and produces an
Xcode `.gputrace` package. Model loading and parameter materialization occur
before capture begins.

The installed MLX 0.31.2 wheel is not acceptable for this evidence because its
build does not establish `MLX_METAL_DEBUG=ON`. The first queue job builds core
MLX commit `c215b6f88cf0fee0b0895623e4046cda797ef397` into a fresh dedicated
import-precedence target with `CMAKE_ARGS=-DMLX_METAL_DEBUG=ON`, which upstream says
records Metal source and labels Metal objects. The build tool emits a receipt
binding the source commit, CMake/setup inputs, build flag, interpreter, and
resulting `mlx.core` binary hash.

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
  --mlx-source /Users/george/osaurus/Packages/swift_convert/.build/checkouts/mlx-swift/Source/Cmlx/mlx \
  --target /Users/george/.cache/rvllm-mlx-metal-debug-c215b6f \
  --receipt /Users/george/.cache/rvllm-mlx-metal-debug-c215b6f-build.json \
  --plan-only

/Users/george/christian_mystics/venv/bin/python \
  tools/mlx_gemma4_gputrace_capture.py \
  --mlx-source /Users/george/osaurus/Packages/swift_convert/.build/checkouts/mlx-swift/Source/Cmlx/mlx \
  --mlx-lm-source /Users/george/.cache/rvllm-mlx-lm-87b7b583 \
  --model /Users/george/.cache/rvllm-mlx-gemma4-12b-affine4-g64-20260924 \
  --capture /Users/george/.cache/rvllm-mlx-gemma4-q4-pp256-20260924.gputrace \
  --plan-only
```
