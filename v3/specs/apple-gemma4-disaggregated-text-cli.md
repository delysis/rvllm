# Gemma 4 12B: Metal prefill and ANE text decode

The macOS research CLI runs the local dense Gemma 4 12B checkpoint with Metal prefill and first-token sampling, then imports its actual KV cache into all 48 ANE decode layers. Large decode projections, FFNs, attention and the vocabulary head execute on ANE. Small normalization, RoPE, residual and sampling operations use Rust on the CPU. There is no CPU or Metal decode fallback.

This currently supports greedy, non-thinking, single-user text turns with a total input/decode capacity of 1,024 tokens. A persistent stdin mode accepts successive independent user turns. The HTTP adapter accepts the same single-user turn format. Message histories, tools and multimodal input are not yet supported. The qualified checkpoint is `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The formatter verifies the checkpoint template digest before encoding it; it does not implement arbitrary Jinja templates.

## Build and run

Run from `v3`. Build the Rust executable and a BF16 macOS Metal library from the same source checkout. The second command emits source and its pipeline manifest; the following commands require Apple's Metal toolchain.

```sh
cargo build --release -p rvllm-runtime --features macos-private-ane-research --bin rvllm_disaggregated_infer
cargo run -p rvllm-apple-metal --bin emit_metal_kernels -- bf16 target/gemma4-bf16.metal target/gemma4-bf16-pipelines.json
xcrun --toolchain Metal -sdk macosx metal -std=metal3.1 -c target/gemma4-bf16.metal -o target/gemma4-bf16.air
xcrun --toolchain Metal -sdk macosx metallib target/gemma4-bf16.air -o target/gemma4-bf16.metallib
```

The cache provisioning steps below are needed before the first ANE run. Once prepared:

```sh
target/release/rvllm_disaggregated_infer \
  --model-dir /absolute/path/to/gemma-4-12B-it \
  --metallib-bf16 target/gemma4-bf16.metallib \
  --ane-compile-budget 16 \
  --prompt 'Reply with just the capital of France.'
```

Generated text is streamed to stdout; status goes to stderr. `--prompt-file PATH` reads UTF-8 text. Repeated prompt arguments share initialization by batching all prefills before ANE preparation. `--interleave true` instead keeps both backends loaded and completes each prefill/ANE decode before beginning the next request. `--output-dir PATH` optionally writes local token/timing reports into a new directory. Without that flag, ordinary text mode writes no prompt or output report files. `--capture-layer-states true` requires an output directory and additionally exports KV and the first ANE step's layer states.

For a resident session, add `--interactive true` and omit `--prompt`. Wait for the status line confirming both backends are loaded, then enter one user prompt per line. Each line is a separate single-user turn; prior answers are not included in later prompts. End stdin to unload both backends. Empty lines are ignored and over-budget prompts are rejected before accelerator work. Initial `--prompt` arguments may precede interactive input. The prepared Metal scratch covers the configured capacity, so later prompts can be longer than the first.

Interactive sessions retain only the latest receipt in their summary report. With `--output-dir`, full results are saved separately under `case-N/result.json`; `requests_completed` is the cumulative count. Without that option, no request history is retained after the next request completes. Accelerator errors terminate the session.

`--runtime-worker true` selects the serial `EngineHandle` owner, with both devices constructed, used and destroyed on its accelerator thread. It requires the qualified INT8/MMA/SIMD configuration and currently excludes layer capture. The same library handle supports cancellation, bounded response backpressure and queued requests. Cancellation is observed at synchronous operation boundaries; dropping an unfinished request cancels it. Critical memory pressure releases the owner between requests and requires explicit reinitialization. CLI input remains one completed turn at a time. This explicit route passed live A/B/A, cancellation followed by fresh requests, critical-pressure release and normal CLI exit with zero compiler calls. See `worker-shutdown-live-cancellation/` and `joined-worker-text-three/` in the evidence directory. The HTTP adapter below uses this same serial worker.

Both CLI routes now record actual process CPU cycles/instructions and sampled AC/battery, power-mode and thermal state. Metal prefill additionally records the completed command-buffer GPU interval. These are separate quantities: CPU cycles do not normalize ANE or Metal execution time. See [measurement and comparison rules](apple-performance-measurement.md). `rvllm_compare_disaggregated` refuses unlike or unobserved power states, thermal pressure, diagnostic captures and different workloads.

The request API now retains its accelerator thread handle. Explicit
`EngineHandle::shutdown()` or dropping the final engine handle stops admission,
cancels outstanding work and waits for owner destruction at a synchronous
boundary. Dropping another clone keeps the session alive. This wait is required
at CLI exit: detaching the thread allowed process termination to interrupt ANE
unload/source-directory cleanup and could block the next launch. A live response
stream with a full bounded queue does not prevent shutdown. Shutdown requested
from the owner thread itself signals stop without attempting to join itself.

Defaults are `--ane-weights static-int8-ffn-cached`, `--context-capacity 1024`, and `--max-new-tokens 64`. INT8 quantizes FFN weights per output channel; other ANE weights remain FP16. `--ane-weights static-all-cached` selects FP16 FFNs. The first Metal output counts toward the token limit. Admission checks `prompt_tokens + max_new_tokens - 1 <= context_capacity` before loading accelerators, with no silent truncation. EOS IDs come from checkpoint metadata, including the turn-ending token.

For the experimental stacked gate/up FFN layout, separately prepare
`--prepare-ane-cache ffn-int8-stacked` in a fresh, journaled process. This part
has at most 48 compiler calls and does not alter the default `all-int8` batch.
`--ane-weights static-int8-stacked-ffn-checked` retains both original and
stacked FFNs (210 programs total), executes both on every decode activation,
and requires finite, bit-identical outputs before continuing with the
candidate output. Any mismatch stops inference; no fallback occurs. This mode
requires pinned references, an output directory, `RVLLM_ANE_DIAGNOSTIC_JOURNAL`,
and zero compile budget. `stacked_ffn_checks_per_layer` records cumulative
successful comparisons for all 48 layers. Its timings include duplicate work
and must not be used for performance claims.

The separate `static-int8-stacked-ffn-cached` plan loads only the candidate FFNs
(162 programs) for later performance qualification. Both experimental plans
use the original INT8 integers/scales and FP16 activations, and keep QKV,
output, attention and head unchanged. They are available on the explicit CLI
route; the runtime worker and HTTP service retain the qualified default.

MMA and SIMD-group attention are enabled by default for the qualified native BF16 prefill shapes on Apple9/10. `RVLLM_METAL_PREFILL_ATTENTION=off` selects the previous scalar attention kernel. `RVLLM_METAL_PREFILL_GEMM=off` selects the previous Metal routing. The release executable requires a precompiled Metal library: `--metallib-bf16` or `RVLLM_METAL_METALLIB_BF16` supplies it. The CLI resolves these choices once into `ModelMetalOptions`, always selects native BF16 with FP32 accumulation, and passes context/batch limits directly. It does not mutate process environment. The explicit library constructor uses the same immutable policy for encoding and numeric identity, and consumes one budgeted plan for allocation/loading. Diagnostic environment overrides for dtype, capacity, tracing and forced synchronization do not change this route.

## Serve over HTTP

Build the private research feature and keep the signing identifier consistent
with cache provisioning:

```sh
cargo build --release -p rvllm-serve --features macos-private-ane-research --bin rvllm-server
codesign --force --sign - --identifier rvllm_disaggregated_infer-9d1f7275eb5937c9 target/release/rvllm-server
codesign --verify --strict target/release/rvllm-server
target/release/rvllm-server \
  --backend metal-ane --addr 127.0.0.1:8080 \
  --model-dir /absolute/path/to/gemma-4-12B-it \
  --metallib-bf16 target/gemma4-bf16.metallib \
  --max-total-tokens 1024 --max-new-tokens 64
```

The server prepares once, then accepts `GET /healthz` and
`POST /v1/completions`. The default ANE compilation budget is zero: prepare the
cache first. Capacity must be 64 or 1024 and the BF16 library path is explicit.
The health response identifies the route, readiness and actual preparation
receipt. A failed or stopped owner reports HTTP 503.

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Reply with just the capital of France.","max_tokens":16}'
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Reply with just the capital of France.","max_tokens":16,"stream":true}'
```

`prompt` is ordinary user text, wrapped using the pinned Gemma non-thinking
formatter. This route provides greedy generation; it does not implement chat
messages or sampling controls. JSON replies include text, token counts, exact
prompt/output IDs and phase measurements. SSE sends incremental text, a terminal
report and `[DONE]`. Invalid context budgets return HTTP 400 before generation
or streaming headers. After streaming starts, runtime errors use SSE error frames.

One request runs on the device owner at a time, with four queued requests and
bounded token delivery. A disconnected stream cancels at the next observed
socket write and synchronous accelerator boundary. Cancellation need not be
instantaneous. Every new request starts with fresh KV state. Ctrl-C or SIGTERM
stops admission, closes active sockets, cancels work and waits for all ANE
unloads and worker destruction. The loop bounds simultaneous HTTP connections
at 64 and applies 30-second socket I/O timeouts.

The signed release `a2e9f7a47d28d9a8fb2dc9587a70f0e5110c6371a4501da0297569f60d5a124f`
passed real JSON/SSE, queued requests, active-stream disconnect recovery and
SIGTERM cleanup: 26 reference output IDs, 162 cache hits, zero compiler calls,
5,200 completed evaluations and 162 successful unloads. No owned staging
folders remained and the boot identity did not change. Evidence:
`reports/gemma4-12b-evidence-20260914/http-worker-live/`. The driver-journal run
qualifies correctness and lifecycle, not throughput.

A separate unjournaled measurement of that binary used two warmups followed by
seven identical requests. On battery, low-power off and nominal thermal state,
median ANE decode was 6.09 tokens/s; the 84-token prompt and 10-token response
took 2.30 seconds end to end, with 0.803 seconds for prefill plus import. Every
request matched the reference. This is a workload-specific baseline, not a
before/after speedup or an estimate for another power mode. Raw phase counters,
power observations and timing are in `http-worker-measurements/`.

## Prepare the ANE cache

Use a consistent code-signing identifier for provisioning and inference.
Changing Rust dependencies can change the linker's generated identifier, even
when the executable filename stays the same. For the existing research cache,
the qualified identifier is `rvllm_disaggregated_infer-9d1f7275eb5937c9`:

```sh
codesign --force --sign - --identifier rvllm_disaggregated_infer-9d1f7275eb5937c9 target/release/rvllm_disaggregated_infer
codesign --verify --strict target/release/rvllm_disaggregated_infer
```

This is local ad-hoc signing, not distribution signing. Preserve binary SHA and
CDHash separately. A stable identifier does not guarantee cache persistence.

To inspect a part without compiling, use `--inspect-ane-cache PART` in place of
`--prepare-ane-cache PART`. The strict inspector attempts each graph's existing
cache load and immediately drops it, recording available/missing entries. It
recreates caller-owned source staging, but permits zero compiler calls and zero
evaluations. Only a confirmed cache absence is collected as a missing entry;
other construction/load failures stop inspection. Unload return values are
recorded separately in the optional driver journal. Availability is a point-in-time observation,
not a promise that all programs can later remain resident together.

Inspect or prepare the complete INT8 decoder with one command. Supply a fresh
output directory for each invocation and the same context capacity as inference:

```sh
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --context-capacity 1024 --inspect-ane-cache all-int8 --output-dir cache-inspection
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --context-capacity 1024 --prepare-ane-cache all-int8 --output-dir cache-preparation
```

`all-int8` visits QKV, output, INT8 FFN, then head/attention in four fresh,
serial child processes of the same signed executable. Preparation allows at
most 48, 48, 48 and 18 compiler calls respectively; inspection allows zero.
Neither operation evaluates the model. Each graph is unloaded before the next
visit. The parent verifies each child's receipt and driver journal, including
completed compiles and unloads, before starting the next part. A failure stops
the batch without retry. `report.json` records completed parts and a final
`complete`, `failed` or `cancelled` status. A complete inspection can contain
missing entries; check `all_programs_available_at_visit` as well as status.

SIGINT, SIGTERM and SIGHUP request cancellation at the next graph boundary.
The parent signals its child through a marker file and waits for it to finish
the current synchronous operation and unload. It does not kill the child.
Cancellation and failures preserve per-part logs and journals; aggregate
verified counts cover only successfully validated parts. They do not count
work from an interrupted part whose final receipt is unavailable.

Individual parts remain available for targeted recovery. These commands compile
missing entries and perform no inference evaluations:

```sh
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --prepare-ane-cache qkv --output-dir cache-receipt-qkv
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --prepare-ane-cache output --output-dir cache-receipt-output
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --prepare-ane-cache ffn-int8 --output-dir cache-receipt-ffn-int8
target/release/rvllm_disaggregated_infer --model-dir /absolute/path/to/gemma-4-12B-it --prepare-ane-cache head-attention --output-dir cache-receipt-head-attention
```

The experimental `static-int8-ffn-sliding-qkv-cached` plan uses the ordinary
INT8 FFNs and replaces only the forty sliding QKV projections with the
qualified per-output INT8 encoding. The eight global packed QKV projections,
output matrices and vocabulary remain FP16. `--prepare-ane-cache
qkv-sliding-int8` visits those forty candidate graphs and the eight original
global graphs; `all-int8` still prepares the default plan. The candidate uses
162 programs and defaults to strict cached loading. It has component evidence
only: full-model quality and performance remain unqualified. It is not used by
the runtime worker or server default. Provision and compare only after the
current baseline campaign, using a separately frozen executable and receipts.

For FP16 FFNs, provision `ffn` instead of `ffn-int8`. Cached programs can be evicted by the system. Inference defaults to zero compiler attempts; the explicit `--ane-compile-budget 16` permits at most sixteen process-wide repairs, then returns an error before decoding if preparation remains incomplete. It neither falls back nor loops indefinitely. Provisioning receipts do not prove subsequent residency or model correctness.

The ANE path uses private macOS APIs and is gated behind `macos-private-ane-research`. All multiple-input/output requests are rejected before entering those APIs following the earlier driver panic. See [the incident record](../reports/ane-panic-20260914.md) and [current measurements and acceptance boundaries](../reports/gemma4-12b-metal-ane-progress-20260914.md). Initialization still costs tens of seconds; warm decode throughput is reported separately from that startup cost.

The initial late-arriving lifecycle test sent four prompts only after each preceding request completed, with lengths 21, 84, 652 and 21. All 17 generated IDs matched independent CPU references; the repeated short prompt reproduced all 144 captured state/KV files byte-for-byte. Its durable diagnostic journal intentionally synchronized every ANE evaluation, so its decode timings are not representative of ordinary use. See `interactive-late-arrival-four` in the evidence directory for identities and sequencing.
