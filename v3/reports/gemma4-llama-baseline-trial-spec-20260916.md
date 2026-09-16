# Native-kit / llama.cpp baseline preparation

2026-09-16. Source, artifact metadata and GGUF-header inventory; no model, GPU or ANE execution. A standalone safe Rust runner is prepared separately from product sources. Its offline release build and two pure host tests passed; execution must go through the coordinator's serialized experiment queue.

**A comparable workload is available, but a same-checkpoint baseline is not currently established.** The local 12B GGUF is Google's official native QAT Q4_0 artifact; rvllm's accepted path uses the standard BF16 checkpoint with INT8 FFNs. Report this as a practical alternative configuration, not a kernel-only speedup or numerical equivalence. Targeted native-kit/model-cache inventory found no standard-checkpoint BF16/FP16/Q8 12B GGUF.

## Which native-kit

The requested `/Users/george/delysis/native-kit` path does not exist. The historical checkout is `/Users/george/Documents/llama-native-kit`, HEAD `c3fe09b782469c88c6a3e0bf3a35f38394ce6569`; merged sources are `/Users/george/delysis/native-platform/crates/native`, parent HEAD `308f5f05a453374f03b3d3dc1c5528bf1531451a`. The inspected native source paths are clean. Both current manifests pin **delysis/llama-cpp-rs `a3cf95eb1d4fa748480eb780e6fcbfc1a5c1c391`**, whose vendored llama.cpp is **`5f55650a78f92aff4d48d671423e888fac0469ff`**. See [historical manifest:22](/Users/george/Documents/llama-native-kit/crates/llama-native-engine/Cargo.toml:22), [merged manifest:16](/Users/george/delysis/native-platform/crates/native/crates/llama-native-engine/Cargo.toml:16).

This llama.cpp pin recognizes `gemma4` and implements its sliding/global geometry, absent-V/raw-K case, proportional global RoPE and dense FFN. It labels unlisted layer counts, including 48, `UNKNOWN`; that switch supplies a size label, not a loader rejection. Actual loading/numerical behavior is still unqualified. [registration](https://github.com/ggml-org/llama.cpp/blob/5f55650a78f92aff4d48d671423e888fac0469ff/src/llama-arch.cpp#L58), [Gemma source](https://github.com/ggml-org/llama.cpp/blob/5f55650a78f92aff4d48d671423e888fac0469ff/src/models/gemma4.cpp#L3-L135).

The old `Documents/llama-native-kit/target/release/mom-llama-app` is not a benchmark. Its SHA256 is `c9565301b430b75ff5ec10bd90c8e4d311332473afee4c8f22ce7716ff7d23be`, but build metadata says llama commit `unknown` and points to registry `llama-cpp-sys-2-0.1.153`, with `LLAMA_BUILD_TOOLS=OFF`. Do not assign the current source pin to that binary. Existing native-kit receipts found are smaller Qwen or Gemma E2B/E4B runs, not this 12B baseline.

## Existing model and alternate tools

The existing official model is [Google's pinned QAT repository](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/tree/29d097773436b69ff9feafd636ab4cf873786537). Local path:

`/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-gguf/snapshots/29d097773436b69ff9feafd636ab4cf873786537/gemma-4-12b-it-qat-q4_0.gguf`

It resolves to a 6,975,879,296-byte blob. Google's file metadata gives SHA256 `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`; the full local tensor payload was **not rehashed** during inventory. Header inspection found GGUF v3, 667 tensors: 338 F32, 328 Q4_0, one Q6_K embedding. Geometry is 48 layers / H3840 / I15360 / 16 query heads / sliding window 1024 / no per-layer embedding. It is not uniformly four-bit. No model download or conversion is needed for this candidate.

Separately installed Homebrew tools are b8640, source `7992aa7c8e21ea2eb7a5e4802da56eec7b376036`, linked to system GGML 0.9.11. They are **not native-kit's pin**. No tool was executed, including help/version.

| Installed artifact | SHA256 |
|---|---|
| `/opt/homebrew/Cellar/llama.cpp/8640/bin/llama-completion` | `f5f45c4eba7bd9ba347c64c0193f1e3418914509f61b7538e383a2c2d640ccc4` |
| `/opt/homebrew/Cellar/llama.cpp/8640/bin/llama-bench` | `39d3d1515991a2320a37cafcbaa58ff7eedf9914b226b682b660a46549b3bb48` |
| `/opt/homebrew/Cellar/llama.cpp/8640/lib/libllama.0.0.8640.dylib` | `9d186ad0c331278eb23be0849ce2ed7e521f9c210e95ce15f34d458ac29c9dca` |
| `/opt/homebrew/opt/ggml/libexec/libggml-metal.so` | `94aa7d46eec7040da0929cc130d3c334b1765ae4eb904381550d6868ca28cfbb` |

`llama-bench` uses random token IDs in `test_prompt`/`test_gen`; `-p 84 -n 10` is not this text request, and ten generation evaluations differs from nine continuation evaluations. Keep it a separately labeled synthetic diagnostic. [Pinned implementation:1997–2044](https://github.com/ggml-org/llama.cpp/blob/7992aa7c8e21ea2eb7a5e4802da56eec7b376036/tools/llama-bench/llama-bench.cpp#L1997-L2044).

## Isolated retained-context runner

Source and exact build/run commands: [bench README](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/bench/README.md), [main.rs](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/bench/src/main.rs), [Cargo.toml](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/bench/Cargo.toml). It uses native-kit's exact safe-binding pin, with Metal enabled and Rust common/mtmd features disabled; the pinned sys build nevertheless configured CMake `LLAMA_BUILD_COMMON=ON`. This is a **native-kit pinned backend benchmark**, not a measurement of the complete native-kit request/event layer.

The [build receipt](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/build-receipt.json) records the exact command, lockfile, source hashes, build log, two passing pure host tests and frozen binary. Executable SHA256: `91fcfde541135113652c62dd696fd8a560071d505e50784470901b10a9cff8fe`. There are no model-execution results yet.

The runner consumes [the saved 84-ID reference](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/int8-stacked-full-model-20260916/checked/case-1/report.json) and [the exact prompt file](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/int8-stacked-comparison-20260916/copy-prompt.txt), verifies their hashes, and verifies GGUF tokenization against all IDs using the SHA-checked checkpoint template. It preserves the same non-thinking prefix/suffix as [rvllm's text encoder:9–39](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/text_generation.rs:9). No additional BOS or chat template is allowed.

Fixed work/configuration: context 1024, all-layer Metal requested, F16 K/V, Flash Attention enabled, batch/ubatch 512, one sequence, 8 CPU threads, mmap enabled, mlock disabled, greedy argmax; 2 warmups plus 7 measured requests in one process, with KV cleared before each. Ten outputs require exactly nine decode evaluations after sampling the first output from prefill. EOG is respected and recorded; early termination invalidates the fixed-work trial. Repeated IDs must agree. A QAT/standard-output difference is separately labeled rather than hidden.

Native-kit's public `GenerationMetrics.tokens_per_second` divides completion count by whole-request duration; it is not decode-only throughput. [metric calculation:3542](/Users/george/delysis/native-platform/crates/native/crates/llama-native-engine/src/lib.rs:3542). The safe bindings expose `reset_timings`, `timings`, `clear_kv_cache`, `decode` and logits reads. The pinned native [logits getter synchronizes:3692–3706](https://github.com/ggml-org/llama.cpp/blob/5f55650a78f92aff4d48d671423e888fac0469ff/src/llama-context.cpp#L3692-L3706); [internal counters:3210–3234](https://github.com/ggml-org/llama.cpp/blob/5f55650a78f92aff4d48d671423e888fac0469ff/src/llama-context.cpp#L3210-L3234) are elapsed timers and clamp zero counts. The runner therefore records actual completed evaluations independently, synchronized PP wall time, individual decode eval wall times, sampling-inclusive decode wall time, TTFT, whole-request wall time, internal PP/decode time and actual IDs. None is a device-cycle measure.

## Trial acceptance

The queue should run only one accelerator trial at a time, retain stderr placement logs, hash the model once outside timing, freeze the built executable/lockfile, and reject failed loads, shortened work, counter errors or power-stratum changes. Do not measure while repairing ANE cache. Pair rvllm and llama runs in a predeclared ABBA order with the same sampled power source, low-power mode and thermal eligibility. Report PP milliseconds/84 tokens, nine-step decode time/rate, and full request time separately; llama has no Metal-to-ANE import phase, while rvllm's end-to-end result must include it. Warmup and load time remain separately visible.

The historical ANE 6.08594 steps/s observation was battery / low-power off / mode 0 / nominal. It cannot be compared directly with a new AC / low-power-on result. No matched llama result exists yet. Matching prompt/work/power yields a useful practical baseline; an equal-coefficient claim needs both systems to use the same checkpoint and reconstructed weights, which this task does not implement.
