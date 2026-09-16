# Native-kit pinned Gemma 4 12B backend baseline

This standalone safe Rust executable uses the **same llama-cpp-rs/llama.cpp source pins as both current native-kit checkouts**. It measures a retained model/context through the safe bindings, not the full native-kit request/event/cache layer. It does not use the old `mom-llama-app` executable or Homebrew's different llama.cpp build.

**Model mismatch is intentional and explicit:** the existing artifact is Google's native-QAT Q4_0 GGUF, with Q6_K token embeddings. rvllm currently uses the standard BF16 checkpoint plus INT8 FFNs. Equal workload is possible; equal model coefficients/precision is not established. This baseline does not authorize a QAT conversion or rvllm four-bit implementation.

The runner is fixed to 84 prompt tokens, context 1024, 10 sampled outputs (9 continuation evaluations), 2 warmups and 7 measured repetitions. It retains model/context on one thread, clears KV per repetition, requests all GPU layers, Flash Attention enabled, F16 KV, batch/ubatch 512, and greedy first-maximum sampling with no penalties. Requested offload must be checked against captured placement logs. No warm-prefix reuse is allowed.

It accepts the actual `prompt_token_ids` field from the frozen rvllm reference, checks its SHA256, count and vocabulary bounds, and independently verifies that the GGUF tokenizer produces exactly those IDs from the frozen prompt and checkpoint's SHA-checked non-thinking template. A model/template/token mismatch stops before any decode. Model loading itself initializes the Metal backend: **run only through the coordinator's serialized accelerator queue**.

Build only (one package, no model access):

```sh
CARGO_TARGET_DIR=/Users/george/Documents/llama-native-kit/target cargo build --offline --locked --release -j 2 --manifest-path /Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/bench/Cargo.toml
```

The initial build generates `Cargo.lock`; subsequent builds use `--locked`. C++ compilation may be needed: older release artifacts used registry bindings, not this Git pin. No production sources are changed.

Queue this future command only after recording the model's SHA256 once outside timing and arranging the same power/thermal stratum as rvllm:

```sh
/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/rvllm-native-kit-baseline \
  --model /Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-gguf/snapshots/29d097773436b69ff9feafd636ab4cf873786537/gemma-4-12b-it-qat-q4_0.gguf \
  --prompt-reference /Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/int8-stacked-full-model-20260916/checked/case-1/report.json \
  --prompt-file /Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/int8-stacked-comparison-20260916/copy-prompt.txt \
  --threads 8 \
  --output-json /ABSOLUTE/NEW/TRIAL/backend.json
```

The parent directory must exist and the output path must be new. Preserve stdout/stderr separately. The runner does not poll power, wait for favorable conditions, spawn subprocesses, mutate environment, compile models or retry failures; the external queue owns eligibility and serialization. Do not run a live ANE process concurrently with this model.

The JSON report has `schema: "rvllm_llama_native_kit_baseline_v1"`, `status`, `preparation`, `configuration`, `repetitions` and `failure`. Each repetition includes `warmup`, all `prompt_token_ids`, actual `output_token_ids`, EOG status, `requested_decode_steps`, `actual_decode_steps`, `n_generated`, `prefill_wall_ms`, `decode_wall_ms` (including sampling), `decode_eval_wall_sum_ms`, per-step synchronized wall time, TTFT, whole-request wall time, and internal `internal_prompt_ms` / `internal_prompt_tokens` / `internal_decode_ms` / `internal_decode_runs_raw`. `llama_get_logits_ith` synchronizes in this pinned source. These are host elapsed measurements, **not GPU timestamps or cycle counters**. The native timer's zero eval count is clamped to one; actual completed work is separately recorded.

The tenth sampled output is not evaluated. EOG is included in the output list and stops generation; EOG before output ten invalidates the fixed-work trial rather than being suppressed. Any wrong workload count, internal count mismatch, nonfinite logit, or differing IDs across repetitions stops the process with a failure record and nonzero status. A difference from rvllm's standard-checkpoint output is recorded but is not automatically a runner failure: the checkpoints differ. Quality acceptance is a separate decision.

Receipts are buffered in memory outside request timing and the single JSON file is written/synced after the campaign, including completed repetitions on a returned error. A native process crash can leave an empty file; queue exit status and stderr must reject it. The runner checks the inventoried model size and metadata but deliberately does not reread/hash 7 GB per run or claim a hash check it did not perform. Expected model SHA256: `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.

Review the report `gemma4-llama-baseline-trial-spec-20260916.md` for source pins, interpretation and matched-trial requirements. Building this tool is not qualification of loading, Metal execution, output quality or performance.

Preparation receipt: `../build-receipt.json`. The offline release build passed; both pure host unit tests passed. The frozen binary above has SHA256 `91fcfde541135113652c62dd696fd8a560071d505e50784470901b10a9cff8fe` (5,283,784 bytes). No model has been loaded or executed. Rust `common`/`mtmd` features are disabled; the pinned sys build still configured CMake `LLAMA_BUILD_COMMON=ON`, as recorded in the receipt.
