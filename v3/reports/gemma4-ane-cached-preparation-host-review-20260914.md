# Strict cached ANE preparation: avoid rebuilding source weights

2026-09-14. Read-only source/evidence review; no hardware, builds, private selector calls, cache access, or implementation edits. The rejected 32×64 GEMM experiment does not change the qualified 32×32 production choice.

**The strongest measured opportunity is host FFN source preparation before the ANE constructor, not the 162 framework load calls.** Persisting the exact already-qualified INT8 MIL/blob representation during provisioning would avoid repeated checkpoint conversion, row quantization, and serialization. Keep the current one-input, strict cache-check/load/cleanup lifecycle initially. Source artifacts are not daemon-compiled artifacts.

## Existing evidence already narrows the bottleneck

I summed adjacent preparation-stage timestamps before the first evaluation in two completed all-hit journals. Program order is two attention graphs, then QKV/O/FFN for each of 48 layers, then 16 vocabulary tiles. The FFN subtotal maps every third projection constructor accordingly.

| Wall-clock scope | Retained run | Phased run |
|---|---:|---:|
| Reported total ANE preparation | 54.584 s | 60.773 s |
| Prior request created → next program constructor requested | 30.472 s | 35.514 s |
| FFN-only subset of preceding row | **25.717 s** | **29.985 s** |
| Constructor requested → descriptor/identifier/directory ready | 9.831 s | 11.043 s |
| Cache hit → source files staged | 3.210 s | 3.418 s |
| Load completed → source-data verified/unlinked | 2.324 s | 2.536 s |
| All 162 load-begin → load-completed intervals | **1.373 s** | **1.451 s** |

Evidence: [retained journal](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/disaggregated-mma-extended-int8-retained-capital/driver-phases.jsonl), [retained report](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/disaggregated-mma-extended-int8-retained-capital/inference/report.json), [phased journal](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/disaggregated-mma-extended-int8-capital/driver-phases.jsonl), [phased report](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/disaggregated-mma-extended-int8-capital/inference/report.json).

Both runs recorded **162 cache hits and zero compiler calls**. Their configured recovery budget was 16, not strict-zero, but the executed hit branch is shared with strict loading. These are historical, journaled application intervals, not isolated CPU/device timings or matched residency comparisons. [Every journal record calls sync_all](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/diagnostic_journal.rs:41), and timestamps precede that sync; each interval includes logging/scheduling overhead. Table rows are selected scopes, with the FFN row a subset, not additive totals. Nevertheless, source work before descriptor creation is unambiguously substantial; the data do not establish how much is conversion versus quantization.

## Why cache hits still do model-sized host work

1. [Layer preparation](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:353) reads Q/K/V, O, and all three FFN tensors, then [loads all vocabulary tiles](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:431). [read_values](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:801) opens each source file, allocates/read-fills bytes, allocates FP16 values, and converts/checks every element. Source checkpoint reads cover approximately 23.81 GB of projections/head before small norms.
2. [load_static_ffn](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:187) unconditionally quantizes INT8 on every load, including strict cached loading. [RowInt8::quantize](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_int8_ffn_weights.rs:15) scans each row for its maximum, stores an FP16 scale, then performs division/round-ties-even/clamping per weight: 8,493,465,600 FFN coefficients across the model. [blob_and_constants](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_int8_ffn_weights.rs:131) copies the integer vectors/scales into another owned blob.
3. [AneLinear packing](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:58) serializes FP16 projections/head again. The [sys constructor](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:309) copies the blob into NSData and constructs the descriptor/model/identifier **before** querying the compiled cache at line 364. The 9.8–11.0 s descriptor interval combines these operations; it does not prove which private operation hashes/copies weights.
4. Even on a hit, [staging](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:371) writes the complete blob, hardlinks it as `data`, then [compares every byte before unlinking](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/source_staging.rs:8). The hardlink already prevents two separate caller-staging payload copies. Across INT8 FFNs plus FP16 projections/head, source blobs total approximately **15.323 GB**; that amount is written and subsequently read for verification, excluding any daemon work.

The architecture/header scan is small metadata work. Small norm loads, scalar setup, and repeated file opens merit timing but are lower-priority than the identified FFN region. Host source vectors already drop per graph; another late `drop` does not remove descriptor-owned NSData.

## Ranked changes

**1. Reuse exact prepared INT8 FFN source artifacts — highest expected impact, explicit disk tradeoff.** At provisioning, save the exact MIL and blob emitted by the existing quantizer, with model/checkpoint content identity, layer/dimensions, source precision, quantizer/MIL version, byte length, and digest. Normal startup reads/verifies these bytes and calls the existing strict constructor. Start with FFNs only: **8,496,804,864 bytes (7.913 GiB)** plus manifests/MIL. A complete source package would retain approximately 15.323 GB (14.271 GiB), additional to checkpoint and daemon storage. Do not assume available disk or create a second full package during atomic replacement without budgeting it. Initially preserve the existing staging/verification path; this removes repeated quantization without changing its established lifecycle.

Require byte-identical MIL/blob and unchanged graph/cache identifiers versus current provisioning, and reject stale/mismatched artifacts. Select an explicitly content-identified model package; a pathname/mtime alone must not authorize reuse for changed checkpoint weights. A source artifact's presence cannot authorize compile fallback or prove a daemon hit. This is a proposed safe-Rust packaging layer, not an existing loader API or measured startup speedup.

**2. Transfer blob ownership to NSData — small implementation scope, transient-copy savings.** Replace the borrowed-blob-only internal construction path with ownership transfer once staging access is accounted for. objc2's safe `NSData::from_vec(Vec<u8>)` retains Vec capacity in its deallocator; `with_bytes` uses copying initialization. Keep immutable access and ownership inside the FFI crate, without recreating a copy through `NSData::to_vec`/`iter`. This avoids one approximately 177 MB FFN copy per construction, not all loaded-source residency or descriptor identity work. [Verified objc2 pin 7b1abfd…](https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/framework-crates/objc2-foundation/src/data.rs#L35), [deallocator](https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/framework-crates/objc2-foundation/src/data.rs#L305).

**3. Defer source-link/mapping optimizations.** h3's primary implementation restores a hardlinked/copy-backed source tree and uses no-copy NSData ownership. Its “compiled” marker and stored files do not establish independently portable lowered programs; its compile fallback also differs from strict policy. The useful precedent is ownership and source reuse, not copying its cache authority or deletion behavior. [h3 pin 6538b226…](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L73). Keep verified cleanup and exclusive directory ownership; do not clear descriptors, guess nil setters, bypass cache queries, or parallelize private-object loading.

## Timing scopes to add before choosing further work

Collect monotonic durations in memory, aggregate by graph kind/layer, and emit once after preparation. Preserve durable lifecycle journaling separately; adding another fsync per timing span would distort attribution.

- `checkpoint_read` and `bf16_to_fp16`: separate byte read/allocation from conversion.
- `ffn_quantize`, `blob_serialize`, and source-artifact `read_validate` if introduced.
- `nsdata_create`, `descriptor_create`, `model_create`, `identifier`, and `cache_query`: split the present combined descriptor interval without changing selectors/options.
- `source_write`, `source_link`, `load_call`, `verify_source`, `unlink_source`, `surfaces_create`, and `request_create`.

Record bytes, cache hit/miss/actual compile counts, total prepare time, and peak client memory. Keep the verified row-scale rounding/division intact while measuring; replacing division with reciprocals can change quantized bytes and therefore both quality and cache identity. The persistent same-thread owner already amortizes preparation across prompts; it should retain that benefit while startup source reuse is evaluated.
