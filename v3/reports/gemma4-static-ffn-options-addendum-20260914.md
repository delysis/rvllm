# Static FFN options for Gemma 4 12B ANE decode

Source-only addendum, 2026-09-14. The original [megakernel research report](/Users/george/Downloads/rvllm/v3/reports/megakernel-gigakernel-research-20260914.md) is unchanged. No ANE/GPU execution, benchmarks, builds, or product edits were performed for this addendum.

## Recommendation and evidence boundary

**First investigate 116 programs: 48 static FFNs, 48 static QKV, two shape-shared dynamic O projections, 16 vocabulary tiles, and two shared attention graphs.** This improves the earlier 70-program proposal by making only the smaller O weights dynamic while preserving the checked host normalization and RoPE boundaries. Static O+FFN fusion offers a stronger eventual critical path at 114 programs, but requires new normalization and rounding qualification. A real load-without-compile artifact cache exists upstream; its production resource and compatibility validation is a larger task than its short implementation suggests.

The parent reports that the explicit-tanh dynamic FFN passes numerical checks, with contended medians of 0.040 ms input, 18.37 ms evaluate-call, and 0.011 ms output. The evaluate-call includes scheduler waiting; it is not an isolated hardware execution measurement. Eliminating the measured host copies would address only about 0.3% of their combined scale. The historical static FFN observation, 3.076792 ms, predates the reboot and lost artifacts; it is neither a current acceptance result nor a matched uncontended comparison. Static FFN must be requalified with the accepted GELU expression. The parent also reports shared attention and the 1024 ring wrap passing with a 32× probability/PV scaling bracket. Full 115-program end-to-end qualification remains pending as this report is written.

Current source confirms that dynamic FFN weights are staged once, and each token writes only activation elements. Request sharing already exists; it is not a proposed optimization. See [AneDynamicFfnProgram::create_layer and project_profiled](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_dynamic_ffn.rs:36) and [AneInMemoryProgram::create_request](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:322).

All counts below assume one context configuration, the current 16 head tiles, and two attention programs. They count graph instances requiring compilation, not hardware instructions, wired-memory limits, or a guaranteed safe ceiling.

| Option | Programs | Implementation/qualification cost | Expected critical-path effect, conditional on measurement |
|---|---:|---|---|
| Current dynamic FFN | 115 | Implemented; end-to-end qualification pending | Baseline |
| **Static FFN + dynamic O only** | **116** | Lowest additional semantic change; two linear shapes and static GELU FFN | Removes dynamic treatment from dominant FFNs; introduces it only in O |
| Static O+FFN tail per layer | 114 | Moderate graph work; substantial norm/rounding gate | Keeps all projections static and removes one dispatch per layer plus host tail math |
| All-static separate graphs via artifact cache | 162 loaded graphs; potentially zero runtime compile calls on a complete cache hit | Small cache prototype; higher cross-process, disk, residency and failure-path qualification | Retains existing host math and static O; caching itself changes startup, not evaluation speed |
| Earlier dynamic QKV+O proposal | 70 | More dynamic layouts and QKV qualification | More compile headroom, but more dynamic weight traffic than 116 |
| Static full layer | 64 | Highest; attention, KV packing, norms and RoPE change together | Broadest fusion; greatest compiler/resource/numerical uncertainty |

## 1. The 116-program path: exact O layout

O consumes the concatenated attention output: `K = 16 × head_dim`, hence 4096 for 40 sliding layers and 8192 for eight global layers; output width `N = 3840`. These dimensions are explicit in [decoder loading](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:116). O does not require KV-head-dependent variants.

Reuse the proven dynamic FFN's single-surface packing convention, without its activation function:

| Property | Sliding O | Global O |
|---|---:|---:|
| One external FP16 input | `[1,4096,1,3872]` | `[1,8192,1,3872]` |
| Logical activation slice | `[1,4096,1,1]` | `[1,8192,1,1]` |
| Weight slice | `[1,4096,1,3840]` | `[1,8192,1,3840]` |
| Packed input bytes/request | 31,719,424 | 63,438,848 |
| Actual weight bytes/request | 31,457,280 | 62,914,560 |

`3872 = 32 + N`. For input channel `i`, activation `x[i]` occupies spatial slot zero, slots 1–31 are padding, and slot `32+j` contains checkpoint weight `W[j,i]`, where checkpoint O is `[N,K]`. Its byte offset is `2*(i*3872 + 32+j)`. Slice and reshape activation to `[1,1,1,K]`, weights to `[1,1,K,N]`, apply `matmul` with both transpose flags false, then reshape to logical `[1,3840,1,1]`. Under the current 64-byte channel-stride convention, allocate/read 245,760 output bytes and extract each channel's first FP16 element.

Compile once per K, create independent layer requests, stage transposed weights once, then update activation with `write_tensor_strided(0, 0, 3872*2, 2, activation_bytes)`. This API already exists in [the isolated ANE FFI wrapper](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:399); the graph idiom is in [PackedFfnLayout::mil](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_ffn_layout.rs:113). These exact O dimensions and compiler lowering remain **unexecuted proposals**.

Across 48 requests, packed O input storage is 1,776,287,744 bytes; actual dynamic O weights are 1,761,607,680 bytes/token, **7.40%** of the projection-plus-head weight budget. Dynamic QKV+O in the 70-program design covers 4,812,963,840 bytes, **20.21%**. FFNs remain 16,986,931,200 bytes, **71.33%**, before compression. These are calculated from the [durable weight budget](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/weight-budget.json), not measured traffic.

The 116 arrangement leaves only three nominal compile slots below 119. Provision in a fresh process; account for failed compiles and probes, not only successful graph objects. Preserve current CPU norms, residual rounding, layer scale, Q/K/V handling, and the accepted GELU expression while comparing O numerics and timing. If O's dynamic penalty proves material, use that result to justify fusion or caching. Use 70 only when additional context variants require its headroom; do not make QKV dynamic merely to reduce the count further.

## 2. Reusable compiled artifacts: a concrete private-API implementation

The strongest current source is **maderix/h3.c-ane**, commit `6538b226efe5c4fff905d299dcc4426db2f44de4` (latest observed 2026-09-14). Its [bridge implementation, lines 161–225](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L161) performs:

1. `_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:` and `_ANEInMemoryModel inMemoryModelWithDescriptor:`.
2. Reads `hexStringIdentifier`, then restores compiled artifacts to `NSTemporaryDirectory()/identifier`.
3. Calls `loadWithQoS:options:error:` **without compile** on a cache hit.
4. Calls `compileWithQoS:options:error:` only when restoration or loading fails, then loads and saves artifacts.

Callable C entry points are `h3_ane_model_create`, `h3_ane_model_unload`, `h3_ane_model_reload`, and `h3_ane_model_cache_hit`. [Reload, lines 292–310](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L292), restores the cache and calls load directly. The [test source, lines 437–475](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/tests/test_ane_full_block.c#L437), checks three unload/reload/evaluate cycles for bit-identical output. I inspected this source; I did not run its tests.

This gives a specific adaptation point: split the current [AneInMemoryProgram::compile_inner](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:232) into artifact restore/load and cold compile paths. Keep Objective-C calls isolated there; Rust should own cache manifests, checked sizes, request ownership and cleanup. A complete static set could be provisioned across fresh compiler processes and loaded later. **Source evidence does not prove that 162 models can be loaded together, that load never compiles internally, or that this bypasses every cause behind the observed ~119 limit on this machine.** Validate actual compile-call counts, clean-process reuse and residency before relying on it. Do not silently recompile a large cache miss in the inference process.

The cache is weight-specific, not a way to change baked constants. Upstream [maderix/ANE M5 results](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/m5result.md#L9) report that overwriting the original weight blob and unloading/reloading leaves output unchanged; `weightsBuffer` also failed to override constants. Resetting a bridge counter is not a demonstrated compiler-resource reset.

**Disk constraint:** h3's [cache store](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L108) explicitly excludes `weights/` and `model.mil`; its compiled `data` artifact embeds constants. Cache restoration uses hard links with copy fallback. Sources are written for cold compilation, and the original checkpoint still exists. Its [README](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/README.md) reports an additional daemon-owned compiled-model cache and roughly doubled disk cost. Those are upstream observations, not our measured footprint.

With approximately 45 GiB free reported by the parent, **do not provision a whole FP16 static cache on assumption**: two copies of the 23,813,160,960-byte weight budget alone are 44.36 GiB, before compiler expansion, metadata, staging or unrelated caches. FFNs alone doubled are 31.64 GiB. These calculations illustrate the constraint, not exact incremental usage; some existing daemon artifacts may already overlap, hard links may save space, or compilation may expand weights. First measure one representative artifact's logical/allocated bytes, daemon delta and transient peak. Use a durable manifest keyed by graph, weight digest, tensor ABI, OS build and ANE/compiler identity; upstream's temporary directory and identifier are not a sufficient product compatibility policy. Do not rotate models from SSD every token to solve residency: that moves loading onto the serial decode path.

ANEForge does **not** provide this shortcut as written. At pinned `caeef8edf13b9ec7a3338826daaa27f99e1663d1`, [compile_and_build_op](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_lib/ane_e5rt_dispatch.mm#L343) sets `e5rt_e5_compiler_options_set_force_recompilation(options, 1)` and calls `e5rt_e5_compiler_compile`. It then creates a precompiled compute operation from the returned program function. The word “precompiled” here does not establish a saved-library load-only API or compatibility with the `_ANEInMemoryModel` limit.

## 3. Static O+FFN tail: 114 programs, one input and output

Pack the current residual `h[3840]` and attended vector `a[4096 or 8192]` into one surface. Slice internally; all O/gate/up/down matrices, three norm gammas and the layer scalar are constants. The sole output is the updated hidden state. This avoids importing KV, masks or RoPE into the new graph. Exact algebra, following [current decode ordering](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:318), is:

```text
b = RMSNorm_post_attention(O(a))
r = round_fp16(h + b)
n = RMSNorm_pre_ffn(r)
f = Down(tanh_GELU(Gate(n)) * Up(n))
z = RMSNorm_post_ffn(f)
out = round_fp16(round_fp16(r + z) * layer_scalar)
```

RMSNorm uses actual gamma, epsilon `1e-6`, and current [FP32 statistics and final FP16 conversion](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/gemma_decode_math.rs:9). Preserve materialization boundaries and the accepted [explicit tanh GELU](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_ffn_layout.rs:110). A mathematically similar FP16 reduction, gamma-plus-one norm or native GELU is not an established equivalent. FP32 reduction support/performance inside this private compiled graph remains unresolved; this is the main reason to prefer the 116 path first.

Static convolutional FFN fusion has concrete precedent in [maderix/ANE test_full_fused.m](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/test_full_fused.m#L315), but that uses SiLU. h3's [block generator](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L166) demonstrates norm/reduction and broader fusion with different numerical choices. **Its block actually binds two external inputs** at [lines 785–788](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L785); it is not evidence that our single-input Gemma block works. Preserve the local [multi-tensor quarantine](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:191).

A full-layer graph could nominally reduce the count to 48+16=64, but would also need packed KV/context, position data, exact local/global QK/V norms and RoPE, attention scaling, ring semantics and compact new-KV output. No inspected source demonstrates that combination for Gemma. It is a later experiment, not the immediate FFN remedy.

## 4. Compression: retain the constant-weight opportunity

Start with static FP16 plus accepted GELU, then compare compressed variants independently of fusion:

- **Concrete private-MIL INT8 precedent:** h3 [ane_program_int8](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_linear.m#L105) emits `constexpr_affine_dequantize` with INT8 `[N,K,1,1]` data, FP16 **rank-one `[N]`** scales, axis zero and zero point zero, feeding constant-weight convolution. `h3_ane_linear_create_int8` is its callable constructor. The source notes rejection of `[N,1,1,1]` scales. Its checkpoint uses ConvRot; do not copy rotations into unrotated Gemma or infer Gemma accuracy from it.
- **Official palettization path:** `coremltools.optimize.coreml.palettize_weights(model, OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4)))` replaces constants with LUT operations. Exact source pin: apple/coremltools `181cbd341fa394ea53fd6895a2e0a46aaec27cb1`, [API implementation](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/coreml/_post_training_quantization.py#L188), [iOS18 constexpr_lut_to_dense](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L168). This is an offline artifact/reference API, not permission to introduce Python product runtime code, nor proof the current handwritten private MIL path accepts that exact compressed graph.

Apple states that [palettization latency gains require just-in-time decompression](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html); more LUT groups can cost speed. Its [optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) distinguishes preruntime decompression from runtime traffic reduction. A small source blob alone proves neither compressed residency nor faster evaluation. Inspect compiled bytes/residency and measure matched execution. Dynamic FP16 surface weights do not automatically receive constant-weight compression benefits. Quantization changes numerics and needs calibration/quality gates; retain the mixed BF16-prefill → FP16-decoder imported-KV oracle as well as end-to-end quality checks.

## Bounded decision sequence

1. Finish the parent's existing 115-program qualification. Capture quiet, matched static/dynamic FFN timing with the same real layer, input, GELU and sample policy. Record evaluate-call latency separately from proven hardware time.
2. Qualify the two proposed dynamic O shapes against the existing static O and FP32 accumulation reference. If their added cost is small, assemble the 116 arrangement with static FP16 FFNs. This preserves the most existing numerical evidence.
3. Evaluate one compressed static FFN and one packed static O+FFN tail separately. Record numerical error, latency, compiled/storage size and memory residency. Expand only the variant that improves the actual token critical path.
4. Validate cache restoration on a small model across fresh processes with compile fallback disabled, then one real static FFN. Test weight-specific identity, OS/ABI mismatch rejection, missing/corrupt artifacts, cleanup and disk peaks. Only then decide whether all-static 162-model residency is feasible. Cache work may be useful for startup even if the 114/116 design remains preferable.

All proposed hardware steps require the parent's active validation authority and scheduling. This research performed none of them.
