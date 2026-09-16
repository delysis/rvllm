# Gemma 4 12B: single-I/O ANE multi-token verification feasibility

Source review, 2026-09-16. No accelerator calls, builds, cache operations, or model downloads. The target remains the qualified standard-checkpoint path: Metal BF16 prefill, INT8 ANE FFNs, other FP16 ANE projections. The newly host-tested sliding INT8 QKV option does not change this review's acceptance baseline.

Subsequent implementation note: the coordinator added a bounded logical-batch
FFN constructor and packing interface, with host-only checks and unchanged S=1
MIL. See [the implementation receipt](gemma4-int8-ffn-logical-batch-20260916.md).
The local source pointers/hashes below describe the reviewed pre-batching
snapshot. The subsequent two-token FFN qualification now has a device result:
84,480 values bit-identical to serial, with timing still pending. Batched
linear interfaces and live-input diagnostics have also been implemented; see
[that follow-up](gemma4-batched-projections-and-live-inputs-20260916.md).
No full target verifier is established.

**Recommendation:** a two-logical-token, real layer-0 INT8 FFN is a justified next component experiment, after the current comparison campaign. Full speculative decoding is feasible in principle within one external input/output, but is not a small extension of the existing decoder. First establish useful target-side batching; defer a model-wide compilation campaign and drafter integration until that gate passes. No M4 multi-token speedup is established here.

## What the present code establishes

| Boundary | Source evidence | Consequence |
|---|---|---|
| Linear tensor geometry | [ane_linear.rs:26](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:26) rounds physical spatial stride to 32 FP16 elements; [MIL:166](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:166) independently declares logical `S`. | A 64-byte channel row does not prove 32 computed tokens. |
| Linear wrapper | [project:129](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:129) accepts one vector and writes/reads lane zero, even when compilation accepts larger `S`. | Neither existing calls nor their larger-spatial tests qualify independent-token batching. Preserve logical token count separately from physical stride in any future wrapper. |
| FFN | [INT8 constants:272](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:272), [I/O:307](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:307), [MIL:352](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:352): three static 1×1 convolutions, explicit FP16 tanh GELU, all logical tensors `S=1`; padded I/O uses 32. | Independent token columns can mathematically share these weights. Changing `S` requires a new graph and numerical/device qualification; filling unused physical bytes in today's graph is insufficient. |
| Attention | [layout:4](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_attention_layout.rs:4), [MIL:226](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_attention_layout.rs:226): initial width lanes represent GQA groups—2 sliding, 16 global—followed by K/V and one mask row. | These are head groups, not token positions. Parallel queries need distinct causal masks. The present broadcast mask cannot safely verify a future-token block in one attention call. |
| Cache mutation | [attention decode:167](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_attention.rs:167) writes K/V before evaluation, then advances one position. [decoder:753](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:753) invalidates its frontier before starting all 48 layers. | No current commit/truncate/rollback operation exists. Preserve the fail-stop behavior on partial failure. |

An inspectable upstream precedent is the field guide's [four independent width-lane inputs](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/tests/test_w_lane_batching.cpp#L33-L104), compared with separate matvec evaluations. This establishes an implementation pattern, not compatibility or measured bandwidth for our M4, INT8 encoding, H3840/I15360 FFN, or request wrapper.

Spatial batching creates reuse opportunity, not proof that weights are fetched once. Compiler tiling, intermediate tensors and actual lowered weight representation remain unknown. Source blob bytes divided by latency are an effective rate, not a measured DRAM counter. Repeated copies of the same activation only test replication; use distinct vectors.

## A concrete drafter exists, but requires shared target state

Google publishes the **standard-target** [gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant/tree/46d4c6f13f0ac0ad827b915669b8df9b81c64c51), revision `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`. Its [model card](https://huggingface.co/google/gemma-4-12B-it-assistant/blob/46d4c6f13f0ac0ad827b915669b8df9b81c64c51/README.md) explicitly pairs it with `google/gemma-4-12B-it`, rather than a QAT target; the card declares Apache-2.0. This corrects the earlier broad report's statement that no trained 12B drafter had been established.

The pinned [configuration](https://huggingface.co/google/gemma-4-12B-it-assistant/blob/46d4c6f13f0ac0ad827b915669b8df9b81c64c51/config.json) specifies BF16, four layers, hidden 1024, FFN 8192, backbone hidden 3840, vocabulary 262144, and all four layers KV-shared. It has three sliding layers (16 Q heads, 8 KV heads, D256, window1024) and one global layer (16 Q heads, 1 KV head, D512, 25% proportional RoPE). `use_ordered_embeddings=false`: do not assume a centroid-pruned vocabulary head. The [repository metadata](https://huggingface.co/api/models/google/gemma-4-12B-it-assistant/revision/46d4c6f13f0ac0ad827b915669b8df9b81c64c51) reports 422,856,964 BF16 parameters, about 846 MB of coefficient payload; this is metadata, not downloaded or resident-model measurement.

vLLM revision `3bb782621492711485dc86791b5978128783814a` explicitly supports this [12B unified-assistant model type](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/docs/features/speculative_decoding/mtp.md#L12-L36). Its implementation:

- Shares target input embeddings but retains the assistant's own 1024-dimensional vocabulary head: [proposer:171–204](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/gemma4.py#L171-L204).
- Maps assistant attention to the last target layer of each attention type: [KV mapping:306–360](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/gemma4.py#L306-L360).
- Uses target/feedback hidden states and target KV; holds draft positions and sequence lengths constant between proposal steps: [proposer:33–70](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/gemma4.py#L33-L70), [model forward](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/model_executor/models/gemma4_mtp.py#L401-L436).

Thus it is a credible trained assistant, not an independently runnable tiny autoregressive substitute. A local implementation must preserve these interfaces, expose the correct target hidden-state boundary, and mirror/share only the required target KV with a drafter backend. Current ANE attention has no exported resident-input read API; target K/V are available on the host when computed, so maintaining selected committed KV mirrors is a possible design. Cross-device packing, draft latency, extra residency and acceptance against our INT8/FP16 target are **unmeasured**. Mixed target arithmetic may lower acceptance. Google's generic speedup/quality description is not an M4 measurement or proof of parity with rvllm's numerical path.

## Verification without a new attention graph

For a known candidate block, process layers in order. Within each layer: batch QKV columns; apply the existing norms and absolute-position RoPE independently; run the existing attention request sequentially in token order; batch O and FFN columns while preserving the intervening normalization/residual boundaries; then finish each token's post-norm/residual/scalar. Batch the vocabulary tiles too if their weight traffic is to be amortized. The current layer pipeline and head are at [gemma_ane_decode.rs:789](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:789) and [875](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:875). Layers cannot be independently batched together: their residuals depend on previous layers and their weights differ. Preserve raw global K-as-V before K norm/RoPE, both Q/K geometries, FP16 boundaries, and probability scaling in existing attention.

The smallest useful speculative cycle is **one draft plus a bonus prediction**. If `x` is the already sampled token absent from KV, and `d` is the assistant proposal after `x`, evaluate inputs `[x,d]`. First output verifies `d`; second supplies a bonus token only if `d` matched. On rejection, commit only `x` and emit the first target output; discard the second output/state. On acceptance, commit both inputs and emit `d` plus the bonus. EOS/context/output limits can shorten this work. This explicit frontier avoids counting two evaluated tokens as two accepted advances.

Before that can be correct, all 48 caches need one transaction. For an initial verifier constrained to `position + 2 <= 1024` with the current global capacity, rejected new slots have not overwritten old sliding KV: restoring committed counts/masks and masking the unused tail can suffice, but the APIs and tests do not exist. If later allowing absolute positions beyond a sliding ring, restore overwritten K/V too; cursor rollback alone is wrong. Two-token undo payload across these 48 geometries is 688,128 bytes, excluding masks/metadata. The existing [sys wrapper:530–584](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:530) reads output surfaces only; do not invent an input-read capability or private selector. A maintained host mirror or a separately reviewed bounded owned-surface operation would be needed. Failed partial evaluation must leave the decoder unusable until safe restoration/reimport.

Exact greedy output preservation requires batch numerics to retain the serial target's argmax, including softcap/tie behavior; mathematical independence alone does not prove this after compiler shape changes. Stochastic sampling would require a separate correct rejection sampler and probability interface, outside this proposal.

## One bounded experiment and decision gate

Use the existing real layer-0 INT8 coefficients, scales and zero points unchanged. Compile **one** candidate: logical input/output `[1,3840,1,2]`, intermediate tensors `[1,15360,1,2]`, unchanged three-convolution/tanh-GELU ordering and opset. Keep a single external input and output. For two logical columns, proposed physical stride remains 32 FP16 elements; pack token `j`, channel `c` at byte `2*(32*c+j)`. Each I/O allocation remains 245,760 bytes. Validate the backend's actual stride instead of assuming this from allocation alone. Do not combine this test with stacked gate/up, new quantization, or new runtime options.

Compare one two-column call against two serial calls, using captured distinct real activations plus lane-swap/zero-isolation controls. Check every output against serial INT8 and independent exact-reconstruction CPU references; retain existing backend numerical tolerances and finite-value checks, including large tanh-GELU inputs. Only after correctness, use the queue's matched conditions and rotating order to report synchronized call time, host packing/read time, whole-block latency and latency per useful token. Require a reproducible saving over two serial calls before expanding. This is component evidence, not speculative throughput.

For the one-draft cycle, with acceptance probability `p`, serial decode latency `T1`, batched verification `T2` and draft/transaction overhead `D`, expected steady-state benefit requires `(1+p)*T1 > T2+D` (ignoring EOS truncation). Merely showing `T2 < 2*T1` is insufficient. If only FFNs are batched, sequential attention/projections/head limit savings. Every additional logical-shape graph may add compiled storage/residency; cache identity/lifetime and lowered weight sharing are not guaranteed.

**Defer full integration** until the component wins, batched target argmax/rollback gates are designed, a local assistant adapter and acceptance corpus are available, and their measured cost satisfies that inequality. No evidence here warrants another full cache provisioning campaign, a multi-I/O path, or a model/precision change.

## Local source identity

These are working-tree observations, not claims about a clean commit. SHA-256 at review:

```text
ane_linear.rs           5f0ea2837d6de78dd03fe83dbf7edc87f009ae9ba80227c9e40b1d0408ee4663
ane_attention.rs        4cf9b757b5b832f3fa29fb39dbbace0187d09b3e26c089b806520ad7630020b0
ane_attention_layout.rs 5c4833b49a066c5d7bfe8dfb641fbb6b20d064a161388f72d7c06bafb010154e
gemma_ane_decode.rs     e9ad4a9b90fedc2a0e657f96991faeb27e88148ece3ce77008ba58b219ca38bb
in_memory.rs            c3257ecf04589803c94d0355315b53eab2e3102bbf5f69ce95d6d3948a39d305
```
