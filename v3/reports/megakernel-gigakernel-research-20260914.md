# Megakernels, persistent inference, and the Gemma 4 12B split-device plan

Source review dated 2026-09-14. This investigation read public primary sources and local source/evidence only. It ran no GPU/ANE workload, compiler, benchmark, training job, or hardware test. No product code was changed. Published results below are authors' measurements, not independently reproduced results.

## Recommendation

Borrow **resident data, reusable schedules, precise dependencies, and specialized kernels** before attempting a whole-model persistent kernel. The current 12B workload needs competitive projection/attention implementations and an actual ANE decoder first. A megakernel does not eliminate reading a dense model's weights, and the historical ANE FFN observation already suggests a substantial bandwidth floor.

The most useful references are Hazy's task interpreter, Mirage's event-driven runtime, Cohere's September 2026 serving implementation, and MLX's actual Metal kernels. Cohere is especially instructive because it abandoned shared-memory paging in favor of simpler static per-operation pipelines. PDL reconstruction and Bonsai provide necessary counterexamples: persistence is not automatically the fastest implementation.

The local [progress report](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-metal-ane-progress-20260914.md) establishes 16 greedy-token agreement for one Metal BF16 case, 1.1415 tok/s, and real all-layer KV export. It does **not** establish consumption by an ANE decoder. ANE attention remains quarantined after the driver panic. This report does not authorize lifting that quarantine.

## What the names mean in the inspected projects

There is no consistent megakernel/gigakernel size boundary. Distinguish four properties explicitly: **fusion scope, kernel lifetime, scheduling mechanism, and storage location**.

| Implementation | Actual scope/lifetime | Scheduling |
| --- | --- | --- |
| Hazy low-latency Megakernels | Llama forward-pass operations in a kernel | Host-generated per-SM instruction streams; device interpreter and counters |
| Mirage Persistent Kernel (MPK) | Tensor task graph; runtime can continue across decode iterations and requests | Worker queues, event queues, dedicated scheduler warps, device batch/page management |
| Cohere | North Mini Code decode forward pass; native serving loop surrounds it | Mostly static per-SM schedule, dynamic attention/MoE work queues |
| Luce's current standard CUDA path | 24 transformer layers plus final norm in one dispatch; **separate LM-head dispatch** | Handwritten sequential layer loop and custom grid barrier |
| Mixture-of-Kittens / Alpha-MoE | A fused MoE component, not the complete transformer | Specialized compute/communication or projection pipelines |
| AutoMegaKernel / Bonsai | One token forward pass per launch | Host autoregressive loop; persistent/cooperative execution within a step |
| Enigma Qwen example | **One synthetic FP32 layer**, one threadgroup | Threadgroup barriers; no full-device scheduler |

I did not find a public primary-source implementation that establishes a distinct open-source category called **gigakernel**. Searches for that exact term largely led to secondary descriptions and unrelated uses. Do not infer an available port from the name. If the intended meaning is a device-resident, multi-token inference loop, MPK already provides inspectable source for that behavior: [`prepare_next_batch` and scheduler end-of-graph handling](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/include/mirage/persistent_kernel/persistent_kernel.cuh#L450-L708), [end-of-graph dispatch](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/include/mirage/persistent_kernel/persistent_kernel.cuh#L1244-L1288).

## Measurements with their actual boundaries

| Source | Hardware / model / precision | Workload | Published result and qualification |
| --- | --- | --- | --- |
| [Hazy, May 2025](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) | H100 and B200; Llama-3.2-1B; BF16 | Batch 1; prompt 32, generate 128; no speculation | H100 under 1 ms/forward, 78% bandwidth utilization; over 1.5× SGLang. B200 under 680 µs. Small-model decode, not 12B Apple inference. |
| [Hazy throughput, September 2025](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) | Tensor-parallel Llama-70B on H100s | 65,536 ShareGPT prompts through Tokasaurus; mixed serving workload | Over 22% higher end-to-end throughput than SGLang. This is a different objective from single-user batch-1 latency. |
| [MPK paper v1](https://arxiv.org/html/2512.22219v1) | A100/H100/B200; five dense/MoE models; BF16 | Offline batch maxima 1–16; prompt 64, output 1,024 | Batch-1 improvement 1.0–1.7× across evaluated cases. Qwen3-8B/A100: 14.5 → 12.5 ms/token. These bounded paper results are more useful than the repository's broad 1.2–6.7× headline. |
| [Cohere pinned README](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/README.md#L77-L119) | One H100, SM90a; North Mini Code; BF16; vLLM 0.24/FA3/Triton-MoE comparison | 8K context; real weights but **synthetic KV**, prefill disabled | Batch 1: 292 vs 185 tok/s; batch 8: 1,000 vs 757 aggregate tok/s. Separate real-prompt serving gains are smaller; prefill pauses decode. |
| [Luce current README](https://github.com/Luce-Org/lucebox/blob/1a7ccdfd26c95773542233e1b6d32225f18a89d5/optimizations/megakernel/README.md#L14-L49) | RTX 3090; Qwen3.5-0.8B; BF16 | Batch 1; pp520 / tg128 | 21,347 prefill tok/s and 413 decode tok/s; llama.cpp 11,247 / 267. Older forks/search results show 37,800 prefill: do not mix versions. The prefill implementation is separate from decode fusion. |
| [PDL reconstruction](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/blob/8664720072fefb62cc3a097c9c5bba8c936294c5/README.md#L94-L120) | H100 SXM5 80 GB; Llama-3.2-1B; matching BF16 operators, CUDA 12.8 | Batch 1; P32/D128; complete real-weight decode step | Persistent 983.127 µs; 81-kernel PDL chain 1,009.711 µs; default dependencies 1,169.071 µs. PDL gets 97.4% of persistent throughput. Separate synthetic-body numbers must not replace the full-step result. |
| [Bonsai](https://github.com/RightNow-AI/bonsai-turbo/blob/7003767c868bcef81b03a21df8e6098a2e68892e/README.md#L1-L34) | H100; Bonsai 27B; model-specific ternary / 1-bit packs; FP16 KV | Batch 1 decode, tg128; not a matched Gemma context experiment | Ternary: CUDA graph 151.1 vs megakernel 149.6 tok/s. One-bit: 133.0 vs 158.7. Fusion wins depend on the actual packing/kernel regime. |
| [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel/blob/a514bbc20a03bbf698a17443f8f14a27a617fc10/README.md#L32-L69) | Several NVIDIA GPUs and Llama-shaped cases | Batch 1, position 0 / low context | Reported W8A16 wins compare with **BF16** cuBLAS. Its equal-BF16 path trails cuBLAS; A100/H100 W8A16 cases also trail. This does not demonstrate an equal-precision universal megakernel advantage. |

Cohere's README contains an arithmetic inconsistency: the SciCode row says 711 vs 560 tok/s and 1.37×, although those two rates imply **1.27×**. Its other table rows and complete benchmark context should be examined individually rather than repeating the range as independently verified. The [authors' design post](https://cohere.com/blog/megakernels) also separates synthetic-cache decode, serving, and quality measurements.

Mixture-of-Kittens reports up to 2.37× MXFP8 forward and 1.92× BF16 forward for MoE layers on NVL72 hardware. Its benchmark sweep and production-training claim are not a single model/context/batch decode measurement; they are excluded from the comparable decode table. [Pinned scope and requirements](https://github.com/cursor/mixture-of-kittens/blob/22fc95ae6e331a738c4a58a227a8b03cac586e12/README.md).

## Implementation findings worth transferring

### 1. Hazy: overlap weight movement across task boundaries

The persistent kernel assigns warps distinct consumer, loader, storer, launcher, and controller roles. Two instruction stages let the next instruction prepare while the previous one finishes. Shared memory is explicitly partitioned into 16 KiB pages; the configuration requires 13 pages, and page reuse follows each operation's release order. Cross-SM data dependencies still involve global memory and counters. A whole model's weights and activations do **not** magically become on-chip resident. [Interpreter](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/megakernel.cuh#L118-L140), [resource constants](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/config.cuh#L7-L51), [page allocator](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/controller/page_allocator.cuh#L21-L70).

The throughput branch deliberately changes the instruction decomposition for GEMM rather than GEMV. It overlaps arithmetic, HBM traffic, and NVLink traffic; replicated O-projection weights trade memory for less communication. That trade is instructive but is not an ANE/GPU split-device recipe. [Attention-prefill source](https://github.com/HazyResearch/Megakernels/blob/91eaff262c2b473cfdcb135f5f2abefbe2835fe9/demos/cross-gpu-llama/attention_prefill.cu), [GEMM pipeline](https://github.com/HazyResearch/Megakernels/blob/91eaff262c2b473cfdcb135f5f2abefbe2835fe9/demos/cross-gpu-llama/matmul_pipeline.cuh).

**Transfer:** define explicit immutable weights, scratch lifetimes, producer completion, and consumer readiness. Recompute a small norm only when measured cheaper than materialization. **Do not transfer:** Hopper page counts, TMA, WGMMA, `setmaxnreg`, or the assumption that one resident block can be assigned to every Apple GPU core.

### 2. MPK: a real in-kernel scheduler, including iteration control

MPK's worker loop consumes task descriptors and publishes completion with release atomics; the scheduler consumes events with acquire loads and enqueues successors. Dedicated scheduler warps consume physical resources. The current online-pinned path additionally publishes token/progress/completion through system-scope acquire/release operations, tracks row ownership, and checks shutdown. This is substantive persistent serving control, not just fused mathematical expressions. [Worker loop](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/include/mirage/persistent_kernel/persistent_kernel.cuh#L821-L1100), [scheduler](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/include/mirage/persistent_kernel/persistent_kernel.cuh#L1119-L1288), [runtime descriptors](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/include/mirage/persistent_kernel/runtime_header.h#L324-L431).

The Python interface keeps attached tensors alive because their pointers enter generated code. It also supports kernel metadata/reuse, separate decode attention variants, and speculative-decoding integration. [Tensor ownership/reuse](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/python/mirage/mpk/persistent_kernel.py#L407-L430), [model integration](https://github.com/mirage-project/mirage/blob/1f3338f9ae084559726fb1fc898ae077ff7d171a/demo/qwen3/demo.py).

**Transfer:** persistent Rust model ownership, validated descriptors, schedule reuse, fixed shape families, bounded completion protocols. **ANE limit:** the inspected private model interface accepts a compiled graph/request; it does not expose MPK-style programmable resident worker queues. A fused MIL graph and an in-kernel scheduler are different capabilities.

### 3. Cohere: simpler pipelines can beat a general memory allocator

This September 8 release uses a fixed operation ABI, per-SM task descriptors, and a controller that prefetches descriptors into a ring. Operations have statically laid-out shared memory. GEMMs issue immutable weight loads **before** waiting for activations, and consecutive same-type tiles preserve pipeline phase rather than draining between tiles. Dynamic queue claimers balance attention splits and expert work. [Weight prefetch and dependency ordering](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh#L944-L1060), [controller](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh#L3070-L3110), [schedule construction](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/schedule.py#L1025-L1287).

Synchronization is not “free”: output publication uses async-proxy and device fences plus a counter increment; waiters spin then fence. Worker-only barriers exclude the independent controller. The launch fixes one block per requested SM and reserves almost the available shared memory. [Barrier implementation](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh#L515-L601), [launch](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/launch.cuh#L153-L166).

**Transfer:** immutable weight prefetch, context buckets, a small fixed ABI, and exact barrier accounting. Gemma's sequential attention/FFN structure is not Cohere's parallel-branch/MoE structure; do not copy its wave order.

### 4. Luce: inspect the actual dispatch boundary

The current standard path implements `AtomicGridSync`, with its launch grid capped by an occupancy query. It runs the 24 hybrid layers, then launches a separate vocabulary kernel. Thus “all layers in one dispatch” is true at this source pin, but “the entire token including LM head in one dispatch” is false. Intermediate activations and DeltaNet state still use global buffers. [Custom barrier](https://github.com/Luce-Org/lucebox/blob/1a7ccdfd26c95773542233e1b6d32225f18a89d5/optimizations/megakernel/kernel.cu#L134-L162), [launch boundary](https://github.com/Luce-Org/lucebox/blob/1a7ccdfd26c95773542233e1b6d32225f18a89d5/optimizations/megakernel/kernel.cu#L952-L1034).

Its prefill path uses cuBLAS GEMMs and separate tiled attention operations. Current NVFP4 code is a distinct Blackwell path; its existence does not change the standard BF16 experiment. [Prefill](https://github.com/Luce-Org/lucebox/blob/1a7ccdfd26c95773542233e1b6d32225f18a89d5/optimizations/megakernel/prefill.cu#L1536-L1565), [NVFP4 source](https://github.com/Luce-Org/lucebox/blob/1a7ccdfd26c95773542233e1b6d32225f18a89d5/optimizations/megakernel/kernel_gb10_nvfp4.cu).

**Transfer:** specialize each phase independently and stack compatible projections at load time. Keep the vocabulary projection separately tunable if its occupancy differs. Do not port a software global spin barrier to Metal from this code.

### 5. MoE kernels: broad fusion is not necessarily whole-model fusion

Mixture-of-Kittens divides communication and compute clusters, overlaps microbatches, and reuses macrobatch ring buffers with readiness/done signals. Its forward/backward scope targets Blackwell NVL72 MoE training. The useful ideas are bounded workspaces and explicit buffer reuse; multi-GPU token dispatch, backward recomputation, and MXFP8 training do not directly help this dense 12B decoder. [Forward scheduling](https://github.com/cursor/mixture-of-kittens/blob/22fc95ae6e331a738c4a58a227a8b03cac586e12/csrc/megakernel/forward.cuh#L1-L127), [workspace API](https://github.com/cursor/mixture-of-kittens/blob/22fc95ae6e331a738c4a58a227a8b03cac586e12/mok/functional.py).

Alpha-MoE instead fuses FP8 up/gate, activation quantization, and down projection. It requires specific scale granularity, interleaved weights, and a zeroed accumulation output. These constraints make it an instructive epilogue-fusion reference, not a drop-in dense Gemma GELU FFN. [Pinned interface/packing contract](https://github.com/Aleph-Alpha/Alpha-MoE/blob/0fbed62bdda3702fd5d4b8c497e68746f0304c3a/README.md), [implementation](https://github.com/Aleph-Alpha/Alpha-MoE/blob/0fbed62bdda3702fd5d4b8c497e68746f0304c3a/csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_acc.cu).

### 6. Validators and counterexamples

AutoMegaKernel's most reusable contribution here is its host-only schedule model: true join counters must await **all** producers, because a first-k counter does not identify which producers finished; per-SM serial queue order must also avoid cycles. On-chip buffers require co-located users. However, the implementation explicitly **skips transitive RAW/WAW checks above 8,000 tasks**. Its “provably correct” wording must not be treated as a universal correctness guarantee. [Invariants](https://github.com/RightNow-AI/AutoMegaKernel/blob/a514bbc20a03bbf698a17443f8f14a27a617fc10/schedule/ir.py#L20-L51), [coverage cap and placement checks](https://github.com/RightNow-AI/AutoMegaKernel/blob/a514bbc20a03bbf698a17443f8f14a27a617fc10/schedule/ir.py#L1036-L1055).

PDL reconstruction demonstrates that much of persistence's benefit can come from early admission of successor kernels and issuing their independent memory loads before waiting for predecessors. Its remaining gap includes resource lifetime/occupancy constraints. CUDA PDL has no direct Metal equivalence established here; the transferable lesson is to benchmark a simpler reusable dispatch graph before accepting a permanent device VM. [Experiment mechanism](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/blob/8664720072fefb62cc3a097c9c5bba8c936294c5/docs/reconstructing-a-megakernel-with-pdl.md#L107-L125).

Bonsai additionally shows how load-time retiling and packed GEMV dominate performance. Its cooperative launch checks actual occupancy/support. Ternary/one-bit model weights are a model-specific numerical regime, not a lossless conversion of Gemma BF16. [Cooperative path](https://github.com/RightNow-AI/bonsai-turbo/blob/7003767c868bcef81b03a21df8e6098a2e68892e/src/cuda/mega.cu#L749-L787), [packing/GEMV design](https://github.com/RightNow-AI/bonsai-turbo/blob/7003767c868bcef81b03a21df8e6098a2e68892e/README.md#L24-L34).

### 7. The Metal references are MLX, not Enigma's headline

Enigma's Qwen example says “single-layer” in its source. It uses random FP32 weights, one 256-thread group, and 33 cache positions; the host inserts the current K/V before launch. The printed “tok/s” is repeated evaluation of that one layer, not generated model tokens. It is a readable fusion demonstration but cannot validate a 12B whole-model design. [Scope/shapes](https://github.com/Klyne-org/Enigma-DSL/blob/fe33b39242be266261679b15518eaff7f8fcc62f/examples/qwen_megakernel.py#L1-L36), [host KV injection and timing](https://github.com/Klyne-org/Enigma-DSL/blob/fe33b39242be266261679b15518eaff7f8fcc62f/examples/qwen_megakernel.py#L360-L419).

MLX supplies direct Metal implementation references: tiled GEMV, vector attention with online-softmax accumulation, a two-pass attention alternative, and distinct quantized matrix-vector/matrix-matrix kernels. Review the dispatch eligibility as well as the shader: newer NAX paths must not be assumed available on M4/macOS 15.6, and head dimension 512 must receive its own supported-path audit. [GEMV](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/gemv.h), [attention reduction](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/sdpa_vector.h#L92-L168), [dispatch selection](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/scaled_dot_product_attention.cpp), [quantized kernels](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/quantized.h#L699-L824).

## The exact 12B cost model

The coordinating investigation derived the following counts from the official safetensors header; this review inspected the saved [weight-budget artifact](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/weight-budget.json). Bytes assume FP16 storage read once per token, excluding compiled expansion, KV, scratch, and extra passes.

| Dense weight group | Parameters | FP16 bytes/token |
| --- | ---: | ---: |
| 48 FFNs | 8,493,465,600 | 16,986,931,200 |
| Attention projections | 2,406,481,920 | 4,812,963,840 |
| Tied vocabulary head | 1,006,632,960 | 2,013,265,920 |
| Total above | 11,906,580,480 | **23,813,160,960** |

Apple advertises up to 546 GB/s unified-memory bandwidth for M4 Max. Dividing the above bytes by that advertised maximum gives **43.61 ms/token, or 22.93 tok/s**, as an optimistic weight-only bound. This is an arithmetic roofline, not a forecast, and specifically **not measured ANE bandwidth**. CPU/GPU/ANE consumers share system memory resources. [Apple specification](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/).

The coordinating investigation reported a **historical pre-panic** layer-0 constant-weight ANE fused-FFN median of 3.076792 ms, equivalent to 115.02 GB/s of FP16 weight bytes. The artifacts were lost in the reboot; this is not current acceptance evidence. If representative and serialized across layers, 48 FFNs alone cost **147.686 ms/token**, limiting the full decoder to below **6.77 tok/s** before attention, projections, head, or host work. At a hypothetical uniform 115 GB/s, all listed weights would cost **207.071 ms/token**. Dynamic-input matmul must be measured independently; constant-weight convolution timing does not establish its performance.

The current dynamic FFN surface is 354,140,160 bytes/layer versus 353,894,400 weight bytes. Its value is eliminating recompilation and per-step host repacking, not avoiding the device's weight read. Forty-eight such surfaces occupy about 17.0 GB, before other weights and duplicate backend representations. [Layout and packing](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_ffn_layout.rs:19).

For FP16 KV, a real 1,024-token ring for the 40 sliding layers requires **320 MiB**. The eight global layers require **16 KiB per live token** together. These counts exclude padding/surfaces. Current packed attention uses `spatial = 2*capacity + 32`, capped at 65,536 with 32-aligned capacity: its maximum is **32,736 tokens**, not 32,768. Moreover, masking a static-capacity matmul does not prove the compiler avoids processing the masked tail. [Capacity checks](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_attention_layout.rs:22), [MIL matmul/softmax extents](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_attention_layout.rs:199).

Using that maximum capacity for all 48 packed inputs would allocate **11,275,072,512 bytes**. Using 1,024 for the 40 sliding layers and 32,736 for the eight global layers would require **878,610,432 bytes**. These are layout calculations, not measured device allocations. Subsequent host work implemented the ring layout and wired the quarantined decoder wrapper; [host validation and its limits](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-metal-ane-progress-20260914.md) are recorded in the progress report. ANE execution remains unvalidated.

## What transfers to Apple, and what does not

| Technique | Metal prefill / fallback decode | ANE decode |
| --- | --- | --- |
| Load-time packing and resident immutable weights | Directly useful; tile layout must match actual M4 kernel | Directly useful, but compiled constants and dynamic surface weights have different costs |
| Fused norm/projection/activation epilogues | Useful if occupancy and rounding remain acceptable | Fused MIL subgraphs are plausible within validated operation/shape support |
| Online/split-KV attention | Strong kernel reference; specialize 256/512 dimensions and GQA | Cannot assume the private compiler selects a fused flash implementation from two matmuls + softmax |
| Cached task/dispatch topology | ICBs can amortize command encoding; query pipeline/device support | Reuse compiled program plus separately owned requests/surfaces, subject to validation |
| CUDA cooperative global barriers, TMA, WGMMA, register reassignment | No direct portable implementation established; requires redesign | No exposed equivalent in the inspected graph/request interface |
| One indefinite device loop | Poor first experiment on a display-driving GPU; progress/cancellation/residency must be proven | Not established by either private API's graph support |
| Simultaneous prefill and decode for independent requests | Potential throughput gain | Shared memory bandwidth/power can make simultaneous operation slower; measure whole-system behavior |

Metal's `threadgroup_barrier(mem_device)` synchronizes threads **within that threadgroup**; “device” specifies memory effects, not a whole-grid barrier. Concurrent command dispatch also requires explicit resource synchronization. These are reasons to prefer bounded dispatches over an improvised global spin protocol. [Apple synchronization guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon). Apple's [indirect command buffers](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu) and [indirect compute commands](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand) provide a public route to command reuse. They reduce encoding overhead; they do not imply CUDA PDL scheduling semantics.

Local source already encodes normal prefill and decode into **one command buffer per step**, with many internal operations. Recommending “one command buffer” as a new fix would duplicate existing work. It also explicitly prefers cooperative GEMV plus a separate RMSNorm for small batches because the single-threadgroup fused path under-occupies large projections. [Prefill batching](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4023), [decode batching](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4176), [fusion policy](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:296).

The pinned [maderix/ANE limitations](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/README.md#L187-L192) explicitly include the single-input constraint and approximately 119 compilations per process. Its INT8 constant dequantization and activation-cache claims do not prove compressed dynamic FFN inputs or full-model speed. [ANEForge](https://github.com/sbryngelson/ANEForge/tree/caeef8edf13b9ec7a3338826daaa27f99e1663d1) uses a different e5rt route; neither multi-input support nor fusion guarantees transfer automatically to `_ANEInMemoryModel`.

Apple documents that compression may either expand weights before runtime or decompress them during execution; only the latter can reduce weight traffic. Palettization is particularly worth investigating for ANE, but Core ML guidance does not establish support or selection in this private API. [Compression behavior](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html), [palettization performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html).

## Ranked optimization plan

All hardware measurements below are **future acceptance work**, not actions performed by this report. ANE work must remain paused until explicitly approved. Host-only implementation/review can continue under existing authorization.

1. **Make the critical-path budget observable.** Separate model load/packing/compilation, CPU encoding, queue wait, device execution, result collection, cache conversion, and sampling. Attribute QKV, attention, O, FFN, and vocabulary cost. Record actual dispatch/encoder counts, shapes, bytes, dtype, and compiled-program count. The first decision is whether 1.14 tok/s comes mainly from slow kernels, excess weight traffic, or orchestration; published launch savings cannot answer it.

2. **Complete a correct bounded decoder and reusable ownership model.** Use safe Rust types for compiled-program identity, request ownership, tensor layout, cache position, and completion tickets; isolate FFI in the existing sys boundary. Retain one compiled dynamic FFN program with separate layer requests/surfaces only after request-sharing correctness is established. Validate all 48 layers and the vocabulary head. Keep the 115-program proposal explicit: 48 QKV + 48 O + 16 vocabulary + 2 attention + 1 FFN. It leaves little headroom below an approximate upstream limit; budget temporary compiles, reloads, and context variants, and measure how failed attempts affect the limit. Do not rely on process restart as a correctness or stability solution.

3. **Optimize the weight path before broad persistence.** Compare constant-weight and dynamic-weight FFNs at exact 3840/15360 shapes, timing both device execution and host overhead. Confirm weights are packed once and only activation lanes change per step. Audit compiled weight expansion and hidden copies. Optimize the 2.013 GB vocabulary read too. For each improvement, report end-to-end tokens/s and joules/token; the 147.7 ms conditional FFN floor is the key challenge.

   Before fixing the 115-program arrangement, compare **static palettized FFNs** with the dynamic FP16 FFN: FFNs account for **71.3%** of the listed weight traffic. A proposed alternative is 48 static FFNs + 2 shape-shared dynamic QKV programs + 2 shape-shared dynamic O programs + 16 vocabulary tiles + 2 attention programs = **70 programs**, before context variants. It trades dynamic weights in smaller projections for potentially compressed dominant FFN weights. This is a program-count hypothesis; private-API compression, projection sharing, numerical behavior, and performance all require hardware proof.

4. **Keep Metal prefill a GEMM/attention problem.** Use appropriately tiled matrix kernels with enough threadgroups, and fuse cheap epilogues where they help. Audit the sliding 256-dimension and global 512-dimension attention paths separately. Compare against pinned MLX-style kernels at matched precision and prompt lengths. The existing six-token prompt timing is not a long-prefill throughput result. After kernel quality is competitive, evaluate ICB reuse if CPU encoding is material; preserve all dependency and scratch-lifetime rules.

5. **Finish the cache handoff and bound attention work.** Consume the real exported cache, track absolute RoPE positions, and import only valid entries. The subsequently implemented host ring still needs ANE validation. Choose a supported global-capacity policy; static context buckets trade padding against more compiled programs. Count those variants before changing the 115-program design. Prove GPU completion before host/ANE access; unified allocation is not sufficient proof of zero-copy or synchronization. The current 36.89 ms capture and 220.08 ms packing/hashing/writing figures are handoff observations, not ANE execution.

6. **Evaluate compression with a separate numerical contract.** Weight compression has a stronger potential effect than removing a handful of launches for a 23.8 GB/token dense read. Measure actual transferred bytes and dequantization cost; retain group scales and FP32 reductions where required. Start with W8 before aggressive W4 as an engineering hypothesis, not a quality guarantee. Verify whether the chosen ANE route really keeps weights compressed during execution; compile-time expansion can erase the expected bandwidth gain. Preserve the BF16 reference path.

7. **Only then consider speculation, overlap, or persistence.** Speculation can amortize target weight reads across several verified candidates, but the present FFN graph selects one activation column and is not a multi-token verifier. Require target verification, rejection/rollback of both cache families, and compatible model/chat semantics. No trained Gemma 12B drafter was established by this investigation. For one request, GPU prefill must finish before ANE continuation, so disaggregation alone supplies no pipeline overlap. For multiple requests, test whether overlap improves aggregate throughput without worsening first-token/decode latency or power. A whole-model Metal scheduler is last, contingent on measured residual dispatch stalls and a documented progress/synchronization design.

## Numerical and acceptance requirements

The coordinating investigation's independently loaded CPU references retain FP32 RoPE buffers and still select different first tokens for the raw prompt: BF16 9079 versus FP16 496. Two correctly formatted chat prompts agree on the first token. This is a real precision boundary, not evidence that FP16 always fails or that two chat examples establish equivalence. The local progress report records the evidence scope.

BF16 Metal prefill followed by FP16 ANE decode is also a **mixed numerical path**. Build an oracle that starts with the exact exported, converted KV tensors and then executes the reference decode equations. Pure FP16 end-to-end upstream generation is not a definitive oracle for BF16-prefilled KV. Evaluate both this exact handoff boundary and end-to-end quality.

Preserve actual Gemma gamma, FP32 RMS statistics, Q/K normalization, unit attention scale **1.0**, GELU's correct approximation, proportional global RoPE, residual/layer scaling, and the global K projection's separate raw-V use. Do not copy Qwen/Llama `1/sqrt(d)`, SiLU, gamma-offset, head geometry, or full-dimensional RoPE assumptions. Fusion can remove FP16 rounding boundaries and change reductions even when all algebra looks equivalent.

The minimum future decision set is:

- Layer-by-layer outputs against independently loaded BF16 and FP16 references, especially logits/top-k margins and first divergence; finite-value/range checks do not substitute for numerical agreement.
- Real prompts with the actual chat template; multiple prompts and multiple generated tokens. Include positions 0/1/511/1023/1024/1025, sliding wrap, partial pages, and the largest supported global capacity.
- Reused request/surface tests with different layer weights, different contexts, sequential requests, and explicit completion ownership. Include shape/stride/size failures that must return before FFI.
- Paired ablations: existing dispatch, optimized individual kernels, selected epilogue fusion, cached dispatch, and only then persistence. Hold weights, precision, attention context, sampling, warmup, and output criteria constant.
- Durable hardware identity and stage logs for any approved ANE experiment; report first compile, warm execution, tail latency, whole-generation time, actual bytes, and power separately. Small attention success must precede a layer; a layer must precede 48-layer decode. No hardware run occurred during this research.

## Immutable source pins

Pins were resolved from GitHub on 2026-09-14. A pin identifies the source inspected, not the code revision used for every historical published chart.

| Repository / branch | Commit | Commit date |
| --- | --- | --- |
| [Hazy Megakernels / main](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e) | `7309cec801537b61fea3b50d7dfe454a6cde578e` | 2025-06-02 |
| [Hazy Megakernels / throughput](https://github.com/HazyResearch/Megakernels/tree/91eaff262c2b473cfdcb135f5f2abefbe2835fe9) | `91eaff262c2b473cfdcb135f5f2abefbe2835fe9` | 2025-09-28 |
| [Mirage / mpk](https://github.com/mirage-project/mirage/tree/1f3338f9ae084559726fb1fc898ae077ff7d171a) | `1f3338f9ae084559726fb1fc898ae077ff7d171a` | 2026-09-12 |
| [Cohere / main](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c) | `67d0b9ca22ea3652796b715d1d1863459e0e2c3c` | 2026-09-08 |
| [Lucebox / main](https://github.com/Luce-Org/lucebox/tree/1a7ccdfd26c95773542233e1b6d32225f18a89d5) | `1a7ccdfd26c95773542233e1b6d32225f18a89d5` | 2026-09-14 |
| [Mixture-of-Kittens / main](https://github.com/cursor/mixture-of-kittens/tree/22fc95ae6e331a738c4a58a227a8b03cac586e12) | `22fc95ae6e331a738c4a58a227a8b03cac586e12` | 2026-08-14 |
| [Alpha-MoE / main](https://github.com/Aleph-Alpha/Alpha-MoE/tree/0fbed62bdda3702fd5d4b8c497e68746f0304c3a) | `0fbed62bdda3702fd5d4b8c497e68746f0304c3a` | 2025-12-10 |
| [AutoMegaKernel / main](https://github.com/RightNow-AI/AutoMegaKernel/tree/a514bbc20a03bbf698a17443f8f14a27a617fc10) | `a514bbc20a03bbf698a17443f8f14a27a617fc10` | 2026-06-29 |
| [Bonsai / main](https://github.com/RightNow-AI/bonsai-turbo/tree/7003767c868bcef81b03a21df8e6098a2e68892e) | `7003767c868bcef81b03a21df8e6098a2e68892e` | 2026-07-16 |
| [PDL reconstruction / main](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/tree/8664720072fefb62cc3a097c9c5bba8c936294c5) | `8664720072fefb62cc3a097c9c5bba8c936294c5` | 2026-08-05 |
| [Enigma / main](https://github.com/Klyne-org/Enigma-DSL/tree/fe33b39242be266261679b15518eaff7f8fcc62f) | `fe33b39242be266261679b15518eaff7f8fcc62f` | 2026-07-23 |
| [MLX / main](https://github.com/ml-explore/mlx/tree/d9add9d11f3154111a4c85f267ec2fd307ecd18e) | `d9add9d11f3154111a4c85f267ec2fd307ecd18e` | 2026-09-14 |

The [Event Tensor paper](https://arxiv.org/abs/2604.13327) and [Ada-MK paper](https://arxiv.org/abs/2605.11581) are relevant newer compiler research: dynamic dependency representation, and schedule/shared-memory search with phase-specific inference. No corresponding public implementation was established in this bounded review, so neither is presented as code ready to port. The selected implementations above provide sufficient concrete evidence for the recommended next steps.

## 2026-09-15 follow-up: cache reliability and startup cost

The original plan above is a dated research baseline. Subsequent work qualified a one-input/output cached static decoder and a reusable same-thread Metal/ANE owner; current acceptance belongs to the detailed evidence reports, not the original 115-program proposal. The [cache identity and source-lifetime follow-up](/Users/george/Downloads/rvllm/v3/reports/gemma4-ane-cache-lifetime-research-20260915.md) now puts stable signing identity, strict cache availability, and a bounded two-fixture lifetime check ahead of further full-model provisioning. Primary sources connect ANE `csIdentity` to the signing identifier rather than CDHash, but same-identity cache disappearances remain unexplained. Successful load and byte-identical staging do not establish durable cache persistence; immediate source unlinking, maintenance, and disk pressure must not be conflated.

For the startup critical path, the [host-preparation review](/Users/george/Downloads/rvllm/v3/reports/gemma4-ane-cached-preparation-host-review-20260914.md) found historical all-hit setup dominated by source conversion/quantization/packing, with only 1.37–1.45 s across 162 framework load calls. Keep the warm owner reusable and consider exact prepared INT8 source reuse with an explicit disk budget; those artifacts are not lowered programs or authority to bypass daemon checks. This priority concerns startup reliability and latency. Warm decode still requires measured weight-path improvements, and broader dispatch fusion cannot remedy repeated source preparation or a missing compiled entry. Earlier sequential residency timings do not establish a hardware penalty without paired evidence.
