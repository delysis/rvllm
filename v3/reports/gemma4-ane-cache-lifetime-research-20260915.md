# ANE cache identity and caller-source lifetime

2026-09-15. Bounded source and existing-evidence review for macOS 15.6 / build 24G84 / AppleNeuralEngine 8.600.2. No hardware execution, builds, private selector calls, cache inspection/mutation, or implementation changes were performed. The experiment below is proposed, not executed. Multi-input graphs remain outside scope; preserve the qualified single-input/output ABI.

**Conclusion:** a changed signing identifier plausibly creates a different daemon cache namespace. It does not explain missing entries under the same identifier. Immediate source-file unlinking is numerically qualified for loaded execution, but its safety for persistent cache lifetime is not established. No inspected primary source provides a documented persistence barrier or proves that successful load means durable cache storage.

## What `csIdentity` means

Extracted ANECompilerService strings explicitly associate `csIdentity` with `SecTaskCopySigningIdentifier()` and separately associate `teamIdentity` with `SecTaskCopyTeamIdentifier()`. Apple's Security implementation obtains the former through `CS_OPS_IDENTITY`; XNU exposes the CodeDirectory hash through the distinct `CS_OPS_CDHASH` operation. This supports **signing identifier**, rather than CDHash or executable SHA, as the meaning of this field. The ANE extraction is from an iOS 18.5-to-26 comparison, not the exact installed macOS daemon; it does not establish every component of the 24G84 cache key. [ANE extraction, pin 826027d4463eab23b96c4325fa7154ef43c75acd](https://github.com/blacktop/ipsw-diffs/blob/826027d4463eab23b96c4325fa7154ef43c75acd/18_5_22F76__vs_26_0_23A5260n/MACHOS/ANECompilerService.md), [Apple Security, pin db15acbe6a7f257a859ad9a3bb86097bfe0679d9](https://github.com/apple-oss-distributions/Security/blob/db15acbe6a7f257a859ad9a3bb86097bfe0679d9/sectask/SecTask.c), [XNU, pin f6217f891ac0bb64f3d375211650a4c1ff8ca1ea](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/sys/codesign.h).

The local [signing inventory](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/power-measurement-20260915/executable-signing-identities.json) records three different CDHashes with `Identifier=rvllm_disaggregated_infer-9d1f7275eb5937c9`; the newer release uses `...-4ea237a076515df2`. A separately [ad-hoc-signed copy](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/power-measurement-20260915/stable-identity-signature.txt) restores the old identifier with another CDHash and [hits sliding attention](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/power-aware-worker-three/stable-identity-journal.jsonl). That supports namespace reuse across executable-content changes. The coordinator reports that the new identifier missed all 18 head/attention programs; that count is a reported observation, not independently re-executed here.

Use a deliberate, stable product signing identifier for future provisioning and use. Record it alongside executable SHA/CDHash, user, OS/framework build, complete model identifier, source digests and compile options. Neither equal signing identifiers nor equal graph hashes alone prove cache availability. Executable path is not established as the primary key.

## Source files and persistence are different questions

The [current constructor](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:309) retains source NSData in the model/descriptor path, queries the cache, stages source, and calls compile/load. After successful load it [verifies and removes `data`](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:433), then removes `weights/weight.bin`. On cache hits, those names are hardlinks to one payload. The [verification helper](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/source_staging.rs:8) establishes regular-file status, exact length and byte equality. It does **not** establish that a later daemon task will never reopen the pathname. Retained NSData and surviving open mappings do not answer that question either.

Two primary implementations use longer source lifetimes:

- [maderix/ANE, pin d91c9845c0784dec7753048954fc6d0e8411fe29](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m) keeps staging until `ane_bridge_free`, which calls unload before removing the directory.
- [h3, pin 6538b226efe5c4fff905d299dcc4426db2f44de4](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L99) preserves `data` and other staging entries through hardlinks/copies in a separate cache and restores them before load. Its comments call them compiled artifacts, but neither this implementation nor its marker proves that these are independently portable lowered programs. See the earlier [provenance correction](/Users/george/Downloads/rvllm/v3/reports/ane-daemon-cache-provenance-20260914.md).

These are lifetime precedents, not evidence that early unlink causes a race. Generated wrappers for `saveModelFiles` and compile/load merely expose selectors; they provide no durable-storage completion contract. Do not treat that selector as a flush operation. [tmc/apple, pin 872b4b8c75cb24617d606e0484c1db1e2948f01e](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_in_memory_model.gen.go).

## What the local failures establish

The [output provisioning journal](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/immutable-defaults-cache-refresh/output/driver-phases.jsonl) shows layer 44 compile completion, load completion and unload return. The [same-provisioning-executable strict check](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/immutable-exact-provision-path-cache-check/result.json) later records 135 hits followed by that identical model identifier missing, with zero compiles/evaluations. The provisioned graph spent approximately 34 ms between load completion and unload return. This narrows the failure beyond a simple changed-path explanation; it does not demonstrate a minimum lifetime requirement.

The global-attention identifier beginning `CAE7E1BA0E24C37F` hits in that [earlier exact-path journal](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/immutable-exact-provision-path-cache-check/driver-phases.jsonl), but misses in the later [stable-identifier journal](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/power-aware-worker-three/stable-identity-journal.jsonl). The full identifiers match, so changed global geometry is not the explanation for this pair.

Delayed persistence, source-path-dependent maintenance, and disk-pressure eviction remain hypotheses. The storage-maintainer wrapper exposes `purgeDanglingModelsAt:withReply:` but establishes neither scheduling nor eligibility rules. Falling free space or renewed allocation does not prove which mechanism occurred. [Pinned storage-maintainer wrapper](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_storage_maintainer_protocol_protocol.gen.go).

There is also an evidence gap: [the destructor discards unload's Boolean and NSError](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:596). `unload_returned` means the call returned, not that unload succeeded. Record the existing return/error before calling these clean unloads. No new private selector is required.

## One controlled experiment, at most two compiles

Use two cold, arithmetic-identical **O44-sized** fixtures, distinguished only by recorded graph names, under one fixed executable and signing identifier. The existing O44 failure makes this approximately 30 MiB FP16 source matrix a useful lower-cost target; it does not test the 177 MB INT8 FFN case.

1. Record both complete model identifiers, identical weight digest, MIL difference, signing identifier/CDHash, build, and free disk. Abort if either fixture is already cached or the agreed disk budget cannot accommodate the experiment. Never purge to make it cold.
2. Permit exactly two compiler calls total, zero evaluations, and only the qualified one-input/output graph ABI. Hold each loaded graph for the same five seconds. A uses current immediate verified source cleanup; B retains staging until unload. Keep order and timestamps explicit.
3. Record the already checked `compiledModelExists` result after load and before unload, plus actual unload Boolean/error. Record whether the framework itself removes staging; do not assume the client controls that lifetime.
4. Query from a fresh descriptor/process immediately and 30 seconds afterward using the same established selector. Query-only construction must avoid the production constructor's exclusive-directory-creation conflict while B's directory exists; do not bypass that ownership guard in the normal loader. Then perform strict-zero load-only verification through the normal constructor after exclusive owned staging is available. Stop on unexpected compilation or disk growth.

A reproducible A-miss/B-hit would implicate the early-cleanup interval. Both hitting leaves larger graphs and prolonged pressure unresolved. Both missing shows early unlink is not a sufficient explanation. A single pair is a diagnostic lead, not a production lifetime guarantee; it deliberately avoids repeated full-model provisioning and daemon cleanup.

## Critical-path priority

### Coordinator experiment result

The proposed pair was subsequently executed with source in
`crates/rvllm-apple-ane-sys/src/cache_lifetime_tests.rs`. See
[the receipt](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/cache-lifetime-pair-20260915/result.json).
Both 4096-input/3840-output fixtures used the same dense 31,457,408-byte source
blob and distinct operation names. Both were confirmed absent before their two
bounded compiler calls. No requests or evaluations were created. Each was held
for five seconds; the retained arm's two source paths remained present, and the
immediate arm's paths were absent. All six loads across the three processes
reported successful unloads through the now-checked existing return value.

Both entries hit in the first fresh process and again 63.388 seconds after the
compile process's last unload. These were strict load checks after descriptor
construction, not independent concurrent query-only probes while the retained
directory existed. No compile was permitted in either verifier. The boot time
was unchanged. This pair did not reproduce the loss, so it does not justify
changing production source lifetime or rule out larger-FFN/disk-pressure causes.

### Remaining priority

First make cache namespace and startup failure explicit, then isolate the lifetime issue above. Preserve the same-thread warm owner so successive prompts do not repeatedly pay setup. For cold setup, the [earlier source-preparation review](/Users/george/Downloads/rvllm/v3/reports/gemma4-ane-cached-preparation-host-review-20260914.md) measured historical all-hit preparation of 54.584/60.773 s, including 25.717/29.985 s in pre-constructor FFN work, versus only 1.373/1.451 s across all 162 load calls. Those journaled wall intervals are not isolated CPU timings or a retained-versus-phased performance comparison. Reusing exact prepared INT8 sources can remove repeated host quantization/packing, but a complete source package costs about 14.27 GiB in addition to checkpoint/daemon storage. It does not replace daemon-cache validation or justify more full provisioning under disk pressure. Startup/cache work and warm-token kernel optimization must be reported separately.
