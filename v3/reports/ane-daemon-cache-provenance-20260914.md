# ANE daemon cache: provenance and scoped lifecycle candidates

Source-only follow-up, 2026-09-14, targeting the parent's macOS 15.6 / AppleNeuralEngine 8.600.2 investigation. No private selector calls, hardware execution, builds, local cache inspection/mutation, or other users' model-content access were performed. Earlier reports are preserved; the correction below supersedes their interpretation of h3's saved files.

**Conclusion:** caller staging and daemon-compiled storage must be tracked separately. Upstream exposes plausible per-model cache-query/purge selectors and path-discovery selectors, but I found no primary source establishing their exact ABI, permissions, completion semantics or successful eviction on framework **8.600.2**. The existing model object is the strongest candidate for a scoped lifecycle operation; neither deleting staging nor calling unload is proof that daemon disk storage was reclaimed.

## 1. Correction: h3 does not prove that its saved files are lowered programs

The parent found that `net.plist` is byte-identical to `model.mil`, and `data` retains the original weight-container header and length: 353,894,656 bytes for FP16 and 177,016,768 for INT8. These files therefore establish **source staging**, not the ANE's lowered weight representation. The modest INT8 timing improvement does not change that conclusion.

At maderix/h3.c-ane pin `6538b226efe5c4fff905d299dcc4426db2f44de4`, [bridge_cache_store](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L108) copies/hard-links every caller-staging file except `weights/` and `model.mil`, then writes its own `compiled.ok` marker. It does **not** identify `model.hwx`, parse a compiled container, inventory daemon storage, or test whether preserved `data` is merely the original blob. Its comment calls these compiled artifacts; the implementation does not prove that designation.

The [load path](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_bridge.m#L181) restores those files and calls `loadWithQoS:options:error:` before falling back to compile. This proves a client-side load-before-compile strategy exists. **It does not prove the copied files independently contain everything necessary to recreate a lowered program.** Success can depend on an already populated daemon cache. h3's [rotation test](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/tests/test_ane_full_block.c#L437) unloads/reloads and compares outputs; it never demonstrates restoration after the matching daemon entry has been removed.

Accordingly, the [README's roughly 19 GB cache and doubled disk-cost statements](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/README.md) remain the author's storage observations. They are not an audited inventory separating staging copies, daemon source storage, lowered weights and programs. My earlier phrase “real compiled-artifact cache” was too strong: **staging preservation plus apparent daemon-cache reuse** is the supported interpretation pending stronger evidence. The earlier ANEForge `weights.bin` measurements likewise measure source blobs.

## 2. Where lowered programs may be stored

The clearest primary historical path is Mohamed Ghannam's 2022 research, [author's slides, pages 89–96](https://github.com/0x36/weightBufs/blob/3b73a86a20ee1213469f69ff34d9eda95e3a41b2/attacking_ane_poc2022.pdf):

```text
/Library/Caches/com.apple.aned/<build>/InMemoryModelCache/<csIdentity>/<model>/model.hwx
```

The slides distinguish source `net.plist` from compiled `model.hwx`; name daemon `_ANEInMemoryModelCacheManager cachedModelPathMatchingHash:csIdentity:` as the identity-aware lookup; and identify `_ANEStorageHelper memoryMapModelAtPath:isPrecompiled:modelAttributes:` as the mapper. **This is historical macOS 12-era evidence, not a verified macOS 15.6 path or an application-callable cache manager.** Only the cache layout/lifecycle evidence is relevant here; its exploit techniques are unrelated to this task.

For version-sensitive path discovery, current generated bindings provide better candidates than assuming a fixed path. At tmc/apple pin `872b4b8c75cb24617d606e0484c1db1e2948f01e`, inspect these `_ANEStrings` class selectors:

| Selector | Source location | Potential use; not invoked here |
|---|---|---|
| `buildSpecificModelDataVaultDirectory`, `buildSpecificUserModelDataVaultDirectory` | [lines 97–104](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_strings.gen.go#L97) | Resolve build-scoped system/user storage candidates |
| `inMemoryModelCacheName` | [line 173](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_strings.gen.go#L173) | Obtain the cache-component name |
| `modelBinaryName`, `modelDataVaultDirectory`, `modelSourceStoreName` | [lines 213–232](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_strings.gen.go#L213) | Distinguish binary/source/storage naming |
| `systemModelsCacheDirectory` | [line 301](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_strings.gen.go#L301) | Resolve a system cache candidate |
| `userModelDataVaultDirectory` | [line 405](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_strings.gen.go#L405) | Resolve a user storage candidate |

Bindings establish selector names, not that each exists or is side-effect-free on 8.600.2. They do not supply returned paths. `_ANEInMemoryModel localModelPath`/`modelURL` are also exposed, but their names do not establish that they point beyond caller staging.

The parent reports readable-but-empty `/private/var/db/neuralengine`, an uninspected `_coreml`-owned directory, and both system/user ANE daemons and storage-maintainer processes. Those observations do not locate an actual compiled entry. The source offers no basis to inspect unrelated model contents or broaden filesystem permissions.

## 3. Model-scoped query, unload and purge candidates

These are distinct operations. The generated bindings are a primary source for wrapper implementation, not Apple documentation or successful behavioral tests.

| Receiver and selector | Source evidence | Boundary |
|---|---|---|
| Existing `_ANEInMemoryModel`: `compiledModelExists` | [wrapper lines 225–227](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_in_memory_model.gen.go#L225), Boolean wrapper result | Best model-object-scoped existence-query candidate; unknown exact cache partition/meaning |
| Existing `_ANEInMemoryModel`: `unloadWithQoS:error:` | [lines 283 onward](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_in_memory_model.gen.go#L283); already used locally | Releases loaded-model resources; does not establish persistent eviction |
| Existing `_ANEInMemoryModel`: `purgeCompiledModel` **with no colon/argument** | [lines 276–278](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_in_memory_model.gen.go#L276) | Most narrowly targeted purge candidate; no success/error result in wrapper and no tested 8.600.2 contract |
| `_ANEClient`: `compiledModelExistsFor:`, `compiledModelExistsMatchingHash:` | [lines 257–263](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_client.gen.go#L257) | Model/hash queries; do not assume the hash equals a hexadecimal string or uniquely names every client partition |
| `_ANEClient`: `purgeCompiledModel:`, `purgeCompiledModelMatchingHash:` | [lines 488–493](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_client.gen.go#L488) | Wrapper discards return value; model type and ownership/partition semantics need confirmation |

There is an ABI warning: mdaiter/ane's [README at e13f458…](https://github.com/mdaiter/ane/blob/e13f45818cbc8edeb6627d86a547f6e10d3883a6/README.md#L977) declares client `purgeCompiledModel:` as returning BOOL, whereas tmc exposes no result and internally uses a generic Objective-C return type. **Do not derive the Rust FFI signature from either alone.** Exact local selector presence and Objective-C type encoding must precede any future invocation; a missing method must not silently become “cache absent.”

mdaiter's [daemon protocol](https://github.com/mdaiter/ane/blob/e13f45818cbc8edeb6627d86a547f6e10d3883a6/README.md#L393) also lists `purgeCompiledModel:withReply:`. This does not prove direct XPC access or reply semantics for our process. Its entitlement list distinguishes all-partition model purge and storage maintenance. The broader `purgeDanglingModelsAt:withReply:` [storage-maintainer protocol](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_storage_maintainer_protocol_protocol.gen.go#L35) is not an established per-owned-model cleanup mechanism and is not recommended here.

## 4. Identity interpretation and the next bounded decision

The parent reports byte-identical old/current original-control MIL and the same graph identifier, yet a new probe executable's first original/dense/INT8 runs added roughly 355/355/185 MB persistent storage; later same-binary runs added no meaningful disk. Historical `csIdentity` partitioning makes a client/executable namespace **plausible**. It is not proof: code-signing identity, resolved source path, options, user/system partition and compiler cache policy could contribute. Equal graph hashes alone do not establish equal daemon keys.

For the parent's read-only inventory, record **owned** graph/weight identity, exact executable/code-signing identity, source URL, options, OS/framework build, observed compile/load state and storage deltas. Query/path candidates should be checked against the exact installed ABI without calling purge. An owned model's missing cache query after unload would still require interpretation before claiming disk reclamation.

If a later, separately authorized eviction experiment is needed, use one disposable **owned** model object with no live requests, an established existence result, and known identity/partition. Verify the postcondition and reclaimed storage; do not infer success from a void return or process exit. Shared cache entries may serve other owners, so automatic purge in every model destructor is not justified by these sources. Broad cache deletion, service termination, entitlement changes and touching unrelated entries are outside this report's recommendation.

The current performance conclusion remains limited: the parent's INT8 median improves versus its dense reconstruction under varying load, but staging-size ratios cannot establish the lowered representation, exact runtime bandwidth reduction, or an available safe eviction API.
