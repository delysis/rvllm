# AppleInference Swift package

This package is the native Swift boundary for the rvllm Apple runtime. It
supports macOS 15+ and iOS 18+ and exposes `AppleInferenceEngine`, an actor whose
`generate(_:)` method returns `AsyncThrowingStream<TokenEvent, Error>`.

The v2 initializer requires a local model-package URL and optionally accepts a
separate resource-bundle URL. The package manifest authenticates every weight,
tokenizer, metallib, and pipeline manifest before backend creation. The native
library creates the built-in continuous Metal worker on macOS and iOS when no
worker factory is injected. Injected factories remain available for tests and
host-specific integrations. If the package is invalid, its authenticated
resources are incomplete, or the real Metal backend is unavailable,
initialization fails explicitly. No test or synthetic token path exists in the
shipping C/Swift API.

```swift
let engine = try AppleInferenceEngine(
    modelURL: Bundle.main.url(forResource: "Model", withExtension: "rvllm")!,
    resourcesURL: Bundle.main.resourceURL
)
```

Encrypted persistent-cache host material is opt-in. The default memory-only
configuration uses ABI v2 and performs no Keychain or persistent-cache
filesystem access. To request T3, the host must provide explicit consent, a
bounded quota, a stable host identifier, and an authenticated engine-level
tenant namespace:

```swift
var config = AppleEngineConfig()
config.cachePolicy = .persistentEncrypted
config.persistentCacheConsent = true
config.persistentCacheBytes = 512 * 1024 * 1024
config.persistentCacheHostIdentifier = "com.example.MyApp"
config.persistentCacheNamespace = authenticatedTenantID

let engine = try AppleInferenceEngine(
    modelURL: modelURL,
    resourcesURL: Bundle.main.resourceURL,
    config: config
)
```

The tenant namespace is engine configuration and is never accepted from a
generation request. When T3 is enabled, Swift obtains or creates one 256-bit
per-install key using public Security/Keychain APIs with a `ThisDeviceOnly`
accessibility class. It creates a private cache directory under the host's
Application Support container, rejects symlink escapes, applies iOS Data
Protection and verifies it, verifies backup exclusion, and passes borrowed
path/key ranges to ABI v3. A protected, non-backed-up installation marker is
cryptographically tied to the key. If a `ThisDeviceOnly` Keychain item survives
an uninstall while the app-owned marker does not, setup rotates the stale key
under a cross-process installation lock before any cache record is used.
Rust copies the key synchronously into a zeroing, non-cloneable wrapper and
moves it once to the worker thread. Setup fails closed if any Keychain, path,
quota, namespace, or protection precondition fails. ABI v1/v2 structures remain
unchanged and cannot enable persistence.

The public quota is capped at 4 GiB. Until chunked record I/O is promoted,
individual records have a separate safety cap of 32 MiB on iOS and 128 MiB on
macOS; this cap is not additional cache capacity. This host boundary does not
by itself claim persistent-cache performance or promotion—the runtime
correctness, corruption, cancellation, restore-cost, and soak gates still
apply.

Hosts should forward memory-pressure lifecycle changes through
`handleMemoryPressure(_:)`. `.warning` keeps active requests running, reduces
new admission to half the configured concurrency, and permits only surviving
zero-copy T1 hits. `.critical` limits new admission to one request and disables
all new cache attachment. Excess requests remain queued. After the operating
system reports recovery, the host must send `.normal` explicitly; this restores
the configured admission and cache policy for later requests without
resurrecting evicted entries.

On macOS, with the two Rust iOS targets installed, build the three-slice static
package with:

```sh
scripts/build_apple_xcframework.sh /absolute/output/RvllmApple.xcframework
```

The result contains macOS arm64, iOS arm64 device, and iOS arm64 simulator
slices plus the canonical C header. The packaging command fails rather than
replacing an existing output, and scans every slice for private Apple API
references before promotion. Link that XCFramework into the host application;
the source-only Swift package intentionally does not hide a missing native
library behind placeholder symbols.

Opaque engine and request handles have unique ownership. Swift manages these
automatically. C callers must destroy each successful handle exactly once, may
cancel a request concurrently with its blocking receive, and must wait for any
receive to return before destroying that request.
