import CRvllmApple
import Foundation

private let requestReceiveQueue: OperationQueue = {
    let queue = OperationQueue()
    queue.name = "org.rvllm.apple-inference.receive"
    queue.qualityOfService = .userInitiated
    queue.maxConcurrentOperationCount = 16
    return queue
}()

public enum AppleBackendPolicy: UInt32, Sendable {
    case automatic = 0
    case metalOnly = 1
    case coreMLPreferred = 2
    case coreMLOnly = 3
}

public enum AppleCachePolicy: UInt32, Sendable {
    case disabled = 0
    case memoryOnly = 1
    case persistentEncrypted = 2
}

public enum AppleWorkloadProfile: UInt32, Sendable {
    case interactive = 0
    case balanced = 1
    case throughput = 2
}

public enum AppleMemoryProfile: UInt32, Sendable {
    case conservative = 0
    case balanced = 1
    case maximumPerformance = 2
}

public enum MemoryPressureLevel: UInt32, Sendable {
    case normal = 0
    case warning = 1
    case critical = 2
}

public struct AppleEngineConfig: Sendable {
    public var backendPolicy: AppleBackendPolicy = .automatic
    public var cachePolicy: AppleCachePolicy = .memoryOnly
    public var workloadProfile: AppleWorkloadProfile = .balanced
    public var memoryProfile: AppleMemoryProfile = .balanced
    public var maximumConcurrency: UInt32 = 0
    public var ingressQueueCapacity: UInt32 = 64
    public var eventQueueCapacity: UInt32 = 32
    public var hotCacheBytes: UInt64 = 256 * 1024 * 1024
    public var warmCacheBytes: UInt64? = nil
    public var persistentCacheBytes: UInt64 = 0
    public var persistentCacheConsent: Bool = false
    /// Engine-level tenant scope. Never derived from generation request input.
    public var persistentCacheNamespace: String? = nil
    /// Stable application/host identity used to scope Keychain and disk state.
    /// Required explicitly when encrypted persistence is enabled.
    public var persistentCacheHostIdentifier: String? = nil

    public init() {}

    func persistentCacheIsEnabled() throws -> Bool {
        if cachePolicy == .persistentEncrypted {
            guard persistentCacheConsent else {
                throw PersistentCacheHostBoundary.boundaryError(
                    "Encrypted persistent cache requires explicit consent"
                )
            }
            guard persistentCacheBytes > 0,
                  persistentCacheBytes <= PersistentCacheHostBoundary.maximumQuotaBytes
            else {
                throw PersistentCacheHostBoundary.boundaryError(
                    "Persistent-cache quota must be between 1 byte and 4 GiB"
                )
            }
            return true
        }
        guard !persistentCacheConsent, persistentCacheBytes == 0 else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache consent and quota require the encrypted persistent cache policy"
            )
        }
        return false
    }

    fileprivate func rawValueV2() throws -> RvllmAppleEngineConfigV2 {
        var raw = RvllmAppleEngineConfigV2()
        guard rvllm_apple_engine_config_v2_init(&raw) == RVLLM_APPLE_OK else {
            throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Could not initialize the C engine configuration")
        }
        raw.backend_policy = backendPolicy.rawValue
        raw.cache_policy = cachePolicy.rawValue
        raw.workload_profile = workloadProfile.rawValue
        raw.memory_profile = memoryProfile.rawValue
        if maximumConcurrency != 0 {
            raw.maximum_concurrency = maximumConcurrency
        }
        raw.ingress_queue_capacity = ingressQueueCapacity
        raw.event_queue_capacity = eventQueueCapacity
        raw.hot_cache_bytes = hotCacheBytes
        if let warmCacheBytes {
            raw.warm_cache_bytes = warmCacheBytes
        }
        raw.persistent_cache_bytes = persistentCacheBytes
        raw.persistent_cache_consent = persistentCacheConsent ? 1 : 0
        return raw
    }

    fileprivate func rawValueV3() throws -> RvllmAppleEngineConfigV3 {
        var raw = RvllmAppleEngineConfigV3()
        guard rvllm_apple_engine_config_v3_init(&raw) == RVLLM_APPLE_OK else {
            throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Could not initialize the C engine configuration")
        }
        raw.backend_policy = backendPolicy.rawValue
        raw.cache_policy = cachePolicy.rawValue
        raw.workload_profile = workloadProfile.rawValue
        raw.memory_profile = memoryProfile.rawValue
        if maximumConcurrency != 0 {
            raw.maximum_concurrency = maximumConcurrency
        }
        raw.ingress_queue_capacity = ingressQueueCapacity
        raw.event_queue_capacity = eventQueueCapacity
        raw.hot_cache_bytes = hotCacheBytes
        if let warmCacheBytes {
            raw.warm_cache_bytes = warmCacheBytes
        }
        raw.persistent_cache_bytes = persistentCacheBytes
        raw.persistent_cache_consent = persistentCacheConsent ? 1 : 0
        return raw
    }

    func resolvedPersistentCacheNamespace() throws -> [UInt8] {
        guard let namespace = persistentCacheNamespace else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Encrypted persistent cache requires an explicit engine-level tenant namespace"
            )
        }
        let bytes = Array(namespace.utf8)
        guard !bytes.isEmpty, bytes.count <= 128,
              !namespace.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache namespace must contain 1–128 UTF-8 bytes without control characters"
            )
        }
        return bytes
    }

    func resolvedPersistentCacheHostIdentifier() throws -> String {
        let allowed = CharacterSet.alphanumerics.union(
            CharacterSet(charactersIn: ".-_")
        )
        guard let identifier = persistentCacheHostIdentifier,
              !identifier.isEmpty,
              identifier.utf8.count <= 128,
              identifier.unicodeScalars.allSatisfy({ allowed.contains($0) })
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent cache requires a 1–128 character host or bundle identifier"
            )
        }
        return identifier
    }
}

public struct GenerateRequest: Sendable {
    public var promptTokens: [UInt32]
    public var maxOutputTokens: UInt32
    public var priority: UInt8
    public var cachePolicy: AppleCachePolicy

    public init(
        promptTokens: [UInt32],
        maxOutputTokens: UInt32,
        priority: UInt8 = 128,
        cachePolicy: AppleCachePolicy = .memoryOnly
    ) {
        self.promptTokens = promptTokens
        self.maxOutputTokens = maxOutputTokens
        self.priority = priority
        self.cachePolicy = cachePolicy
    }
}

public enum FinishReason: UInt32, Sendable {
    case endOfSequence = 1
    case length = 2
    case stopToken = 3
}

public enum SelectedBackend: UInt32, Sendable {
    case metal = 1
    case coreML = 2
}

public enum CacheTier: UInt32, Sendable {
    case none = 0
    case active = 1
    case hot = 2
    case warm = 3
    case persistent = 4
}

public enum ThermalState: UInt32, Sendable {
    case unknown = 0
    case nominal = 1
    case fair = 2
    case serious = 3
    case critical = 4
}

public struct BackendReport: Sendable {
    public let selectedBackend: SelectedBackend
    public let cacheTier: CacheTier
    public let matchedCacheTokens: UInt32
    public let savedPrefillTokens: UInt32
    public let queueTimeNanoseconds: UInt64
    public let batchSize: UInt32
    public let paddingTokens: UInt32
    public let prefillTimeNanoseconds: UInt64
    public let decodeTimeNanoseconds: UInt64
    public let residentMemoryBytes: UInt64
    public let thermalState: ThermalState
    public let hadFallback: Bool

    fileprivate init(_ raw: RvllmAppleBackendReport) throws {
        guard
            let selectedBackend = SelectedBackend(rawValue: raw.selected_backend),
            let cacheTier = CacheTier(rawValue: raw.cache_tier),
            let thermalState = ThermalState(rawValue: raw.thermal_state)
        else {
            throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Runtime returned an invalid backend report")
        }
        self.selectedBackend = selectedBackend
        self.cacheTier = cacheTier
        self.matchedCacheTokens = raw.matched_cache_tokens
        self.savedPrefillTokens = raw.saved_prefill_tokens
        self.queueTimeNanoseconds = raw.queue_time_ns
        self.batchSize = raw.batch_size
        self.paddingTokens = raw.padding_tokens
        self.prefillTimeNanoseconds = raw.prefill_time_ns
        self.decodeTimeNanoseconds = raw.decode_time_ns
        self.residentMemoryBytes = raw.resident_memory_bytes
        self.thermalState = thermalState
        self.hadFallback = raw.had_fallback != 0
    }
}

public enum TokenEvent: Sendable {
    case token(requestID: UInt64, index: UInt32, tokenID: UInt32, text: String?)
    case finished(requestID: UInt64, reason: FinishReason, report: BackendReport)
}

public struct AppleInferenceError: Error, LocalizedError, Sendable {
    public let status: Int32
    public let message: String

    public var errorDescription: String? { message }

    init(status: Int32, message: String) {
        self.status = status
        self.message = message
    }

    fileprivate init(status: Int32, raw: RvllmAppleError) {
        var raw = raw
        let message = withUnsafePointer(to: &raw.message) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: Int(RVLLM_APPLE_ERROR_MESSAGE_CAPACITY)) {
                String(cString: $0)
            }
        }
        self.init(status: status, message: message.isEmpty ? "Apple inference failed with status \(status)" : message)
    }
}

private final class RequestOwner: @unchecked Sendable {
    private let lock = NSLock()
    private var pointer: OpaquePointer?

    init(pointer: OpaquePointer) {
        self.pointer = pointer
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        guard let pointer else { return }
        _ = rvllm_apple_request_cancel(pointer)
    }

    func destroy() {
        lock.lock()
        let owned = pointer
        pointer = nil
        lock.unlock()
        if let owned {
            rvllm_apple_request_destroy(owned)
        }
    }

    func pointerForReceive() throws -> OpaquePointer {
        lock.lock()
        defer { lock.unlock() }
        guard let pointer else {
            throw AppleInferenceError(status: RVLLM_APPLE_CANCELLED, message: "Request was already released")
        }
        return pointer
    }
}

private final class EngineOwner: @unchecked Sendable {
    let pointer: OpaquePointer

    init(modelURL: URL, resourcesURL: URL?, config: AppleEngineConfig) throws {
        guard modelURL.isFileURL else {
            throw AppleInferenceError(status: RVLLM_APPLE_INVALID_ARGUMENT, message: "Model URL must be a local file URL")
        }
        if let resourcesURL, !resourcesURL.isFileURL {
            throw AppleInferenceError(status: RVLLM_APPLE_INVALID_ARGUMENT, message: "Resources URL must be a local file URL")
        }
        if let hostMaterial = try PersistentCacheHostBoundary.prepare(config: config) {
            pointer = try Self.createV3(
                modelURL: modelURL,
                resourcesURL: resourcesURL,
                config: config,
                hostMaterial: hostMaterial
            )
        } else {
            pointer = try Self.createV2(
                modelURL: modelURL,
                resourcesURL: resourcesURL,
                config: config
            )
        }
    }

    private static func createV2(
        modelURL: URL,
        resourcesURL: URL?,
        config: AppleEngineConfig
    ) throws -> OpaquePointer {
        var rawConfig = try config.rawValueV2()
        var rawError = RvllmAppleError()
        var created: OpaquePointer?
        let modelPath = Array(modelURL.path.utf8)
        let resourcePath = resourcesURL.map { Array($0.path.utf8) }
        let status = modelPath.withUnsafeBufferPointer { modelBytes in
            rawConfig.model_package_path = modelBytes.baseAddress
            rawConfig.model_package_path_length = modelBytes.count
            if let resourcePath {
                return resourcePath.withUnsafeBufferPointer { resourceBytes in
                    rawConfig.resource_bundle_path = resourceBytes.baseAddress
                    rawConfig.resource_bundle_path_length = resourceBytes.count
                    return rvllm_apple_engine_create_v2(&rawConfig, &created, &rawError)
                }
            }
            rawConfig.resource_bundle_path = nil
            rawConfig.resource_bundle_path_length = 0
            return rvllm_apple_engine_create_v2(&rawConfig, &created, &rawError)
        }
        guard status == RVLLM_APPLE_OK, let created else {
            throw AppleInferenceError(status: status, raw: rawError)
        }
        return created
    }

    private static func createV3(
        modelURL: URL,
        resourcesURL: URL?,
        config: AppleEngineConfig,
        hostMaterial: PersistentCacheHostMaterial
    ) throws -> OpaquePointer {
        defer { hostMaterial.key.clear() }

        var rawConfig = try config.rawValueV3()
        var rawError = RvllmAppleError()
        var created: OpaquePointer?
        let modelPath = Array(modelURL.path.utf8)
        let resourcePath = resourcesURL.map { Array($0.path.utf8) }
        let cacheRootPath = Array(hostMaterial.root.path.utf8)
        let status = modelPath.withUnsafeBufferPointer { modelBytes in
            cacheRootPath.withUnsafeBufferPointer { rootBytes in
                hostMaterial.namespace.withUnsafeBufferPointer { namespaceBytes in
                    hostMaterial.key.bytes.withUnsafeBytes { keyBytes in
                        rawConfig.model_package_path = modelBytes.baseAddress
                        rawConfig.model_package_path_length = modelBytes.count
                        rawConfig.persistent_cache_root = rootBytes.baseAddress
                        rawConfig.persistent_cache_root_length = rootBytes.count
                        rawConfig.persistent_cache_key = keyBytes
                            .bindMemory(to: UInt8.self)
                            .baseAddress
                        rawConfig.persistent_cache_key_length = keyBytes.count
                        rawConfig.cache_namespace = namespaceBytes.baseAddress
                        rawConfig.cache_namespace_length = namespaceBytes.count
                        if let resourcePath {
                            return resourcePath.withUnsafeBufferPointer { resourceBytes in
                                rawConfig.resource_bundle_path = resourceBytes.baseAddress
                                rawConfig.resource_bundle_path_length = resourceBytes.count
                                return rvllm_apple_engine_create_v3(
                                    &rawConfig,
                                    &created,
                                    &rawError
                                )
                            }
                        }
                        rawConfig.resource_bundle_path = nil
                        rawConfig.resource_bundle_path_length = 0
                        return rvllm_apple_engine_create_v3(
                            &rawConfig,
                            &created,
                            &rawError
                        )
                    }
                }
            }
        }
        guard status == RVLLM_APPLE_OK, let created else {
            throw AppleInferenceError(status: status, raw: rawError)
        }
        return created
    }

    deinit {
        rvllm_apple_engine_destroy(pointer)
    }

    func submit(_ request: GenerateRequest) throws -> RequestOwner {
        var rawRequest = RvllmAppleGenerateRequest()
        guard rvllm_apple_generate_request_init(&rawRequest) == RVLLM_APPLE_OK else {
            throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Could not initialize the C generation request")
        }
        rawRequest.max_output_tokens = request.maxOutputTokens
        rawRequest.priority = request.priority
        rawRequest.cache_policy = request.cachePolicy.rawValue
        var rawError = RvllmAppleError()
        var submitted: OpaquePointer?
        let status = request.promptTokens.withUnsafeBufferPointer { tokens in
            rawRequest.prompt_tokens = tokens.baseAddress
            rawRequest.prompt_token_count = tokens.count
            return rvllm_apple_engine_submit(pointer, &rawRequest, &submitted, &rawError)
        }
        guard status == RVLLM_APPLE_OK, let submitted else {
            throw AppleInferenceError(status: status, raw: rawError)
        }
        return RequestOwner(pointer: submitted)
    }

    func handleMemoryPressure(_ level: MemoryPressureLevel) throws {
        var rawError = RvllmAppleError()
        let status = rvllm_apple_engine_handle_memory_pressure(pointer, level.rawValue, &rawError)
        guard status == RVLLM_APPLE_OK else {
            throw AppleInferenceError(status: status, raw: rawError)
        }
    }
}

/// Actor-isolated embedded inference engine.
///
/// The actor serializes engine lifecycle changes. Token reads run on a detached
/// task because the C receive operation intentionally blocks without occupying
/// the actor. Ending or cancelling stream consumption propagates cancellation
/// to the request before its opaque handle is released.
public actor AppleInferenceEngine {
    private let engine: EngineOwner

    public init(
        modelURL: URL,
        resourcesURL: URL? = nil,
        config: AppleEngineConfig = AppleEngineConfig()
    ) throws {
        engine = try EngineOwner(modelURL: modelURL, resourcesURL: resourcesURL, config: config)
    }

    public func generate(_ request: GenerateRequest) -> AsyncThrowingStream<TokenEvent, Error> {
        do {
            let owner = try submit(request)
            return AsyncThrowingStream { continuation in
                continuation.onTermination = { @Sendable _ in
                    owner.cancel()
                }
                requestReceiveQueue.addOperation {
                    defer { owner.destroy() }
                    do {
                        let pointer = try owner.pointerForReceive()
                        while let event = try Self.receive(pointer) {
                            continuation.yield(event)
                            if case .finished = event {
                                continuation.finish()
                                return
                            }
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }
    }

    public func handleMemoryPressure(_ level: MemoryPressureLevel) throws {
        try engine.handleMemoryPressure(level)
    }

    private func submit(_ request: GenerateRequest) throws -> RequestOwner {
        try engine.submit(request)
    }

    private nonisolated static func receive(_ request: OpaquePointer) throws -> TokenEvent? {
        var capacity = 256
        while true {
            var event = RvllmAppleTokenEvent()
            var rawError = RvllmAppleError()
            var text = [CChar](repeating: 0, count: capacity)
            let status = text.withUnsafeMutableBufferPointer { buffer in
                rvllm_apple_request_recv(request, &event, buffer.baseAddress, buffer.count, &rawError)
            }
            if status == RVLLM_APPLE_BUFFER_TOO_SMALL {
                capacity = max(capacity * 2, event.text_length)
                continue
            }
            if status == RVLLM_APPLE_END_OF_STREAM {
                return nil
            }
            guard status == RVLLM_APPLE_OK else {
                throw AppleInferenceError(status: status, raw: rawError)
            }
            switch event.kind {
            case RVLLM_APPLE_EVENT_TOKEN:
                let tokenText: String?
                if event.text_length == 0 {
                    tokenText = nil
                } else {
                    let bytes = text.prefix(event.text_length).map { UInt8(bitPattern: $0) }
                    guard let decoded = String(bytes: bytes, encoding: .utf8) else {
                        throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Runtime returned non-UTF-8 token text")
                    }
                    tokenText = decoded
                }
                return .token(requestID: event.request_id, index: event.index, tokenID: event.token_id, text: tokenText)
            case RVLLM_APPLE_EVENT_FINISHED:
                guard let reason = FinishReason(rawValue: event.finish_reason) else {
                    throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Runtime returned an invalid finish reason")
                }
                return .finished(requestID: event.request_id, reason: reason, report: try BackendReport(event.report))
            default:
                throw AppleInferenceError(status: RVLLM_APPLE_INTERNAL_ERROR, message: "Runtime returned an unknown token event")
            }
        }
    }
}
