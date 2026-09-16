import CRvllmApple
import CPersistentCacheHost
import CryptoKit
import Darwin
import Foundation
import Security

struct PersistentCacheKeychainItem: Equatable {
    var key: [UInt8]
    var accessibility: String
}

struct PersistentCacheKeychainWrite: Equatable {
    var service: String
    var account: String
    var key: [UInt8]
    var accessibility: String
    var synchronizable: Bool
}

protocol PersistentCacheKeychainClient {
    func load(service: String, account: String) throws -> PersistentCacheKeychainItem?
    func add(_ item: PersistentCacheKeychainWrite) -> OSStatus
    func replace(_ item: PersistentCacheKeychainWrite) -> OSStatus
    func randomBytes(count: Int) throws -> [UInt8]
}

final class PersistentCacheRootHandle {
    let url: URL
    fileprivate let descriptor: Int32

    init(url: URL, descriptor: Int32 = -1) {
        self.url = url
        self.descriptor = descriptor
    }

    deinit {
        if descriptor >= 0 {
            _ = close(descriptor)
        }
    }
}

protocol PersistentCacheFileClient {
    func prepareProtectedCacheRoot(hostIdentifier: String) throws -> PersistentCacheRootHandle
    func withInstallationLock<T>(
        root: PersistentCacheRootHandle,
        _ body: () throws -> T
    ) throws -> T
    func loadInstallationMarker(root: PersistentCacheRootHandle) throws -> Data?
    func writeInstallationMarker(
        _ marker: Data,
        root: PersistentCacheRootHandle
    ) throws
}

struct SystemPersistentCacheKeychainClient: PersistentCacheKeychainClient {
    func load(service: String, account: String) throws -> PersistentCacheKeychainItem? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
            kSecAttrSynchronizable: kCFBooleanFalse as Any,
            kSecReturnData: kCFBooleanTrue as Any,
            kSecReturnAttributes: kCFBooleanTrue as Any,
            kSecMatchLimit: kSecMatchLimitOne,
        ]
        var rawItem: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &rawItem)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess,
              let attributes = rawItem as? NSDictionary,
              var keyData = attributes[kSecValueData] as? Data,
              let accessibility = attributes[kSecAttrAccessible] as? String
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not read the persistent-cache Keychain item (\(status))"
            )
        }
        defer { PersistentCacheHostBoundary.zero(&keyData) }
        return PersistentCacheKeychainItem(
            key: Array(keyData),
            accessibility: accessibility
        )
    }

    func add(_ item: PersistentCacheKeychainWrite) -> OSStatus {
        SecItemAdd(writeAttributes(item) as CFDictionary, nil)
    }

    func replace(_ item: PersistentCacheKeychainWrite) -> OSStatus {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: item.service,
            kSecAttrAccount: item.account,
            kSecAttrSynchronizable: kCFBooleanFalse as Any,
        ]
        let update: [CFString: Any] = [
            kSecValueData: Data(item.key),
            kSecAttrAccessible: item.accessibility,
        ]
        return SecItemUpdate(query as CFDictionary, update as CFDictionary)
    }

    func randomBytes(count: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: count)
        let status = bytes.withUnsafeMutableBytes { rawBytes -> OSStatus in
            guard let baseAddress = rawBytes.baseAddress else {
                return errSecAllocate
            }
            return SecRandomCopyBytes(kSecRandomDefault, count, baseAddress)
        }
        guard status == errSecSuccess else {
            PersistentCacheHostBoundary.zero(&bytes)
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not generate the persistent-cache installation key"
            )
        }
        return bytes
    }

    private func writeAttributes(_ item: PersistentCacheKeychainWrite) -> [CFString: Any] {
        [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: item.service,
            kSecAttrAccount: item.account,
            kSecAttrSynchronizable: item.synchronizable ? kCFBooleanTrue as Any : kCFBooleanFalse as Any,
            kSecValueData: Data(item.key),
            kSecAttrAccessible: item.accessibility,
        ]
    }
}

struct SystemPersistentCacheFileClient: PersistentCacheFileClient {
    private let manager = FileManager.default
    private let applicationSupportOverride: URL?
    private let markerName = ".installation-key-marker-v1"
    private let lockName = ".installation-key.lock"
    private let beforeMarkerPublish: (() -> Void)?

    init(
        applicationSupportOverride: URL? = nil,
        beforeMarkerPublish: (() -> Void)? = nil
    ) {
        self.applicationSupportOverride = applicationSupportOverride
        self.beforeMarkerPublish = beforeMarkerPublish
    }

    func prepareProtectedCacheRoot(
        hostIdentifier: String
    ) throws -> PersistentCacheRootHandle {
        do {
            return try prepareProtectedCacheRootImpl(hostIdentifier: hostIdentifier)
        } catch let error as POSIXCallError {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not securely prepare the persistent-cache root (\(error.code))"
            )
        }
    }

    private func prepareProtectedCacheRootImpl(
        hostIdentifier: String
    ) throws -> PersistentCacheRootHandle {
        let requestedApplicationSupport = try (
            applicationSupportOverride ?? manager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        ).standardizedFileURL
        // Apple temporary and app-container paths may begin with the system
        // `/var` compatibility symlink. Resolve the existing trusted base once,
        // then pin every resulting component without following further links.
        let applicationSupport = try canonicalExistingDirectory(
            requestedApplicationSupport
        )
        let namespaceName = "\(hostIdentifier).rvllm"
        let rootName = PersistentCacheHostBoundary.cacheDirectory
        let namespace = applicationSupport.appendingPathComponent(
            namespaceName,
            isDirectory: true
        )
        let root = namespace.appendingPathComponent(
            PersistentCacheHostBoundary.cacheDirectory,
            isDirectory: true
        )
        guard root.path.utf8.count <= 4096 else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache directory path is too long"
            )
        }

        let applicationSupportDirectory = try openAbsoluteDirectory(applicationSupport)
        let namespaceDirectory = try openOrCreatePrivateDirectory(
            parent: applicationSupportDirectory.descriptor,
            name: namespaceName
        )
        try synchronize(
            applicationSupportDirectory.descriptor,
            message: "Application Support directory"
        )
        let rootDescriptor = try openOrCreatePrivateDirectory(
            parent: namespaceDirectory.descriptor,
            name: rootName
        ).release()
        try synchronize(
            namespaceDirectory.descriptor,
            message: "host namespace directory"
        )
        let handle = PersistentCacheRootHandle(url: root, descriptor: rootDescriptor)
        try validatePrivateDirectory(handle.descriptor)

        #if os(iOS)
        try setAndVerifyDataProtection(handle.descriptor)
        #endif

        try setAndVerifyBackupExclusion(for: handle.descriptor)
        try synchronize(handle.descriptor, message: "protected cache root")
        try validatePrivateDirectory(handle.descriptor)
        return handle
    }

    func withInstallationLock<T>(
        root: PersistentCacheRootHandle,
        _ body: () throws -> T
    ) throws -> T {
        try validatePrivateDirectory(root.descriptor)
        let descriptor: Int32
        do {
            descriptor = try openAt(
                directory: root.descriptor,
                name: lockName,
                flags: O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC,
                mode: S_IRUSR | S_IWUSR
            )
        } catch let error as POSIXCallError {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not open the persistent-cache installation lock (\(error.code))"
            )
        }
        defer { close(descriptor) }
        var fileInfo = stat()
        guard fstat(descriptor, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFREG,
              fileInfo.st_nlink == 1,
              fileInfo.st_uid == geteuid(),
              (fileInfo.st_mode & 0o777) == (S_IRUSR | S_IWUSR)
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache installation lock is not a private regular file"
            )
        }
        try protectAuxiliaryFile(descriptor)
        try synchronize(descriptor, message: "installation lock")
        try synchronize(root.descriptor, message: "installation lock directory")
        guard flock(descriptor, LOCK_EX) == 0 else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not acquire the persistent-cache installation lock"
            )
        }
        defer { _ = flock(descriptor, LOCK_UN) }
        try validateNamedDescriptor(
            directory: root.descriptor,
            name: lockName,
            descriptor: descriptor
        )
        return try body()
    }

    func loadInstallationMarker(root: PersistentCacheRootHandle) throws -> Data? {
        try validatePrivateDirectory(root.descriptor)
        let descriptor: Int32
        do {
            descriptor = try openAt(
                directory: root.descriptor,
                name: markerName,
                flags: O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK
            )
        } catch let error as POSIXCallError where error.code == ENOENT {
            return nil
        } catch {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not open the persistent-cache installation marker"
            )
        }
        defer { close(descriptor) }
        var fileInfo = stat()
        guard fstat(descriptor, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFREG,
              fileInfo.st_nlink == 1,
              fileInfo.st_uid == geteuid(),
              fileInfo.st_size == PersistentCacheHostBoundary.markerBytes,
              (fileInfo.st_mode & 0o777) == (S_IRUSR | S_IWUSR)
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache installation marker is invalid"
            )
        }
        var marker = [UInt8](repeating: 0, count: PersistentCacheHostBoundary.markerBytes)
        try readExactly(marker: &marker, descriptor: descriptor)
        return Data(marker)
    }

    func writeInstallationMarker(
        _ marker: Data,
        root: PersistentCacheRootHandle
    ) throws {
        guard marker.count == PersistentCacheHostBoundary.markerBytes else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache installation marker has an invalid length"
            )
        }
        try validatePrivateDirectory(root.descriptor)
        let temporaryName = ".installation-key-marker-\(UUID().uuidString).tmp"
        let descriptor: Int32
        do {
            descriptor = try openAt(
                directory: root.descriptor,
                name: temporaryName,
                flags: O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC,
                mode: S_IRUSR | S_IWUSR
            )
        } catch let error as POSIXCallError {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not create the persistent-cache installation marker (\(error.code))"
            )
        }
        var shouldRemoveTemporary = true
        defer {
            close(descriptor)
            if shouldRemoveTemporary {
                try? unlinkAt(directory: root.descriptor, name: temporaryName)
            }
        }
        var fileInfo = stat()
        guard fstat(descriptor, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFREG,
              fileInfo.st_nlink == 1,
              fileInfo.st_uid == geteuid(),
              (fileInfo.st_mode & 0o777) == (S_IRUSR | S_IWUSR)
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache installation marker staging file is invalid"
            )
        }
        try writeExactly(marker: marker, descriptor: descriptor)
        try protectAuxiliaryFile(descriptor)
        try synchronize(descriptor, message: "installation marker")
        beforeMarkerPublish?()
        try renameAt(
            directory: root.descriptor,
            source: temporaryName,
            destination: markerName
        )
        shouldRemoveTemporary = false
        try synchronize(root.descriptor, message: "installation marker directory")
        guard try loadInstallationMarker(root: root) == marker else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not verify persistent-cache installation marker"
            )
        }
    }

    private func readExactly(marker: inout [UInt8], descriptor: Int32) throws {
        var offset = 0
        let total = marker.count
        while offset < total {
            let result = marker.withUnsafeMutableBytes { bytes in
                Darwin.read(
                    descriptor,
                    bytes.baseAddress!.advanced(by: offset),
                    total - offset
                )
            }
            if result < 0, errno == EINTR {
                continue
            }
            guard result > 0 else {
                throw PersistentCacheHostBoundary.boundaryError(
                    "Persistent-cache installation marker is truncated"
                )
            }
            offset += result
        }
    }

    private func writeExactly(marker: Data, descriptor: Int32) throws {
        var offset = 0
        try marker.withUnsafeBytes { bytes in
            while offset < bytes.count {
                let result = Darwin.write(
                    descriptor,
                    bytes.baseAddress!.advanced(by: offset),
                    bytes.count - offset
                )
                if result < 0, errno == EINTR {
                    continue
                }
                guard result > 0 else {
                    throw PersistentCacheHostBoundary.boundaryError(
                        "Could not write the persistent-cache installation marker"
                    )
                }
                offset += result
            }
        }
    }

    private func openAbsoluteDirectory(_ url: URL) throws -> OwnedDescriptor {
        guard url.isFileURL, url.path.hasPrefix("/") else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Application Support must be an absolute local directory"
            )
        }
        var current = OwnedDescriptor(
            try openPath("/", flags: O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        )
        for component in url.pathComponents where component != "/" {
            let next = OwnedDescriptor(
                try openAt(
                    directory: current.descriptor,
                    name: component,
                    flags: O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
                )
            )
            current = next
        }
        try validateDirectory(current.descriptor)
        return current
    }

    private func canonicalExistingDirectory(_ url: URL) throws -> URL {
        var path = [CChar](repeating: 0, count: Int(PATH_MAX))
        let resolved = url.path.withCString { source in
            path.withUnsafeMutableBufferPointer {
                realpath(source, $0.baseAddress)
            }
        }
        guard resolved != nil else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not resolve the Application Support directory"
            )
        }
        let terminator = path.firstIndex(of: 0) ?? path.endIndex
        return URL(
            fileURLWithPath: String(
                decoding: path[..<terminator].map(UInt8.init(bitPattern:)),
                as: UTF8.self
            ),
            isDirectory: true
        )
    }

    private func openOrCreatePrivateDirectory(
        parent: Int32,
        name: String
    ) throws -> OwnedDescriptor {
        do {
            try makeDirectoryAt(parent: parent, name: name, mode: 0o700)
        } catch let error as POSIXCallError where error.code == EEXIST {
            // Open and validate the existing entry below.
        }
        let directory = OwnedDescriptor(
            try openAt(
                directory: parent,
                name: name,
                flags: O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
            )
        )
        try validatePrivateDirectory(directory.descriptor)
        return directory
    }

    private func validateDirectory(_ descriptor: Int32) throws {
        var fileInfo = stat()
        guard fstat(descriptor, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFDIR
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache path is not a directory"
            )
        }
    }

    private func validatePrivateDirectory(_ descriptor: Int32) throws {
        var fileInfo = stat()
        guard fstat(descriptor, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFDIR,
              fileInfo.st_uid == geteuid(),
              (fileInfo.st_mode & 0o777) == 0o700
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache path is not a private owned directory"
            )
        }
    }

    private func protectAuxiliaryFile(_ descriptor: Int32) throws {
        #if os(iOS)
        try setAndVerifyDataProtection(descriptor)
        #endif
        try setAndVerifyBackupExclusion(for: descriptor)
    }

    #if os(iOS)
    private func setAndVerifyDataProtection(_ descriptor: Int32) throws {
        // Darwin protection class C is the descriptor-level equivalent of
        // completeUntilFirstUserAuthentication.
        let completeUntilFirstAuthentication: Int32 = 3
        guard rvllm_persistent_cache_set_protection_class(
            descriptor,
            completeUntilFirstAuthentication
        ) == 0,
            rvllm_persistent_cache_get_protection_class(descriptor)
                == completeUntilFirstAuthentication
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not set or verify persistent-cache Data Protection"
            )
        }
    }
    #endif

    private func setAndVerifyBackupExclusion(for descriptor: Int32) throws {
        let url = try urlForDescriptor(descriptor)
        var mutableURL = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try mutableURL.setResourceValues(values)
        let verified = try mutableURL.resourceValues(forKeys: [.isExcludedFromBackupKey])
        guard verified.isExcludedFromBackup == true else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not verify persistent-cache backup exclusion"
            )
        }
        try validateDescriptor(descriptor, stillNames: mutableURL)
    }

    private func validateDescriptor(_ descriptor: Int32, stillNames url: URL) throws {
        var descriptorInfo = stat()
        var pathInfo = stat()
        guard fstat(descriptor, &descriptorInfo) == 0,
              lstat(url.path, &pathInfo) == 0,
              descriptorInfo.st_dev == pathInfo.st_dev,
              descriptorInfo.st_ino == pathInfo.st_ino
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache object changed while applying host protection"
            )
        }
    }

    private func validateNamedDescriptor(
        directory: Int32,
        name: String,
        descriptor: Int32
    ) throws {
        var descriptorInfo = stat()
        var namedInfo = stat()
        let namedStatus = name.withCString {
            fstatat(directory, $0, &namedInfo, AT_SYMLINK_NOFOLLOW)
        }
        guard fstat(descriptor, &descriptorInfo) == 0,
              namedStatus == 0,
              descriptorInfo.st_dev == namedInfo.st_dev,
              descriptorInfo.st_ino == namedInfo.st_ino,
              descriptorInfo.st_nlink == 1,
              (namedInfo.st_mode & S_IFMT) == S_IFREG
        else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Persistent-cache installation lock changed during acquisition"
            )
        }
    }

    private func urlForDescriptor(_ descriptor: Int32) throws -> URL {
        var path = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        let status = path.withUnsafeMutableBufferPointer {
            rvllm_persistent_cache_get_path(
                descriptor,
                $0.baseAddress,
                $0.count
            )
        }
        guard status == 0 else {
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not resolve the protected persistent-cache object"
            )
        }
        let terminator = path.firstIndex(of: 0) ?? path.endIndex
        return URL(
            fileURLWithPath: String(
                decoding: path[..<terminator].map(UInt8.init(bitPattern:)),
                as: UTF8.self
            )
        )
    }

    private func synchronize(_ descriptor: Int32, message: String) throws {
        while fsync(descriptor) != 0 {
            if errno == EINTR {
                continue
            }
            throw PersistentCacheHostBoundary.boundaryError(
                "Could not synchronize the persistent-cache \(message)"
            )
        }
    }

    private func openPath(_ path: String, flags: Int32) throws -> Int32 {
        while true {
            let descriptor = path.withCString { open($0, flags) }
            if descriptor >= 0 {
                return descriptor
            }
            if errno != EINTR {
                throw POSIXCallError(code: errno)
            }
        }
    }

    private func openAt(
        directory: Int32,
        name: String,
        flags: Int32
    ) throws -> Int32 {
        while true {
            let descriptor = name.withCString { openat(directory, $0, flags) }
            if descriptor >= 0 {
                return descriptor
            }
            if errno != EINTR {
                throw POSIXCallError(code: errno)
            }
        }
    }

    private func openAt(
        directory: Int32,
        name: String,
        flags: Int32,
        mode: mode_t
    ) throws -> Int32 {
        while true {
            let descriptor = name.withCString { openat(directory, $0, flags, mode) }
            if descriptor >= 0 {
                return descriptor
            }
            if errno != EINTR {
                throw POSIXCallError(code: errno)
            }
        }
    }

    private func makeDirectoryAt(
        parent: Int32,
        name: String,
        mode: mode_t
    ) throws {
        while true {
            let status = name.withCString { mkdirat(parent, $0, mode) }
            if status == 0 {
                return
            }
            if errno != EINTR {
                throw POSIXCallError(code: errno)
            }
        }
    }

    private func renameAt(
        directory: Int32,
        source: String,
        destination: String
    ) throws {
        while true {
            let status = source.withCString { sourcePointer in
                destination.withCString { destinationPointer in
                    renameat(
                        directory,
                        sourcePointer,
                        directory,
                        destinationPointer
                    )
                }
            }
            if status == 0 {
                return
            }
            if errno != EINTR {
                throw PersistentCacheHostBoundary.boundaryError(
                    "Could not publish the persistent-cache installation marker"
                )
            }
        }
    }

    private func unlinkAt(directory: Int32, name: String) throws {
        while true {
            let status = name.withCString { unlinkat(directory, $0, 0) }
            if status == 0 || errno == ENOENT {
                return
            }
            if errno != EINTR {
                throw POSIXCallError(code: errno)
            }
        }
    }
}

private final class OwnedDescriptor {
    private var owned: Int32

    init(_ descriptor: Int32) {
        owned = descriptor
    }

    var descriptor: Int32 {
        owned
    }

    func release() -> Int32 {
        let descriptor = owned
        owned = -1
        return descriptor
    }

    deinit {
        if owned >= 0 {
            _ = close(owned)
        }
    }
}

private struct POSIXCallError: Error {
    let code: Int32
}

final class PersistentCacheSensitiveBytes {
    private(set) var bytes: [UInt8]
    private let didClear: (([UInt8]) -> Void)?
    private let lifetimeAnchor: AnyObject?
    private var isCleared = false

    init(
        _ bytes: [UInt8],
        didClear: (([UInt8]) -> Void)? = nil,
        lifetimeAnchor: AnyObject? = nil
    ) {
        self.bytes = bytes
        self.didClear = didClear
        self.lifetimeAnchor = lifetimeAnchor
    }

    deinit {
        clear()
    }

    func clear() {
        guard !isCleared else { return }
        withExtendedLifetime(lifetimeAnchor) {
            PersistentCacheHostBoundary.zero(&bytes)
        }
        isCleared = true
        didClear?(bytes)
    }
}

struct PersistentCacheHostMaterial {
    var root: URL
    var namespace: [UInt8]
    var key: PersistentCacheSensitiveBytes
    fileprivate var rootHandle: PersistentCacheRootHandle
}

enum PersistentCacheHostBoundary {
    static let keyBytes = Int(RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES)
    static let maximumQuotaBytes = UInt64(RVLLM_APPLE_MAX_PERSISTENT_CACHE_BYTES)
    static let markerBytes = 32
    static let keychainAccount = "installation-key-v1"
    static let cacheDirectory = "PersistentPromptCache"
    static let accessibility = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String

    static func prepare(
        config: AppleEngineConfig,
        keychain: PersistentCacheKeychainClient = SystemPersistentCacheKeychainClient(),
        files: PersistentCacheFileClient = SystemPersistentCacheFileClient()
    ) throws -> PersistentCacheHostMaterial? {
        guard try config.persistentCacheIsEnabled() else {
            return nil
        }
        // Resolve caller-owned identity before any Security or filesystem I/O.
        let namespace = try config.resolvedPersistentCacheNamespace()
        let hostIdentifier = try config.resolvedPersistentCacheHostIdentifier()
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: hostIdentifier)
        let key = try files.withInstallationLock(root: root) {
            try loadOrCreateInstallationKey(
                hostIdentifier: hostIdentifier,
                root: root,
                keychain: keychain,
                files: files
            )
        }
        return PersistentCacheHostMaterial(
            root: root.url,
            namespace: namespace,
            key: PersistentCacheSensitiveBytes(key, lifetimeAnchor: root),
            rootHandle: root
        )
    }

    static func loadOrCreateInstallationKey(
        hostIdentifier: String,
        root: PersistentCacheRootHandle,
        keychain: PersistentCacheKeychainClient,
        files: PersistentCacheFileClient
    ) throws -> [UInt8] {
        let service = "\(hostIdentifier).rvllm.PersistentPromptCache"
        let marker = try files.loadInstallationMarker(root: root)
        if let existing = try keychain.load(service: service, account: keychainAccount) {
            let key = try validate(existing)
            if marker == installationMarker(hostIdentifier: hostIdentifier, key: key) {
                return key
            }

            // Keychain ThisDeviceOnly items can outlive an app uninstall while
            // Application Support does not. Missing/mismatched marker means the
            // old installation key must never be reused.
            var replacement = try keychain.randomBytes(count: keyBytes)
            defer { zero(&replacement) }
            let write = keychainWrite(
                service: service,
                key: replacement
            )
            let status = keychain.replace(write)
            guard status == errSecSuccess else {
                throw boundaryError(
                    "Could not rotate the stale persistent-cache installation key (\(status))"
                )
            }
            try files.writeInstallationMarker(
                installationMarker(hostIdentifier: hostIdentifier, key: replacement),
                root: root
            )
            return try loadCanonicalKey(
                hostIdentifier: hostIdentifier,
                service: service,
                root: root,
                keychain: keychain,
                files: files
            )
        }

        var generated = try keychain.randomBytes(count: keyBytes)
        defer { zero(&generated) }
        let status = keychain.add(keychainWrite(service: service, key: generated))
        if status == errSecSuccess {
            try files.writeInstallationMarker(
                installationMarker(hostIdentifier: hostIdentifier, key: generated),
                root: root
            )
        } else if status != errSecDuplicateItem {
            throw boundaryError(
                "Could not store the persistent-cache installation key (\(status))"
            )
        }
        // On duplicate-add, the installation lock ensures the winning process
        // has completed its marker update before this canonical read.
        return try loadCanonicalKey(
            hostIdentifier: hostIdentifier,
            service: service,
            root: root,
            keychain: keychain,
            files: files
        )
    }

    private static func loadCanonicalKey(
        hostIdentifier: String,
        service: String,
        root: PersistentCacheRootHandle,
        keychain: PersistentCacheKeychainClient,
        files: PersistentCacheFileClient
    ) throws -> [UInt8] {
        guard let item = try keychain.load(service: service, account: keychainAccount) else {
            throw boundaryError("Could not recover the persistent-cache installation key")
        }
        let key = try validate(item)
        guard try files.loadInstallationMarker(root: root)
                == installationMarker(hostIdentifier: hostIdentifier, key: key)
        else {
            throw boundaryError(
                "Persistent-cache installation key and protected marker do not match"
            )
        }
        return key
    }

    private static func validate(_ item: PersistentCacheKeychainItem) throws -> [UInt8] {
        guard item.key.count == keyBytes, item.accessibility == accessibility else {
            throw boundaryError(
                "The persistent-cache Keychain item has invalid key material or accessibility"
            )
        }
        return item.key
    }

    private static func keychainWrite(
        service: String,
        key: [UInt8]
    ) -> PersistentCacheKeychainWrite {
        PersistentCacheKeychainWrite(
            service: service,
            account: keychainAccount,
            key: key,
            accessibility: accessibility,
            synchronizable: false
        )
    }

    static func installationMarker(hostIdentifier: String, key: [UInt8]) -> Data {
        var message = Data("rvllm-persistent-cache-installation-v1".utf8)
        message.append(contentsOf: hostIdentifier.utf8)
        let authentication = HMAC<SHA256>.authenticationCode(
            for: message,
            using: SymmetricKey(data: key)
        )
        return Data(authentication)
    }

    static func zero(_ data: inout Data) {
        _ = data.withUnsafeMutableBytes { bytes in
            bytes.initializeMemory(as: UInt8.self, repeating: 0)
        }
    }

    static func zero(_ bytes: inout [UInt8]) {
        _ = bytes.withUnsafeMutableBytes {
            $0.initializeMemory(as: UInt8.self, repeating: 0)
        }
    }

    static func boundaryError(_ message: String) -> AppleInferenceError {
        AppleInferenceError(status: RVLLM_APPLE_INVALID_ARGUMENT, message: message)
    }
}
