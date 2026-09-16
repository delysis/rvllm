@testable import AppleInference
import Foundation
import Security
import XCTest

private final class MockKeychain: PersistentCacheKeychainClient {
    var loadCount = 0
    var addCount = 0
    var replaceCount = 0
    var randomCount = 0
    var stored: PersistentCacheKeychainItem?
    var addStatus: OSStatus = errSecSuccess
    var replaceStatus: OSStatus = errSecSuccess
    var random = [UInt8](repeating: 0xA5, count: PersistentCacheHostBoundary.keyBytes)
    var lastAdd: PersistentCacheKeychainWrite?
    var lastReplace: PersistentCacheKeychainWrite?
    var duplicateWinner: PersistentCacheKeychainItem?
    var onDuplicateWinner: (() -> Void)?

    func load(service: String, account: String) throws -> PersistentCacheKeychainItem? {
        loadCount += 1
        XCTAssertEqual(account, PersistentCacheHostBoundary.keychainAccount)
        return stored
    }

    func add(_ item: PersistentCacheKeychainWrite) -> OSStatus {
        addCount += 1
        lastAdd = item
        if addStatus == errSecSuccess {
            stored = PersistentCacheKeychainItem(
                key: item.key,
                accessibility: item.accessibility
            )
        } else if addStatus == errSecDuplicateItem {
            stored = duplicateWinner
            onDuplicateWinner?()
        }
        return addStatus
    }

    func replace(_ item: PersistentCacheKeychainWrite) -> OSStatus {
        replaceCount += 1
        lastReplace = item
        if replaceStatus == errSecSuccess {
            stored = PersistentCacheKeychainItem(
                key: item.key,
                accessibility: item.accessibility
            )
        }
        return replaceStatus
    }

    func randomBytes(count: Int) throws -> [UInt8] {
        randomCount += 1
        XCTAssertEqual(count, PersistentCacheHostBoundary.keyBytes)
        return random
    }
}

private final class MockFiles: PersistentCacheFileClient {
    var prepareCount = 0
    var lockCount = 0
    var loadMarkerCount = 0
    var writeMarkerCount = 0
    var root = URL(fileURLWithPath: "/tmp/rvllm-test-cache", isDirectory: true)
    var marker: Data?
    var onWrite: ((Data) -> Void)?

    func prepareProtectedCacheRoot(
        hostIdentifier: String
    ) throws -> PersistentCacheRootHandle {
        prepareCount += 1
        return PersistentCacheRootHandle(url: root)
    }

    func withInstallationLock<T>(
        root: PersistentCacheRootHandle,
        _ body: () throws -> T
    ) throws -> T {
        lockCount += 1
        return try body()
    }

    func loadInstallationMarker(root: PersistentCacheRootHandle) throws -> Data? {
        loadMarkerCount += 1
        return marker
    }

    func writeInstallationMarker(
        _ marker: Data,
        root: PersistentCacheRootHandle
    ) throws {
        writeMarkerCount += 1
        self.marker = marker
        onWrite?(marker)
    }
}

final class PersistentCacheHostBoundaryTests: XCTestCase {
    private let host = "com.example.RvllmTests"
    private let namespace = "authenticated-tenant"

    func testMemoryOnlyConfigurationPerformsNoSecurityOrFileAccess() throws {
        let keychain = MockKeychain()
        let files = MockFiles()
        var config = AppleEngineConfig()
        config.persistentCacheHostIdentifier = host
        config.persistentCacheNamespace = namespace

        XCTAssertNil(
            try PersistentCacheHostBoundary.prepare(
                config: config,
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(keychain.loadCount + keychain.randomCount, 0)
        XCTAssertEqual(files.prepareCount + files.lockCount, 0)
    }

    func testConsentIsRequiredBeforeAnySecurityOrFileAccess() {
        let keychain = MockKeychain()
        let files = MockFiles()
        var config = persistentConfig()
        config.persistentCacheConsent = false

        XCTAssertThrowsError(
            try PersistentCacheHostBoundary.prepare(
                config: config,
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(keychain.loadCount + keychain.randomCount, 0)
        XCTAssertEqual(files.prepareCount + files.lockCount, 0)
    }

    func testTenantAndExplicitHostAreValidatedBeforeAnyIO() {
        for mutation: (inout AppleEngineConfig) -> Void in [
            { $0.persistentCacheNamespace = nil },
            {
                $0.persistentCacheNamespace = "tenant"
                $0.persistentCacheHostIdentifier = nil
            },
        ] {
            let keychain = MockKeychain()
            let files = MockFiles()
            var config = persistentConfig()
            mutation(&config)
            XCTAssertThrowsError(
                try PersistentCacheHostBoundary.prepare(
                    config: config,
                    keychain: keychain,
                    files: files
                )
            )
            XCTAssertEqual(keychain.loadCount + keychain.randomCount, 0)
            XCTAssertEqual(files.prepareCount + files.lockCount, 0)
        }
    }

    func testNewKeyUsesNonSynchronizableThisDeviceOnlyAttributes() throws {
        let keychain = MockKeychain()
        let files = MockFiles()
        let material = try XCTUnwrap(
            PersistentCacheHostBoundary.prepare(
                config: persistentConfig(),
                keychain: keychain,
                files: files
            )
        )

        let write = try XCTUnwrap(keychain.lastAdd)
        XCTAssertEqual(write.service, "\(host).rvllm.PersistentPromptCache")
        XCTAssertEqual(write.account, PersistentCacheHostBoundary.keychainAccount)
        XCTAssertEqual(write.accessibility, PersistentCacheHostBoundary.accessibility)
        XCTAssertFalse(write.synchronizable)
        XCTAssertEqual(material.key.bytes, keychain.random)
        XCTAssertEqual(files.writeMarkerCount, 1)
    }

    func testDuplicateAddUsesWinnerAndNeverOverwritesIt() throws {
        let keychain = MockKeychain()
        let files = MockFiles()
        let winning = [UInt8](repeating: 0x5A, count: PersistentCacheHostBoundary.keyBytes)
        keychain.addStatus = errSecDuplicateItem
        keychain.duplicateWinner = PersistentCacheKeychainItem(
            key: winning,
            accessibility: PersistentCacheHostBoundary.accessibility
        )
        keychain.onDuplicateWinner = {
            files.marker = PersistentCacheHostBoundary.installationMarker(
                hostIdentifier: self.host,
                key: winning
            )
        }

        let material = try XCTUnwrap(
            PersistentCacheHostBoundary.prepare(
                config: persistentConfig(),
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(material.key.bytes, winning)
        XCTAssertEqual(keychain.addCount, 1)
        XCTAssertEqual(keychain.replaceCount, 0)
        XCTAssertEqual(files.writeMarkerCount, 0)
    }

    func testUnexpectedAddErrorFailsClosed() {
        let keychain = MockKeychain()
        let files = MockFiles()
        keychain.addStatus = errSecInteractionNotAllowed
        XCTAssertThrowsError(
            try PersistentCacheHostBoundary.prepare(
                config: persistentConfig(),
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(keychain.replaceCount, 0)
        XCTAssertEqual(files.writeMarkerCount, 0)
    }

    func testMissingInstallationMarkerRotatesSurvivingKey() throws {
        let keychain = MockKeychain()
        let files = MockFiles()
        let stale = [UInt8](repeating: 0x11, count: PersistentCacheHostBoundary.keyBytes)
        let replacement = [UInt8](repeating: 0x22, count: PersistentCacheHostBoundary.keyBytes)
        keychain.stored = PersistentCacheKeychainItem(
            key: stale,
            accessibility: PersistentCacheHostBoundary.accessibility
        )
        keychain.random = replacement

        let material = try XCTUnwrap(
            PersistentCacheHostBoundary.prepare(
                config: persistentConfig(),
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(material.key.bytes, replacement)
        XCTAssertEqual(keychain.replaceCount, 1)
        XCTAssertEqual(keychain.lastReplace?.accessibility, PersistentCacheHostBoundary.accessibility)
        XCTAssertEqual(keychain.lastReplace?.synchronizable, false)
        XCTAssertEqual(
            files.marker,
            PersistentCacheHostBoundary.installationMarker(
                hostIdentifier: host,
                key: replacement
            )
        )
    }

    func testRotationErrorFailsClosedWithoutPublishingMarker() {
        let keychain = MockKeychain()
        let files = MockFiles()
        keychain.stored = PersistentCacheKeychainItem(
            key: [UInt8](repeating: 1, count: PersistentCacheHostBoundary.keyBytes),
            accessibility: PersistentCacheHostBoundary.accessibility
        )
        keychain.replaceStatus = errSecInteractionNotAllowed
        XCTAssertThrowsError(
            try PersistentCacheHostBoundary.prepare(
                config: persistentConfig(),
                keychain: keychain,
                files: files
            )
        )
        XCTAssertEqual(files.writeMarkerCount, 0)
    }

    func testSensitiveBufferClearIsObservableAndIdempotent() {
        var cleared: [[UInt8]] = []
        let sensitive = PersistentCacheSensitiveBytes([1, 2, 3]) {
            cleared.append($0)
        }
        sensitive.clear()
        sensitive.clear()
        XCTAssertEqual(sensitive.bytes, [0, 0, 0])
        XCTAssertEqual(cleared, [[0, 0, 0]])
    }

    func testSystemRootAndMarkerArePrivateAndExcludedFromBackup() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: base) }
        let files = SystemPersistentCacheFileClient(applicationSupportOverride: base)
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: host)
        let rootValues = try root.url.resourceValues(
            forKeys: [.isDirectoryKey, .isSymbolicLinkKey, .isExcludedFromBackupKey]
        )
        XCTAssertEqual(rootValues.isDirectory, true)
        XCTAssertNotEqual(rootValues.isSymbolicLink, true)
        XCTAssertEqual(rootValues.isExcludedFromBackup, true)
        let permissions = try FileManager.default.attributesOfItem(atPath: root.url.path)[
            .posixPermissions
        ] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o700)

        let marker = Data(repeating: 7, count: PersistentCacheHostBoundary.markerBytes)
        try files.withInstallationLock(root: root) {
            try files.writeInstallationMarker(marker, root: root)
        }
        XCTAssertEqual(try files.loadInstallationMarker(root: root), marker)
        let markerURL = root.url.appendingPathComponent(".installation-key-marker-v1")
        let markerValues = try markerURL.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isExcludedFromBackupKey]
        )
        XCTAssertEqual(markerValues.isRegularFile, true)
        XCTAssertNotEqual(markerValues.isSymbolicLink, true)
        XCTAssertEqual(markerValues.isExcludedFromBackup, true)
    }

    func testSystemMarkerRejectsSymlinkHardlinkAndTruncation() throws {
        try withSystemRoot { files, root in
            let target = root.url.appendingPathComponent("symlink-target")
            try Data(repeating: 1, count: PersistentCacheHostBoundary.markerBytes)
                .write(to: target)
            let marker = root.url.appendingPathComponent(".installation-key-marker-v1")
            try FileManager.default.createSymbolicLink(at: marker, withDestinationURL: target)
            XCTAssertThrowsError(try files.loadInstallationMarker(root: root))
        }

        try withSystemRoot { files, root in
            let target = root.url.appendingPathComponent("hardlink-target")
            try Data(repeating: 2, count: PersistentCacheHostBoundary.markerBytes)
                .write(to: target)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: target.path
            )
            let marker = root.url.appendingPathComponent(".installation-key-marker-v1")
            try FileManager.default.linkItem(at: target, to: marker)
            XCTAssertThrowsError(try files.loadInstallationMarker(root: root))
        }

        try withSystemRoot { files, root in
            let marker = root.url.appendingPathComponent(".installation-key-marker-v1")
            try Data(repeating: 3, count: PersistentCacheHostBoundary.markerBytes - 1)
                .write(to: marker)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: marker.path
            )
            XCTAssertThrowsError(try files.loadInstallationMarker(root: root))
        }

        try withSystemRoot { files, root in
            let marker = root.url.appendingPathComponent(".installation-key-marker-v1")
            XCTAssertEqual(mkfifo(marker.path, 0o600), 0)
            XCTAssertThrowsError(try files.loadInstallationMarker(root: root))
        }
    }

    func testSystemInstallationLockRejectsHardlink() throws {
        try withSystemRoot { files, root in
            let target = root.url.appendingPathComponent("lock-target")
            try Data().write(to: target)
            let lock = root.url.appendingPathComponent(".installation-key.lock")
            try FileManager.default.linkItem(at: target, to: lock)
            XCTAssertThrowsError(
                try files.withInstallationLock(root: root) {
                    XCTFail("Hard-linked installation lock must not execute its body")
                }
            )
        }
    }

    func testPinnedRootKeepsLockAndMarkerOutOfReplacementDirectory() throws {
        let base = try makeTemporaryBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let files = SystemPersistentCacheFileClient(applicationSupportOverride: base)
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: host)
        let displaced = base.appendingPathComponent("displaced-root", isDirectory: true)
        try FileManager.default.moveItem(at: root.url, to: displaced)
        try FileManager.default.createDirectory(
            at: root.url,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700]
        )

        let marker = Data(repeating: 0x51, count: PersistentCacheHostBoundary.markerBytes)
        try files.withInstallationLock(root: root) {
            try files.writeInstallationMarker(marker, root: root)
        }

        XCTAssertEqual(
            try Data(contentsOf: displaced.appendingPathComponent(".installation-key-marker-v1")),
            marker
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.url.appendingPathComponent(".installation-key-marker-v1").path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.url.appendingPathComponent(".installation-key.lock").path
            )
        )
        XCTAssertEqual(try files.loadInstallationMarker(root: root), marker)
    }

    func testRootReplacementDuringMarkerPublicationCannotRedirectRename() throws {
        let base = try makeTemporaryBase()
        defer { try? FileManager.default.removeItem(at: base) }
        var rootURL: URL?
        let displaced = base.appendingPathComponent("displaced-during-publish", isDirectory: true)
        let files = SystemPersistentCacheFileClient(
            applicationSupportOverride: base,
            beforeMarkerPublish: {
                guard let root = rootURL else {
                    XCTFail("Test root was not initialized")
                    return
                }
                do {
                    try FileManager.default.moveItem(at: root, to: displaced)
                    try FileManager.default.createDirectory(
                        at: root,
                        withIntermediateDirectories: false,
                        attributes: [.posixPermissions: 0o700]
                    )
                } catch {
                    XCTFail("Could not replace root during publication: \(error)")
                }
            }
        )
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: host)
        rootURL = root.url
        let marker = Data(repeating: 0x62, count: PersistentCacheHostBoundary.markerBytes)

        try files.withInstallationLock(root: root) {
            try files.writeInstallationMarker(marker, root: root)
        }

        XCTAssertEqual(
            try Data(contentsOf: displaced.appendingPathComponent(".installation-key-marker-v1")),
            marker
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.url.appendingPathComponent(".installation-key-marker-v1").path
            )
        )
        XCTAssertEqual(try files.loadInstallationMarker(root: root), marker)
    }

    func testMarkerReplacementDuringPublicationIsAtomicAndDoesNotFollowSymlink() throws {
        let base = try makeTemporaryBase()
        defer { try? FileManager.default.removeItem(at: base) }
        var rootURL: URL?
        let target = base.appendingPathComponent("outside-marker-target")
        let targetBytes = Data(repeating: 0x73, count: PersistentCacheHostBoundary.markerBytes)
        try targetBytes.write(to: target)
        let files = SystemPersistentCacheFileClient(
            applicationSupportOverride: base,
            beforeMarkerPublish: {
                guard let root = rootURL else {
                    XCTFail("Test root was not initialized")
                    return
                }
                do {
                    try FileManager.default.createSymbolicLink(
                        at: root.appendingPathComponent(".installation-key-marker-v1"),
                        withDestinationURL: target
                    )
                } catch {
                    XCTFail("Could not inject marker symlink: \(error)")
                }
            }
        )
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: host)
        rootURL = root.url
        let marker = Data(repeating: 0x84, count: PersistentCacheHostBoundary.markerBytes)

        try files.withInstallationLock(root: root) {
            try files.writeInstallationMarker(marker, root: root)
        }

        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
        XCTAssertEqual(try files.loadInstallationMarker(root: root), marker)
        let values = try root.url
            .appendingPathComponent(".installation-key-marker-v1")
            .resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        XCTAssertEqual(values.isRegularFile, true)
        XCTAssertNotEqual(values.isSymbolicLink, true)
    }

    func testHostNamespaceSymlinkIsRejectedByComponentTraversal() throws {
        let base = try makeTemporaryBase()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
        defer {
            try? FileManager.default.removeItem(at: base)
            try? FileManager.default.removeItem(at: outside)
        }
        try FileManager.default.createSymbolicLink(
            at: base.appendingPathComponent("\(host).rvllm", isDirectory: true),
            withDestinationURL: outside
        )
        let files = SystemPersistentCacheFileClient(applicationSupportOverride: base)

        XCTAssertThrowsError(
            try files.prepareProtectedCacheRoot(hostIdentifier: host)
        )
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
    }

    func testPinnedRootAndInstallationLockRejectUnsafeModesAndObjectTypes() throws {
        try withSystemRoot { files, root in
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: root.url.path
            )
            XCTAssertThrowsError(
                try files.withInstallationLock(root: root) {
                    XCTFail("Non-private pinned root must not execute lock body")
                }
            )
        }

        for install: (URL) throws -> Void in [
            { try FileManager.default.createDirectory(at: $0, withIntermediateDirectories: false) },
            { XCTAssertEqual(mkfifo($0.path, 0o600), 0) },
            {
                try Data().write(to: $0)
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o644],
                    ofItemAtPath: $0.path
                )
            },
        ] {
            try withSystemRoot { files, root in
                let lock = root.url.appendingPathComponent(".installation-key.lock")
                try install(lock)
                XCTAssertThrowsError(
                    try files.withInstallationLock(root: root) {
                        XCTFail("Unsafe installation lock must not execute its body")
                    }
                )
            }
        }
    }

    private func persistentConfig() -> AppleEngineConfig {
        var config = AppleEngineConfig()
        config.cachePolicy = .persistentEncrypted
        config.persistentCacheConsent = true
        config.persistentCacheBytes = 1
        config.persistentCacheNamespace = namespace
        config.persistentCacheHostIdentifier = host
        return config
    }

    private func withSystemRoot(
        _ body: (SystemPersistentCacheFileClient, PersistentCacheRootHandle) throws -> Void
    ) throws {
        let base = try makeTemporaryBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let files = SystemPersistentCacheFileClient(applicationSupportOverride: base)
        let root = try files.prepareProtectedCacheRoot(hostIdentifier: host)
        try body(files, root)
    }

    private func makeTemporaryBase() throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: base,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700]
        )
        return base
    }
}
