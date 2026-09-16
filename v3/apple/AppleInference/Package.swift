// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AppleInference",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "AppleInference", targets: ["AppleInference"]),
    ],
    targets: [
        .target(
            name: "CRvllmApple",
            publicHeadersPath: "include"
        ),
        .target(
            name: "CPersistentCacheHost",
            publicHeadersPath: "include"
        ),
        .target(
            name: "AppleInference",
            dependencies: ["CRvllmApple", "CPersistentCacheHost"]
        ),
        .target(
            name: "CRvllmAppleTestSupport",
            dependencies: ["CRvllmApple"],
            path: "Tests/CRvllmAppleTestSupport",
            publicHeadersPath: "include"
        ),
        .testTarget(
            name: "AppleInferenceTests",
            dependencies: ["AppleInference", "CRvllmAppleTestSupport"]
        ),
    ]
)
