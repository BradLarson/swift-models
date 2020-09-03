// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-models",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "0.2.0")),
    ],
    targets: [
        .target(name: "STBImage", path: "Support/STBImage"),
        .target(
            name: "ModelSupport", dependencies: ["STBImage"], path: "Support",
            exclude: ["STBImage"]),
       .target(
           name: "Fractals",
           dependencies: ["ArgumentParser", "ModelSupport"],
           path: "Examples/Fractals"
       )
    ]
)
