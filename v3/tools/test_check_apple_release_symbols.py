#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_apple_release_symbols as scanner


class AppleReleaseSymbolScanTests(unittest.TestCase):
    def artifact(self, root: Path, name: str, body: bytes) -> Path:
        path = root / name
        path.write_bytes(scanner._ARCHIVE_MAGIC + body)
        return path

    def test_clean_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.artifact(
                Path(directory),
                "libshipping.rlib",
                b"CoreML PRIVATE_ANE_ENV_VAR CPUAndNeuralEngine",
            )
            artifacts, allowed, findings = scanner.scan([artifact])
            self.assertEqual(artifacts, [artifact.resolve()])
            self.assertEqual(allowed, [])
            self.assertEqual(findings, [])

    def test_public_coreml_runtime_markers_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.artifact(
                Path(directory),
                "librvllm_apple_coreml_runtime.rlib",
                (
                    b"/System/Library/Frameworks/CoreML.framework/CoreML "
                    b"MLModel MLModelConfiguration MLComputePlan "
                    b"CPUAndNeuralEngine"
                ),
            )
            _, allowed, findings = scanner.scan([artifact])
            self.assertEqual(allowed, [])
            self.assertEqual(findings, [])

    def test_private_markers_are_reported_with_stable_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.artifact(
                Path(directory),
                "libshipping.rlib",
                b"prefix_ANEClient-middle-AppleNeuralEngine.framework",
            )
            findings = scanner.scan_artifact(artifact, chunk_size=11)
            self.assertEqual(
                [(item.marker, item.offset) for item in findings],
                [
                    ("private-ane-class-client", 14),
                    ("apple-neural-engine-binary", 32),
                    ("apple-neural-engine-framework", 32),
                ],
            )

    def test_directory_scan_ignores_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("_ANEClient", encoding="utf-8")
            clean = self.artifact(root, "libshipping.rlib", b"safe")
            artifacts = scanner.discover_artifacts([root])
            self.assertEqual(artifacts, [clean.resolve()])

    def test_xcframework_directory_scans_every_platform_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "RvllmApple.xcframework"
            device = root / "ios-arm64"
            simulator = root / "ios-arm64-simulator"
            device.mkdir(parents=True)
            simulator.mkdir(parents=True)
            clean = self.artifact(device, "librvllm_apple_ffi.a", b"public CoreML")
            private = self.artifact(
                simulator, "librvllm_apple_ffi.a", b"prefix_ANEClient_suffix"
            )

            artifacts, allowed, findings = scanner.scan([root])

            self.assertEqual(
                artifacts,
                sorted([clean.resolve(), private.resolve()], key=lambda path: str(path)),
            )
            self.assertEqual(allowed, [])
            self.assertEqual(
                [(finding.artifact, finding.marker) for finding in findings],
                [(private.resolve(), "private-ane-class-client")],
            )

    def test_missing_or_empty_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(scanner.ArtifactReadError):
                scanner.discover_artifacts([root / "missing"])
            with self.assertRaises(scanner.ArtifactReadError):
                scanner.discover_artifacts([root])

    def test_research_allowlist_requires_exact_feature_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shipping = self.artifact(root, "librvllm_apple.rlib", b"_ANEClient")
            with self.assertRaises(scanner.ScanConfigurationError):
                scanner.validate_research_allowlist(
                    [shipping], enabled_feature=scanner.RESEARCH_FEATURE, system="Darwin"
                )

            research = self.artifact(
                root, "librvllm_apple_ane_sys.rlib", b"_ANEClient"
            )
            with self.assertRaises(scanner.ScanConfigurationError):
                scanner.validate_research_allowlist(
                    [research], enabled_feature=None, system="Darwin"
                )
            allowlist = scanner.validate_research_allowlist(
                [research],
                enabled_feature=scanner.RESEARCH_FEATURE,
                system="Darwin",
            )
            _, allowed, findings = scanner.scan(
                [research], allowed_research_artifacts=allowlist
            )
            self.assertEqual(allowed, [research.resolve()])
            self.assertEqual(findings, [])

    def test_research_allowlist_is_macos_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            research = self.artifact(
                Path(directory), "librvllm_apple_ane_sys.rlib", b"_ANEClient"
            )
            with self.assertRaises(scanner.ScanConfigurationError):
                scanner.validate_research_allowlist(
                    [research],
                    enabled_feature=scanner.RESEARCH_FEATURE,
                    system="Linux",
                )


if __name__ == "__main__":
    unittest.main()
