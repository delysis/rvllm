#!/usr/bin/env python3
"""Fail closed when shipping Apple artifacts reference private ANE APIs.

The scanner intentionally operates on bytes rather than relying on `nm`,
`otool`, or `strings`. That keeps its result identical on developer machines
and minimal CI images, and also catches dynamically looked-up Objective-C
class names and framework paths that do not appear in a symbol table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


RESEARCH_FEATURE = "macos-private-ane-research"

# Do not ban generic "NeuralEngine" text: public Core ML exposes the
# CPUAndNeuralEngine compute-unit request. These byte sequences are private API
# identities or private-framework locations.
FORBIDDEN_MARKERS: tuple[tuple[str, bytes], ...] = (
    ("private-framework-path", b"/System/Library/PrivateFrameworks/"),
    ("apple-neural-engine-framework", b"AppleNeuralEngine.framework"),
    ("apple-neural-engine-binary", b"AppleNeuralEngine"),
    ("private-ane-class-client", b"_ANEClient"),
    ("private-ane-class-compiler", b"_ANECompiler"),
    ("private-ane-class-iosurface", b"_ANEIOSurfaceObject"),
    ("private-ane-class-model", b"_ANEModel"),
    ("private-ane-class-program", b"_ANEProgramForEvaluation"),
    ("private-ane-class-request", b"_ANERequest"),
)

_ARCHIVE_MAGIC = b"!<arch>\n"
_MACHO_MAGICS = {
    bytes.fromhex("feedface"),
    bytes.fromhex("feedfacf"),
    bytes.fromhex("cefaedfe"),
    bytes.fromhex("cffaedfe"),
    bytes.fromhex("cafebabe"),
    bytes.fromhex("bebafeca"),
    bytes.fromhex("cafebabf"),
    bytes.fromhex("bfbafeca"),
}
_BINARY_SUFFIXES = {".a", ".dylib", ".framework", ".rlib"}
_RESEARCH_ARTIFACT_TOKENS = (
    "rvllm_apple_ane_sys",
    "rvllm-apple-ane-sys",
    "macos_private_ane_research",
    "macos-private-ane-research",
)


class ScanConfigurationError(ValueError):
    """The requested scan is unsafe or ambiguous."""


class ArtifactReadError(OSError):
    """An artifact could not be discovered or read."""


@dataclass(frozen=True, order=True)
class Finding:
    artifact: Path
    marker: str
    offset: int


def _is_compiled_artifact(path: Path) -> bool:
    """Return whether a directory member looks like a compiled Apple artifact."""
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError as exc:
        raise ArtifactReadError(f"cannot read artifact candidate {path}: {exc}") from exc
    return header == _ARCHIVE_MAGIC or header[:4] in _MACHO_MAGICS


def discover_artifacts(inputs: Sequence[Path]) -> list[Path]:
    """Expand files/directories into a stable, duplicate-free artifact list."""
    artifacts: dict[Path, Path] = {}
    for supplied in inputs:
        if not supplied.exists():
            raise ArtifactReadError(f"scan input does not exist: {supplied}")
        if supplied.is_file():
            resolved = supplied.resolve()
            artifacts[resolved] = resolved
            continue
        if not supplied.is_dir():
            raise ArtifactReadError(f"scan input is not a file or directory: {supplied}")
        try:
            members = sorted(
                (member for member in supplied.rglob("*") if member.is_file()),
                key=lambda member: str(member),
            )
        except OSError as exc:
            raise ArtifactReadError(f"cannot enumerate scan input {supplied}: {exc}") from exc
        for member in members:
            if _is_compiled_artifact(member):
                resolved = member.resolve()
                artifacts[resolved] = resolved
    if not artifacts:
        rendered = ", ".join(str(path) for path in inputs)
        raise ArtifactReadError(f"no compiled artifacts found in scan inputs: {rendered}")
    return sorted(artifacts.values(), key=lambda path: str(path))


def scan_artifact(path: Path, chunk_size: int = 1024 * 1024) -> list[Finding]:
    """Find the first occurrence of every forbidden marker using bounded RAM."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap_size = max(len(marker) for _, marker in FORBIDDEN_MARKERS) - 1
    first_offsets: dict[str, int] = {}
    tail = b""
    consumed = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                window = tail + chunk
                window_start = consumed - len(tail)
                for marker_name, marker_bytes in FORBIDDEN_MARKERS:
                    if marker_name in first_offsets:
                        continue
                    index = window.find(marker_bytes)
                    if index >= 0:
                        first_offsets[marker_name] = window_start + index
                consumed += len(chunk)
                tail = window[-overlap_size:] if overlap_size else b""
    except OSError as exc:
        raise ArtifactReadError(f"cannot scan artifact {path}: {exc}") from exc
    return sorted(
        (Finding(path, marker, offset) for marker, offset in first_offsets.items()),
        key=lambda finding: (str(finding.artifact), finding.offset, finding.marker),
    )


def validate_research_allowlist(
    allowed: Iterable[Path],
    *,
    enabled_feature: str | None,
    system: str | None = None,
) -> set[Path]:
    """Validate the deliberately narrow macOS research-artifact exception."""
    resolved = {path.resolve() for path in allowed}
    if not resolved:
        if enabled_feature is not None:
            raise ScanConfigurationError(
                "--enabled-feature is only valid with an explicit research artifact allowlist"
            )
        return set()
    if enabled_feature != RESEARCH_FEATURE:
        raise ScanConfigurationError(
            "research exceptions require --enabled-feature macos-private-ane-research"
        )
    host_system = system if system is not None else platform.system()
    if host_system != "Darwin":
        raise ScanConfigurationError("private ANE research artifacts are macOS-only")
    for path in sorted(resolved, key=lambda item: str(item)):
        if not path.exists() or not path.is_file():
            raise ScanConfigurationError(f"research allowlist entry is not a file: {path}")
        name = path.name.lower()
        if not any(token in name for token in _RESEARCH_ARTIFACT_TOKENS):
            raise ScanConfigurationError(
                "research allowlist entry must be explicitly named as the private ANE "
                f"system/research artifact: {path}"
            )
    return resolved


def scan(
    inputs: Sequence[Path],
    *,
    allowed_research_artifacts: set[Path] | None = None,
) -> tuple[list[Path], list[Path], list[Finding]]:
    """Return scanned, explicitly allowed, and violating artifacts."""
    allowlist = allowed_research_artifacts or set()
    artifacts = discover_artifacts(inputs)
    unknown_allowed = allowlist.difference(artifacts)
    if unknown_allowed:
        rendered = ", ".join(str(path) for path in sorted(unknown_allowed, key=str))
        raise ScanConfigurationError(
            f"research allowlist entries are outside the supplied scan inputs: {rendered}"
        )
    allowed: list[Path] = []
    findings: list[Finding] = []
    for artifact in artifacts:
        artifact_findings = scan_artifact(artifact)
        if artifact in allowlist:
            allowed.append(artifact)
        else:
            findings.extend(artifact_findings)
    return artifacts, allowed, sorted(findings)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject private Apple/ANE references in compiled shipping artifacts."
    )
    parser.add_argument("artifact", nargs="+", type=Path, help="artifact file or directory")
    parser.add_argument(
        "--allow-macos-private-ane-research-artifact",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "explicitly allow one clearly named private-ANE research artifact; "
            "requires the exact research feature and is never valid for shipping artifacts"
        ),
    )
    parser.add_argument(
        "--enabled-feature",
        choices=[RESEARCH_FEATURE],
        help="feature proving an allowlisted artifact came from the macOS research build",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        allowlist = validate_research_allowlist(
            args.allow_macos_private_ane_research_artifact,
            enabled_feature=args.enabled_feature,
        )
        artifacts, allowed, findings = scan(
            args.artifact, allowed_research_artifacts=allowlist
        )
    except (ArtifactReadError, ScanConfigurationError, ValueError) as exc:
        print(f"apple release symbol scan configuration error: {exc}", file=sys.stderr)
        return 2

    for artifact in allowed:
        print(
            f"RESEARCH-ONLY ALLOWLIST: {artifact} "
            f"(feature={RESEARCH_FEATURE}; never ship this artifact)"
        )
    if findings:
        print("private Apple API references found in shipping artifacts:", file=sys.stderr)
        for finding in findings:
            print(
                f"  {finding.artifact}: offset {finding.offset}: {finding.marker}",
                file=sys.stderr,
            )
        return 1

    shipping_count = len(artifacts) - len(allowed)
    print(
        f"Apple release symbol scan passed: {shipping_count} shipping artifact(s), "
        f"{len(allowed)} explicit research artifact(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
