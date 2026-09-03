"""Atomic capture-bundle publication for GEO-CAP-001."""
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
from .canonical import canonical_json_bytes
from .capture_common import CaptureContractError
from .capture_verify import verify_capture_bundle


def _fsync_directory(path: Path) -> None:
    """Sync directory metadata where the platform exposes POSIX directory fsync.

    Windows does not permit opening directories with ``os.open`` in the same
    way POSIX does. On Windows the writer still fsyncs every file and publishes
    the staged directory with one same-volume ``os.replace``; the extra
    directory-metadata fsync step is therefore intentionally skipped.
    """

    if os.name == "nt":
        return
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        raise CaptureContractError(
            "canonical publication requires directory-fsync support on non-Windows platforms"
        )
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_capture_bundle(output_dir: Path, request: Mapping[str, Any], manifest: Mapping[str, Any], trajectory: Mapping[str, Any]) -> None:
    validated = verify_capture_bundle(request, manifest, trajectory)
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise CaptureContractError("output_dir already exists; canonical capture bundles are immutable publications")
    payloads = {"capture-request.json": validated, "run-manifest.json": manifest, "captured-trajectory.json": trajectory}
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(parent)))
    published = False
    try:
        for name, payload in payloads.items():
            with (staging / name).open("xb") as handle:
                handle.write(canonical_json_bytes(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if output_dir.exists() or output_dir.is_symlink():
            raise CaptureContractError("output_dir appeared during publication; refusing to replace it")
        os.replace(staging, output_dir)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
