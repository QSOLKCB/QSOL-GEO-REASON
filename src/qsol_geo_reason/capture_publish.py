"""Atomic capture-bundle publication for GEO-CAP-001."""
from __future__ import annotations
import ctypes
import errno
import os
import shutil
import sys
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
    the staged directory with one same-volume no-replace rename; the extra
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


def _raise_publish_error(error_number: int) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CaptureContractError(
            "output_dir already exists; canonical capture bundles are immutable publications"
        )
    raise OSError(error_number, os.strerror(error_number))


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` only if ``destination`` does not exist.

    Canonical publication fails closed when the platform does not expose an
    atomic no-replace directory rename primitive. This removes the TOCTOU gap
    inherent in an existence check followed by ``os.replace``.
    """

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)

    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise CaptureContractError(
                "output_dir already exists; canonical capture bundles are immutable publications"
            ) from exc
        return

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise CaptureContractError(
                "canonical publication requires Linux renameat2(RENAME_NOREPLACE)"
            )
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        if renameat2(at_fdcwd, source_bytes, at_fdcwd, destination_bytes, rename_noreplace) != 0:
            _raise_publish_error(ctypes.get_errno())
        return

    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise CaptureContractError(
                "canonical publication requires macOS renamex_np(RENAME_EXCL)"
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        rename_excl = 0x00000004
        if renamex_np(source_bytes, destination_bytes, rename_excl) != 0:
            _raise_publish_error(ctypes.get_errno())
        return

    raise CaptureContractError(
        "canonical publication requires an atomic no-replace directory rename primitive"
    )


def write_capture_bundle(output_dir: Path, request: Mapping[str, Any], manifest: Mapping[str, Any], trajectory: Mapping[str, Any]) -> None:
    validated = verify_capture_bundle(request, manifest, trajectory)
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
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
        _rename_directory_noreplace(staging, output_dir)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
