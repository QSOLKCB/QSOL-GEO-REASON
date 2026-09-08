"""Content receipts for CUDA/NVIDIA shared objects mapped into the capture process."""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable

from .canonical import sha256_json
from .capture_common import CaptureContractError


_LINUX_CUDA_LIBRARY = re.compile(
    r"^(?:libcuda|libcudart|libcublas(?:Lt)?|libcudnn[^/]*|libcusparse|libcusolver|libcurand|libnvrtc|libnvJitLink|libnccl)[.]so(?:[.].*)?$",
    re.IGNORECASE,
)
_WINDOWS_CUDA_LIBRARY = re.compile(
    r"^(?:nvcuda[.]dll|cudart64_[^/]+[.]dll|cublas(?:Lt)?64_[^/]+[.]dll|cudnn[^/]*[.]dll|cusparse64_[^/]+[.]dll|cusolver64_[^/]+[.]dll|curand64_[^/]+[.]dll|nvrtc64_[^/]+[.]dll|nvJitLink_[^/]+[.]dll|nccl[^/]*[.]dll)$",
    re.IGNORECASE,
)


def _is_cuda_runtime_library(path: str) -> bool:
    name = Path(path).name
    return bool(_LINUX_CUDA_LIBRARY.fullmatch(name) or _WINDOWS_CUDA_LIBRARY.fullmatch(name))


def _linux_loaded_library_paths() -> list[Path]:
    maps = Path("/proc/self/maps")
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise CaptureContractError(
            "canonical CUDA observation cannot enumerate loaded shared objects from /proc/self/maps"
        ) from exc
    result: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        raw = fields[5]
        deleted = raw.endswith(" (deleted)")
        candidate = raw[:-10] if deleted else raw
        if not candidate.startswith("/") or not _is_cuda_runtime_library(candidate):
            continue
        if deleted:
            raise CaptureContractError(
                f"loaded CUDA runtime library was deleted after mapping: {candidate}"
            )
        result.add(Path(candidate))
    return sorted(result, key=lambda item: str(item))


def _windows_loaded_library_paths() -> list[Path]:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
    except Exception as exc:
        raise CaptureContractError("unable to access Windows module enumeration APIs") from exc

    hprocess = kernel32.GetCurrentProcess()
    needed = ctypes.c_ulong()
    count = 1024
    while True:
        modules = (ctypes.c_void_p * count)()
        if not psapi.EnumProcessModules(
            hprocess,
            ctypes.byref(modules),
            ctypes.sizeof(modules),
            ctypes.byref(needed),
        ):
            raise CaptureContractError("unable to enumerate loaded Windows CUDA modules")
        required = needed.value // ctypes.sizeof(ctypes.c_void_p)
        if required <= count:
            break
        count = required + 64
        if count > 65536:
            raise CaptureContractError("loaded Windows module count exceeds canonical bound")

    result: set[Path] = set()
    for index in range(required):
        buffer = ctypes.create_unicode_buffer(32768)
        length = psapi.GetModuleFileNameExW(
            hprocess,
            modules[index],
            buffer,
            len(buffer),
        )
        if not length:
            continue
        candidate = buffer.value
        if _is_cuda_runtime_library(candidate):
            result.add(Path(candidate))
    return sorted(result, key=lambda item: str(item))


def _loaded_cuda_library_paths() -> list[Path]:
    if sys.platform.startswith("linux"):
        return _linux_loaded_library_paths()
    if os.name == "nt":
        return _windows_loaded_library_paths()
    raise CaptureContractError(
        "canonical CUDA observation cannot enumerate loaded CUDA shared objects on this platform"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CaptureContractError(f"unable to hash loaded CUDA runtime library {path}") from exc
    return digest.hexdigest()


def _cuda_runtime_library_receipt(paths: Iterable[Path]) -> tuple[int, str]:
    """Hash actual mapped library bytes; relocation does not change the receipt."""
    by_name: dict[str, str] = {}
    for raw_path in paths:
        try:
            path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise CaptureContractError(f"loaded CUDA runtime library path is unavailable: {raw_path}") from exc
        if not path.is_file() or not _is_cuda_runtime_library(str(path)):
            continue
        name = path.name
        digest = _sha256_file(path)
        prior = by_name.get(name)
        if prior is not None and prior != digest:
            raise CaptureContractError(
                f"multiple loaded CUDA libraries share basename {name!r} with different content"
            )
        by_name[name] = digest
    if not by_name:
        raise CaptureContractError(
            "canonical CUDA observation could not content-bind any loaded CUDA/NVIDIA shared objects"
        )
    return len(by_name), sha256_json(dict(sorted(by_name.items())))


def loaded_cuda_runtime_library_provenance() -> dict[str, int | str]:
    count, receipt = _cuda_runtime_library_receipt(_loaded_cuda_library_paths())
    return {
        "cuda_runtime_library_file_count": count,
        "cuda_runtime_library_receipt_sha256": receipt,
    }


__all__ = ["loaded_cuda_runtime_library_provenance"]
