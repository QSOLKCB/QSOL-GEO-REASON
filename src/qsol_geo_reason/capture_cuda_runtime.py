"""Content receipts for CUDA/NVIDIA shared objects mapped into the capture process."""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

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

_WIN_HANDLE = ctypes.c_void_p
_WIN_HMODULE = ctypes.c_void_p
_WIN_DWORD = ctypes.c_uint32
_WIN_BOOL = ctypes.c_int
_LibraryPredicate = Callable[[str], bool]


class _MappedLibrary:
    """A loaded library plus its kernel-recorded mapped-file identity."""

    __slots__ = ("path", "content_path", "device", "inode")

    def __init__(
        self,
        path: Path,
        content_path: Path,
        device: int | None = None,
        inode: int | None = None,
    ) -> None:
        # Keep this constructor source-backed. Canonical checkout provenance rejects
        # exec-generated callables, including dataclass-generated __init__ methods.
        self.path = Path(path)
        self.content_path = Path(content_path)
        self.device = device
        self.inode = inode


def _is_cuda_runtime_library(path: str) -> bool:
    name = Path(path).name
    return bool(_LINUX_CUDA_LIBRARY.fullmatch(name) or _WINDOWS_CUDA_LIBRARY.fullmatch(name))


def _linux_loaded_library_mappings(
    predicate: _LibraryPredicate = _is_cuda_runtime_library,
) -> list[_MappedLibrary]:
    """Enumerate mapped libraries with their kernel-recorded device/inode identity."""
    maps = Path("/proc/self/maps")
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise CaptureContractError(
            "canonical observation cannot enumerate loaded shared objects from /proc/self/maps"
        ) from exc
    result: dict[tuple[int, int, str], _MappedLibrary] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        mapping_range, device_text, inode_text = fields[0], fields[3], fields[4]
        raw = fields[5]
        deleted = raw.endswith(" (deleted)")
        candidate = raw[:-10] if deleted else raw
        if not candidate.startswith("/") or not predicate(candidate):
            continue
        if deleted:
            raise CaptureContractError(
                f"loaded runtime library was deleted after mapping: {candidate}"
            )
        try:
            major_text, minor_text = device_text.split(":", 1)
            device = os.makedev(int(major_text, 16), int(minor_text, 16))
            inode = int(inode_text, 10)
        except (ValueError, OSError) as exc:
            raise CaptureContractError(
                f"loaded runtime library has malformed mapped file identity: {candidate}"
            ) from exc
        if inode <= 0:
            raise CaptureContractError(
                f"loaded runtime library has no persistent mapped inode: {candidate}"
            )
        key = (device, inode, candidate)
        result.setdefault(
            key,
            _MappedLibrary(
                path=Path(candidate),
                # Do not require /proc/self/map_files: opening those entries needs
                # elevated capabilities on many ordinary Linux/container hosts.
                # The ordinary pathname is opened once and its descriptor is
                # authenticated against this mapped device/inode before hashing.
                content_path=Path(candidate),
                device=device,
                inode=inode,
            ),
        )
    return sorted(
        result.values(),
        key=lambda item: (str(item.path), str(item.content_path)),
    )


def _linux_loaded_library_paths(
    predicate: _LibraryPredicate = _is_cuda_runtime_library,
) -> list[Path]:
    """Compatibility view of mapped Linux libraries by their display path."""
    return [item.path for item in _linux_loaded_library_mappings(predicate)]


def _configure_windows_module_api(kernel32: Any, psapi: Any) -> None:
    """Declare pointer-sized Win32 module-enumeration signatures explicitly."""
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = _WIN_HANDLE
    psapi.EnumProcessModules.argtypes = [
        _WIN_HANDLE,
        ctypes.POINTER(_WIN_HMODULE),
        _WIN_DWORD,
        ctypes.POINTER(_WIN_DWORD),
    ]
    psapi.EnumProcessModules.restype = _WIN_BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        _WIN_HANDLE,
        _WIN_HMODULE,
        ctypes.POINTER(ctypes.c_wchar),
        _WIN_DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = _WIN_DWORD


def _windows_loaded_library_paths(
    predicate: _LibraryPredicate = _is_cuda_runtime_library,
) -> list[Path]:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        _configure_windows_module_api(kernel32, psapi)
    except Exception as exc:
        raise CaptureContractError("unable to access Windows module enumeration APIs") from exc

    hprocess = kernel32.GetCurrentProcess()
    needed = _WIN_DWORD()
    count = 1024
    while True:
        modules = (_WIN_HMODULE * count)()
        if not psapi.EnumProcessModules(
            hprocess,
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
        ):
            raise CaptureContractError("unable to enumerate loaded Windows runtime modules")
        required = needed.value // ctypes.sizeof(_WIN_HMODULE)
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
        if predicate(candidate):
            result.add(Path(candidate))
    return sorted(result, key=lambda item: str(item))


def _darwin_loaded_library_paths(predicate: _LibraryPredicate) -> list[Path]:
    try:
        process = ctypes.CDLL(None)
        image_count = getattr(process, "_dyld_image_count")
        image_name = getattr(process, "_dyld_get_image_name")
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
    except Exception as exc:
        raise CaptureContractError("unable to access Darwin dyld image enumeration APIs") from exc

    result: set[Path] = set()
    count = int(image_count())
    if count > 65536:
        raise CaptureContractError("loaded Darwin image count exceeds canonical bound")
    for index in range(count):
        raw = image_name(index)
        if not raw:
            continue
        candidate = os.fsdecode(raw)
        if predicate(candidate):
            result.add(Path(candidate))
    return sorted(result, key=lambda item: str(item))


def _loaded_cuda_library_entries() -> list[Path | _MappedLibrary]:
    if sys.platform.startswith("linux"):
        return list(_linux_loaded_library_mappings())
    if os.name == "nt":
        return list(_windows_loaded_library_paths())
    raise CaptureContractError(
        "canonical CUDA observation cannot enumerate loaded CUDA shared objects on this platform"
    )


def _loaded_cuda_library_paths() -> list[Path]:
    """Compatibility view for callers that only need display pathnames."""
    entries = _loaded_cuda_library_entries()
    return [entry.path if isinstance(entry, _MappedLibrary) else Path(entry) for entry in entries]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CaptureContractError(f"unable to hash loaded runtime library {path}") from exc
    return digest.hexdigest()


def _sha256_fd(fd: int, where: Path) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise CaptureContractError(f"unable to hash mapped runtime library {where}") from exc
    return digest.hexdigest()


def _runtime_library_digest(
    raw_path: Path | _MappedLibrary,
    *,
    predicate: _LibraryPredicate,
    label: str,
) -> tuple[str, str] | None:
    if isinstance(raw_path, _MappedLibrary):
        display_path = raw_path.path
        if not predicate(str(display_path)):
            return None
        if raw_path.device is None or raw_path.inode is None:
            raise CaptureContractError(
                f"mapped {label} runtime library is missing device/inode identity: {display_path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            # Production Linux mappings use the ordinary pathname from
            # /proc/self/maps. fstat() below proves that this descriptor still
            # names the exact mapped device/inode before any bytes are hashed.
            # Replacement before open fails authentication; replacement after
            # open cannot change the inode pinned by this descriptor.
            fd = os.open(raw_path.content_path, flags)
        except OSError as exc:
            raise CaptureContractError(
                f"unable to open mapped {label} runtime library {display_path}"
            ) from exc
        try:
            observed = os.fstat(fd)
            if not stat.S_ISREG(observed.st_mode):
                raise CaptureContractError(
                    f"mapped {label} runtime object is not a regular file: {display_path}"
                )
            if observed.st_dev != raw_path.device or observed.st_ino != raw_path.inode:
                raise CaptureContractError(
                    f"mapped {label} runtime library identity changed before hashing: {display_path}"
                )
            digest = _sha256_fd(fd, display_path)
        finally:
            os.close(fd)
        return display_path.name, digest

    try:
        path = Path(raw_path).resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError(
            f"loaded {label} runtime library path is unavailable: {raw_path}"
        ) from exc
    if not path.is_file() or not predicate(str(path)):
        return None
    return path.name, _sha256_file(path)


def _cuda_runtime_library_receipt(
    paths: Iterable[Path | _MappedLibrary],
) -> tuple[int, str]:
    """Hash actual mapped library bytes; relocation does not change the receipt."""
    by_name: dict[str, str] = {}
    for raw_path in paths:
        item = _runtime_library_digest(
            raw_path,
            predicate=_is_cuda_runtime_library,
            label="CUDA",
        )
        if item is None:
            continue
        name, digest = item
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
    count, receipt = _cuda_runtime_library_receipt(_loaded_cuda_library_entries())
    return {
        "cuda_runtime_library_file_count": count,
        "cuda_runtime_library_receipt_sha256": receipt,
    }


__all__ = ["loaded_cuda_runtime_library_provenance"]
