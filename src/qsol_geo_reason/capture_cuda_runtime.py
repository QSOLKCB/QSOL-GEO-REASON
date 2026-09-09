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


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _runtime_library_measurement(
    raw_path: Path | _MappedLibrary,
    *,
    predicate: _LibraryPredicate,
    label: str,
) -> tuple[str, str, str, tuple[int, int, int, int, int]] | None:
    if isinstance(raw_path, _MappedLibrary):
        display_path = raw_path.path
        content_path = raw_path.content_path
        expected_device = raw_path.device
        expected_inode = raw_path.inode
        if expected_device is None or expected_inode is None:
            raise CaptureContractError(
                f"mapped {label} runtime library is missing device/inode identity: {display_path}"
            )
    else:
        display_path = Path(raw_path)
        content_path = display_path
        expected_device = None
        expected_inode = None

    if not predicate(str(display_path)):
        return None

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        # Open the ordinary pathname exactly once.  For Linux mappings the
        # descriptor is authenticated against /proc/self/maps device/inode data;
        # this avoids privileged /proc/self/map_files while still pinning identity.
        fd = os.open(content_path, flags)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to open mapped {label} runtime library {display_path}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CaptureContractError(
                f"mapped {label} runtime object is not a regular file: {display_path}"
            )
        if (
            expected_device is not None
            and (before.st_dev != expected_device or before.st_ino != expected_inode)
        ):
            raise CaptureContractError(
                f"mapped {label} runtime library identity changed before hashing: {display_path}"
            )
        before_fingerprint = _stat_fingerprint(before)
        digest = _sha256_fd(fd, display_path)
        after_fingerprint = _stat_fingerprint(os.fstat(fd))
        if after_fingerprint != before_fingerprint:
            raise CaptureContractError(
                f"mapped {label} runtime library changed while hashing: {display_path}"
            )
    finally:
        os.close(fd)
    return str(display_path), display_path.name, digest, before_fingerprint


def _runtime_library_digest(
    raw_path: Path | _MappedLibrary,
    *,
    predicate: _LibraryPredicate,
    label: str,
) -> tuple[str, str] | None:
    measured = _runtime_library_measurement(
        raw_path, predicate=predicate, label=label
    )
    if measured is None:
        return None
    _display_path, name, digest, _fingerprint = measured
    return name, digest


def _runtime_library_snapshot(
    paths: Iterable[Path | _MappedLibrary],
    *,
    predicate: _LibraryPredicate,
    label: str,
    require_nonempty: bool,
) -> tuple[int, str, tuple[tuple[Any, ...], ...]]:
    """Return content provenance plus a stat-bound execution-stability receipt."""
    by_name: dict[str, str] = {}
    stability: list[tuple[Any, ...]] = []
    for raw_path in paths:
        measured = _runtime_library_measurement(
            raw_path, predicate=predicate, label=label
        )
        if measured is None:
            continue
        display_path, name, digest, fingerprint = measured
        prior = by_name.get(name)
        if prior is not None and prior != digest:
            raise CaptureContractError(
                f"multiple loaded {label} libraries share basename {name!r} with different content"
            )
        by_name[name] = digest
        stability.append((display_path, name, digest, *fingerprint))
    if require_nonempty and not by_name:
        raise CaptureContractError(
            f"canonical {label} observation could not content-bind any loaded {label}/NVIDIA shared objects"
            if label == "CUDA"
            else f"canonical {label} observation could not content-bind any loaded runtime shared objects"
        )
    return (
        len(by_name),
        sha256_json(dict(sorted(by_name.items()))),
        tuple(sorted(stability, key=lambda item: (str(item[0]), str(item[1])))),
    )


def assert_runtime_library_state_stable(
    before: tuple[tuple[Any, ...], ...],
    after: tuple[tuple[Any, ...], ...],
    *,
    observation_started_ns: int,
    label: str,
) -> None:
    """Reject runtime-file drift across the canonical observation window.

    Existing mappings are authenticated by exact baseline-to-final descriptor, content,
    and stat equality.  Their absolute mtimes are deliberately not compared with wall
    clock start time because package extraction may preserve a legitimate future build
    timestamp.  A library first observed after the baseline is accepted only when its
    change-time predates the observation; on POSIX, ctime also exposes in-place
    write/restore ABA even when final bytes and preserved mtime match again.
    """
    if (
        isinstance(observation_started_ns, bool)
        or not isinstance(observation_started_ns, int)
        or observation_started_ns <= 0
    ):
        raise CaptureContractError("runtime-library observation timestamp is malformed")

    def normalize(
        entries: tuple[tuple[Any, ...], ...],
    ) -> dict[str, tuple[Any, ...]]:
        normalized: dict[str, tuple[Any, ...]] = {}
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 8:
                raise CaptureContractError(f"{label} runtime stability receipt is malformed")
            display_path = entry[0]
            if not isinstance(display_path, str) or not display_path:
                raise CaptureContractError(f"{label} runtime stability receipt is malformed")
            mtime_ns = entry[6]
            ctime_ns = entry[7]
            if (
                isinstance(mtime_ns, bool)
                or not isinstance(mtime_ns, int)
                or isinstance(ctime_ns, bool)
                or not isinstance(ctime_ns, int)
            ):
                raise CaptureContractError(f"{label} runtime stability receipt is malformed")
            if display_path in normalized:
                raise CaptureContractError(
                    f"duplicate {label} runtime path in stability receipt: {display_path}"
                )
            normalized[display_path] = entry[1:]
        return normalized

    expected = normalize(before)
    observed = normalize(after)
    for display_path, expected_state in expected.items():
        if observed.get(display_path) != expected_state:
            raise CaptureContractError(
                f"{label} runtime library identity/content changed during canonical observation: "
                f"{display_path}"
            )

    # Only entries absent from the pre-execution baseline need a wall-clock
    # freshness test.  Do not use mtime for that test: preserved package/build
    # timestamps may legitimately lie in the future.  ctime/change-time is retained
    # in the exact receipt and is the post-start mutation discriminator here.
    for display_path, observed_state in observed.items():
        if display_path in expected:
            continue
        ctime_ns = observed_state[6]
        if ctime_ns > observation_started_ns:
            raise CaptureContractError(
                f"{label} runtime library changed after canonical observation began: {display_path}"
            )


def _cuda_runtime_library_receipt(
    paths: Iterable[Path | _MappedLibrary],
) -> tuple[int, str]:
    """Hash actual mapped library bytes; relocation does not change the receipt."""
    count, receipt, _stability = _runtime_library_snapshot(
        paths,
        predicate=_is_cuda_runtime_library,
        label="CUDA",
        require_nonempty=True,
    )
    return count, receipt


def loaded_cuda_runtime_library_snapshot() -> tuple[
    dict[str, int | str], tuple[tuple[Any, ...], ...]
]:
    count, receipt, stability = _runtime_library_snapshot(
        _loaded_cuda_library_entries(),
        predicate=_is_cuda_runtime_library,
        label="CUDA",
        require_nonempty=True,
    )
    return (
        {
            "cuda_runtime_library_file_count": count,
            "cuda_runtime_library_receipt_sha256": receipt,
        },
        stability,
    )


def loaded_cuda_runtime_library_provenance() -> dict[str, int | str]:
    provenance, _stability = loaded_cuda_runtime_library_snapshot()
    return provenance


__all__ = ["loaded_cuda_runtime_library_provenance"]
