"""Round-58 hardening for sealed source identity and transient runtime changes."""
from __future__ import annotations

import ctypes
import hashlib
import os
import stat
import struct
import subprocess
import types
import weakref
from pathlib import Path
from typing import Any

from .capture_backend_round56 import HuggingFacePyTorchBackend as _Round56Backend
from .capture_common import CaptureContractError
from . import capture_backend_round56 as _round56
from . import capture_cpu_runtime as _cpu_runtime
from . import capture_cuda_runtime as _cuda_runtime
from . import provenance as _provenance


# ---------------------------------------------------------------------------
# Sealed Git execution and loader-safe child environment
# ---------------------------------------------------------------------------

_DYNAMIC_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_DYNAMIC_LOADER_ENV_NAMES = frozenset(
    {
        "GLIBC_TUNABLES",
        "LIBPATH",
        "SHLIB_PATH",
    }
)


def _git_child_environment_round58() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("GIT_"):
            continue
        if upper.startswith(_DYNAMIC_LOADER_ENV_PREFIXES):
            continue
        if upper in _DYNAMIC_LOADER_ENV_NAMES:
            continue
        environment[key] = value
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def _make_git_runner_round58():
    trusted_git = _round56._trusted_git_executable
    trusted_run = subprocess.run

    def run(
        root: Path, *args: str, **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        git = trusted_git()
        completed = trusted_run(
            [str(git), "-C", str(root), *args],
            env=_git_child_environment_round58(),
            **kwargs,
        )
        trusted_git()
        return completed

    return run


_git_run_round58 = _make_git_runner_round58()
del _make_git_runner_round58


def _clone_provenance_function(
    value: types.FunctionType, private_globals: dict[str, Any]
) -> types.FunctionType:
    clone = types.FunctionType(
        value.__code__,
        private_globals,
        name=value.__name__,
        argdefs=value.__defaults__,
        closure=value.__closure__,
    )
    clone.__kwdefaults__ = value.__kwdefaults__
    clone.__annotations__ = dict(value.__annotations__)
    clone.__qualname__ = value.__qualname__
    clone.__doc__ = value.__doc__
    clone.__dict__.update(value.__dict__)
    return clone


def _make_sealed_provenance_entrypoints_round58():
    # Every function defined in provenance.py is cloned into one private globals
    # dictionary. The private call graph resolves _git_run only to the closure-bound
    # runner above, so later writes to provenance._git_run cannot affect OBSERVATION
    # source identity. Public provenance functions remain untouched for their normal
    # testable/noncanonical API surface.
    private_globals = dict(vars(_provenance))
    clones: dict[str, types.FunctionType] = {}
    for name, value in tuple(private_globals.items()):
        if isinstance(value, types.FunctionType):
            clones[name] = _clone_provenance_function(value, private_globals)
    private_globals.update(clones)
    private_globals["_git_run"] = _git_run_round58
    try:
        return clones["git_source_revision"], clones["resolve_implementation_revision"]
    except KeyError as exc:
        raise RuntimeError("unable to seal canonical provenance entrypoints") from exc


(
    _git_source_revision_round58,
    _resolve_implementation_revision_round58,
) = _make_sealed_provenance_entrypoints_round58()
del _make_sealed_provenance_entrypoints_round58

# Preserve the hardened runner for ordinary provenance callers too. Canonical
# capture does not trust this writable compatibility slot; capture_execute is bound
# to the private resolver at the end of this module.
_provenance._git_run = _git_run_round58


# ---------------------------------------------------------------------------
# Darwin mapped executable-content identity
# ---------------------------------------------------------------------------

_LC_SEGMENT = 0x1
_LC_SEGMENT_64 = 0x19
_VM_PROT_EXECUTE = 0x4
_MAX_MACHO_EXECUTABLE_BYTES = 1024 * 1024 * 1024


class _DarwinMappedLibraryRound58(_round56._DarwinMappedLibrary):
    __slots__ = ("mapped_executable_sha256",)

    def __init__(
        self, path: Path, mapped_uuid: str, mapped_executable_sha256: str
    ) -> None:
        super().__init__(path=path, mapped_uuid=mapped_uuid)
        self.mapped_executable_sha256 = mapped_executable_sha256


def _macho_uuid_and_executable_segments_round58(
    commands: bytes, *, endian: str, ncmds: int
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    uuid: str | None = None
    executable: list[tuple[int, int, int]] = []
    offset = 0
    for _index in range(ncmds):
        if offset + 8 > len(commands):
            raise CaptureContractError("Darwin image has truncated Mach-O load commands")
        command, command_size = struct.unpack_from(endian + "II", commands, offset)
        if command_size < 8 or offset + command_size > len(commands):
            raise CaptureContractError("Darwin image has malformed Mach-O load commands")
        if command == _round56._LC_UUID:
            if command_size < 24:
                raise CaptureContractError("Darwin image has malformed LC_UUID")
            observed_uuid = commands[offset + 8 : offset + 24].hex()
            if uuid is not None and uuid != observed_uuid:
                raise CaptureContractError("Darwin image contains conflicting LC_UUID commands")
            uuid = observed_uuid
        elif command == _LC_SEGMENT:
            if command_size < 56:
                raise CaptureContractError("Darwin image has malformed LC_SEGMENT")
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from(
                endian + "IIII", commands, offset + 24
            )
            initprot = struct.unpack_from(endian + "i", commands, offset + 44)[0]
            if initprot & _VM_PROT_EXECUTE and filesize:
                if filesize > vmsize:
                    raise CaptureContractError(
                        "Darwin executable segment file size exceeds mapped size"
                    )
                executable.append((int(vmaddr), int(fileoff), int(filesize)))
        elif command == _LC_SEGMENT_64:
            if command_size < 72:
                raise CaptureContractError("Darwin image has malformed LC_SEGMENT_64")
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from(
                endian + "QQQQ", commands, offset + 24
            )
            initprot = struct.unpack_from(endian + "i", commands, offset + 60)[0]
            if initprot & _VM_PROT_EXECUTE and filesize:
                if filesize > vmsize:
                    raise CaptureContractError(
                        "Darwin executable segment file size exceeds mapped size"
                    )
                executable.append((int(vmaddr), int(fileoff), int(filesize)))
        offset += command_size

    if uuid is None:
        raise CaptureContractError("loaded Darwin runtime image has no LC_UUID identity")
    if not executable:
        raise CaptureContractError("Darwin runtime image has no file-backed executable segment")
    total = sum(item[2] for item in executable)
    if total > _MAX_MACHO_EXECUTABLE_BYTES:
        raise CaptureContractError("Darwin executable-segment receipt exceeds canonical bound")
    return uuid, tuple(executable)


def _sha256_mapped_executable_segments_round58(
    slide: int, segments: tuple[tuple[int, int, int], ...]
) -> str:
    digest = hashlib.sha256()
    for index, (vmaddr, fileoff, size) in enumerate(segments):
        digest.update(struct.pack(">IQQQ", index, vmaddr, fileoff, size))
        address = slide + vmaddr
        if address <= 0:
            raise CaptureContractError("Darwin executable segment has invalid mapped address")
        remaining = size
        current = address
        while remaining:
            chunk_size = min(remaining, 1024 * 1024)
            try:
                chunk = ctypes.string_at(current, chunk_size)
            except (ValueError, OSError) as exc:
                raise CaptureContractError(
                    "unable to read mapped Darwin executable segment"
                ) from exc
            if len(chunk) != chunk_size:
                raise CaptureContractError("mapped Darwin executable segment is truncated")
            digest.update(chunk)
            current += chunk_size
            remaining -= chunk_size
    return digest.hexdigest()


def _mapped_macho_identity_round58(header_address: int, slide: int) -> tuple[str, str]:
    try:
        header_prefix = ctypes.string_at(header_address, 32)
    except (ValueError, OSError) as exc:
        raise CaptureContractError("unable to read mapped Darwin Mach-O header") from exc
    layout = _round56._MACHO_MAGICS.get(header_prefix[:4])
    if layout is None:
        raise CaptureContractError("loaded Darwin runtime image is not a supported Mach-O image")
    endian, header_size = layout
    ncmds = struct.unpack_from(endian + "I", header_prefix, 16)[0]
    sizeofcmds = struct.unpack_from(endian + "I", header_prefix, 20)[0]
    if ncmds > 8192 or sizeofcmds > 16 * 1024 * 1024 or sizeofcmds < ncmds * 8:
        raise CaptureContractError("mapped Darwin image has unreasonable Mach-O command bounds")
    try:
        commands = ctypes.string_at(header_address + header_size, sizeofcmds)
    except (ValueError, OSError) as exc:
        raise CaptureContractError("unable to read mapped Darwin Mach-O load commands") from exc
    uuid, segments = _macho_uuid_and_executable_segments_round58(
        commands, endian=endian, ncmds=ncmds
    )
    return uuid, _sha256_mapped_executable_segments_round58(slide, segments)


def _sha256_fd_executable_segments_round58(
    fd: int,
    *,
    base_offset: int,
    slice_size: int,
    segments: tuple[tuple[int, int, int], ...],
) -> str:
    digest = hashlib.sha256()
    for index, (vmaddr, fileoff, size) in enumerate(segments):
        if fileoff < 0 or size < 0 or fileoff + size > slice_size:
            raise CaptureContractError("Darwin executable segment escapes its Mach-O slice")
        digest.update(struct.pack(">IQQQ", index, vmaddr, fileoff, size))
        remaining = size
        offset = base_offset + fileoff
        while remaining:
            chunk_size = min(remaining, 1024 * 1024)
            try:
                chunk = os.pread(fd, chunk_size, offset)
            except OSError as exc:
                raise CaptureContractError(
                    "unable to read Darwin executable segment from runtime file"
                ) from exc
            if len(chunk) != chunk_size:
                raise CaptureContractError("Darwin runtime executable segment is truncated")
            digest.update(chunk)
            offset += chunk_size
            remaining -= chunk_size
    return digest.hexdigest()


def _thin_macho_identity_from_fd_round58(
    fd: int, base_offset: int, slice_size: int
) -> tuple[str, str] | None:
    try:
        header_prefix = os.pread(fd, 32, base_offset)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime Mach-O header") from exc
    if len(header_prefix) < 28:
        return None
    layout = _round56._MACHO_MAGICS.get(header_prefix[:4])
    if layout is None:
        return None
    endian, header_size = layout
    ncmds = struct.unpack_from(endian + "I", header_prefix, 16)[0]
    sizeofcmds = struct.unpack_from(endian + "I", header_prefix, 20)[0]
    if ncmds > 8192 or sizeofcmds > 16 * 1024 * 1024 or sizeofcmds < ncmds * 8:
        raise CaptureContractError("Darwin runtime file has unreasonable Mach-O command bounds")
    if header_size + sizeofcmds > slice_size:
        raise CaptureContractError("Darwin runtime Mach-O load commands escape their slice")
    try:
        commands = os.pread(fd, sizeofcmds, base_offset + header_size)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime Mach-O load commands") from exc
    if len(commands) != sizeofcmds:
        raise CaptureContractError("Darwin runtime file has truncated Mach-O load commands")
    uuid, segments = _macho_uuid_and_executable_segments_round58(
        commands, endian=endian, ncmds=ncmds
    )
    return uuid, _sha256_fd_executable_segments_round58(
        fd,
        base_offset=base_offset,
        slice_size=slice_size,
        segments=segments,
    )


def _macho_identities_from_fd_round58(fd: int) -> frozenset[tuple[str, str]]:
    try:
        info = os.fstat(fd)
        prefix = os.pread(fd, 8, 0)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime library identity") from exc
    file_size = int(info.st_size)
    if len(prefix) < 4:
        return frozenset()
    if prefix[:4] in _round56._MACHO_MAGICS:
        value = _thin_macho_identity_from_fd_round58(fd, 0, file_size)
        return frozenset((value,)) if value is not None else frozenset()

    fat_layout = _round56._FAT_MAGICS.get(prefix[:4])
    if fat_layout is None or len(prefix) < 8:
        return frozenset()
    endian, arch_size = fat_layout
    nfat = struct.unpack_from(endian + "I", prefix, 4)[0]
    if nfat < 1 or nfat > 128:
        raise CaptureContractError("Darwin universal runtime image has invalid architecture count")
    try:
        table = os.pread(fd, nfat * arch_size, 8)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin universal runtime architecture table") from exc
    if len(table) != nfat * arch_size:
        raise CaptureContractError("Darwin universal runtime architecture table is truncated")

    identities: set[tuple[str, str]] = set()
    for index in range(nfat):
        offset = index * arch_size
        if arch_size == 20:
            slice_offset, slice_size = struct.unpack_from(endian + "II", table, offset + 8)
        else:
            slice_offset, slice_size = struct.unpack_from(endian + "QQ", table, offset + 8)
        if slice_size < 28 or slice_offset + slice_size > file_size:
            raise CaptureContractError("Darwin universal runtime slice escapes its file")
        value = _thin_macho_identity_from_fd_round58(
            fd, int(slice_offset), int(slice_size)
        )
        if value is not None:
            identities.add(value)
    return frozenset(identities)


def _darwin_loaded_library_paths_round58(
    predicate: Any,
) -> list[Path | _DarwinMappedLibraryRound58]:
    try:
        process = ctypes.CDLL(None)
        image_count = getattr(process, "_dyld_image_count")
        image_name = getattr(process, "_dyld_get_image_name")
        image_header = getattr(process, "_dyld_get_image_header")
        image_slide = getattr(process, "_dyld_get_image_vmaddr_slide")
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        image_header.argtypes = [ctypes.c_uint32]
        image_header.restype = ctypes.c_void_p
        image_slide.argtypes = [ctypes.c_uint32]
        image_slide.restype = ctypes.c_long
    except Exception as exc:
        raise CaptureContractError("unable to access Darwin dyld image identity APIs") from exc

    result: dict[tuple[str, str, str], _DarwinMappedLibraryRound58] = {}
    count = int(image_count())
    if count > 65536:
        raise CaptureContractError("loaded Darwin image count exceeds canonical bound")
    for index in range(count):
        raw = image_name(index)
        header = image_header(index)
        if not raw or not header:
            continue
        candidate = os.fsdecode(raw)
        if not predicate(candidate):
            continue
        mapped_uuid, mapped_digest = _mapped_macho_identity_round58(
            int(header), int(image_slide(index))
        )
        key = (candidate, mapped_uuid, mapped_digest)
        result.setdefault(
            key,
            _DarwinMappedLibraryRound58(
                Path(candidate), mapped_uuid, mapped_digest
            ),
        )
    return sorted(
        result.values(),
        key=lambda item: (
            str(item.path),
            item.mapped_uuid,
            item.mapped_executable_sha256,
        ),
    )


_ROUND56_RUNTIME_LIBRARY_MEASUREMENT = _cuda_runtime._runtime_library_measurement


def _runtime_library_measurement_round58(
    raw_path: Any,
    *,
    predicate: Any,
    label: str,
) -> tuple[str, str, str, tuple[int, int, int, int, int]] | None:
    if isinstance(raw_path, _round56._DarwinMappedLibrary) and not isinstance(
        raw_path, _DarwinMappedLibraryRound58
    ):
        raise CaptureContractError(
            f"mapped {label} Darwin runtime image lacks a cryptographic mapped-code receipt"
        )
    if not isinstance(raw_path, _DarwinMappedLibraryRound58):
        return _ROUND56_RUNTIME_LIBRARY_MEASUREMENT(
            raw_path, predicate=predicate, label=label
        )

    display_path = raw_path.path
    if not predicate(str(display_path)):
        return None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(display_path, flags)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to open mapped {label} Darwin runtime library {display_path}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CaptureContractError(
                f"mapped {label} Darwin runtime object is not a regular file: {display_path}"
            )
        expected_identity = (
            raw_path.mapped_uuid,
            raw_path.mapped_executable_sha256,
        )
        if expected_identity not in _macho_identities_from_fd_round58(fd):
            raise CaptureContractError(
                f"mapped {label} Darwin runtime executable bytes no longer match their pathname: "
                f"{display_path}"
            )
        before_fingerprint = _cuda_runtime._stat_fingerprint(before)
        digest = _cuda_runtime._sha256_fd(fd, display_path)
        after_fingerprint = _cuda_runtime._stat_fingerprint(os.fstat(fd))
        if after_fingerprint != before_fingerprint:
            raise CaptureContractError(
                f"mapped {label} Darwin runtime library changed while hashing: {display_path}"
            )
        if expected_identity not in _macho_identities_from_fd_round58(fd):
            raise CaptureContractError(
                f"mapped {label} Darwin runtime executable identity changed while hashing: "
                f"{display_path}"
            )
    finally:
        os.close(fd)
    return str(display_path), display_path.name, digest, before_fingerprint


_cuda_runtime._darwin_loaded_library_paths = _darwin_loaded_library_paths_round58
_cpu_runtime._darwin_loaded_library_paths = _darwin_loaded_library_paths_round58
_cuda_runtime._runtime_library_measurement = _runtime_library_measurement_round58


# ---------------------------------------------------------------------------
# PyTorch package ABA/change-time stability receipt
# ---------------------------------------------------------------------------

_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _package_file_stat_fingerprint_round58(path: Path) -> tuple[int, ...]:
    try:
        link = os.lstat(path)
        target = os.stat(path)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to stat PyTorch package artifact for stability receipt: {path}"
        ) from exc
    if stat.S_ISDIR(target.st_mode):
        raise CaptureContractError(
            f"PyTorch package stability receipt expected a file, found directory: {path}"
        )
    return (
        int(link.st_mode),
        int(link.st_dev),
        int(link.st_ino),
        int(link.st_size),
        int(link.st_mtime_ns),
        int(link.st_ctime_ns),
        int(target.st_mode),
        int(target.st_dev),
        int(target.st_ino),
        int(target.st_size),
        int(target.st_mtime_ns),
        int(target.st_ctime_ns),
    )


def _torch_package_stability_receipt_round58(
    module: Any,
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise CaptureContractError(
            "canonical capture cannot locate PyTorch for package stability receipt"
        )
    try:
        root = Path(module_file).resolve(strict=True).parent
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CaptureContractError(
            "unable to enumerate PyTorch package for stability receipt"
        ) from exc
    if not root.is_dir():
        raise CaptureContractError("PyTorch package stability root is not a directory")

    entries: list[tuple[str, tuple[int, ...]]] = []
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise CaptureContractError(
                "PyTorch package stability artifact escapes its package root"
            ) from exc
        if path.is_dir():
            if path.is_symlink():
                raise CaptureContractError(
                    f"PyTorch package contains an unenumerated symlinked directory: {path}"
                )
            continue
        if path.suffix in _BYTECODE_SUFFIXES:
            continue
        rel = relative.as_posix()
        if (
            not rel
            or rel.startswith("/")
            or "\\" in rel
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise CaptureContractError(
                f"PyTorch package contains a noncanonical stability path: {rel!r}"
            )
        entries.append((rel, _package_file_stat_fingerprint_round58(path)))
    if not entries:
        raise CaptureContractError(
            "canonical capture found no PyTorch files for package stability receipt"
        )
    return str(root), tuple(entries)


def _make_torch_package_stability_vault_round58():
    baselines: weakref.WeakKeyDictionary[
        Any, tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]
    ] = weakref.WeakKeyDictionary()

    def remember(
        instance: Any,
        value: tuple[str, tuple[tuple[str, tuple[int, ...]], ...]],
    ) -> None:
        if instance in baselines:
            raise CaptureContractError(
                "PyTorch package stability baseline was already initialized"
            )
        baselines[instance] = value

    def recall(
        instance: Any,
    ) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]] | None:
        return baselines.get(instance)

    return remember, recall


(
    _remember_round58_torch_package_stability,
    _recall_round58_torch_package_stability,
) = _make_torch_package_stability_vault_round58()
del _make_torch_package_stability_vault_round58


_ROUND56_INITIALIZE_TORCH_PACKAGE_BASELINE = (
    _Round56Backend._initialize_round56_torch_package_baseline
)
_ROUND56_ASSERT_TORCH_PACKAGE_PROVENANCE = (
    _Round56Backend._assert_round56_torch_package_provenance
)


def _initialize_round58_torch_package_baseline(self: Any) -> None:
    _ROUND56_INITIALIZE_TORCH_PACKAGE_BASELINE(self)
    if self._real_round56_torch_runtime():
        _remember_round58_torch_package_stability(
            self, _torch_package_stability_receipt_round58(self._torch)
        )


def _assert_round58_torch_package_provenance(self: Any) -> None:
    _ROUND56_ASSERT_TORCH_PACKAGE_PROVENANCE(self)
    baseline = _recall_round58_torch_package_stability(self)
    if baseline is None:
        if self._real_round56_torch_runtime():
            raise CaptureContractError(
                "canonical PyTorch package stability baseline is missing"
            )
        return
    if _torch_package_stability_receipt_round58(self._torch) != baseline:
        raise CaptureContractError(
            "imported PyTorch package changed transiently after authenticated backend construction"
        )


# Harden the already-exported Round-56 class in place so every earlier import keeps
# the same class identity while construction/observation checks gain Round-58
# stability semantics before capture_execute freezes the trusted callable surface.
_Round56Backend._initialize_round56_torch_package_baseline = (
    _initialize_round58_torch_package_baseline
)
_Round56Backend._assert_round56_torch_package_provenance = (
    _assert_round58_torch_package_provenance
)

HuggingFacePyTorchBackend = _Round56Backend


# Import only after every backend patch above is installed. capture_execute then
# freezes the hardened adapter surface, after which its canonical source-identity
# global is rebound to the private resolver whose Git runner is closure-sealed.
from . import capture_execute as _capture_execute

_capture_execute.resolve_implementation_revision = _resolve_implementation_revision_round58


__all__ = ["HuggingFacePyTorchBackend"]
