"""Round-56 hardening for immutable trust roots and Darwin runtime identity.

This boundary is imported after every earlier production hardening layer and before
the public execute/verify modules.  It freezes the evidence taxonomy, replaces
PATH-sensitive host/source probes with trusted implementations, binds Darwin runtime
receipts to the actually mapped Mach-O image, re-authenticates the imported PyTorch
package around real observations, and prevents caller mutation of the hook-registry
allowlists from weakening the execution boundary.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import stat
import struct
import subprocess
import sys
import types
import weakref
from pathlib import Path
from typing import Any, Mapping

from .capture_backend_round46 import HuggingFacePyTorchBackend as _Round46Backend
from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance
from . import capture_backend_core as _core
from . import capture_backend_production as _production
from . import capture_common as _common
from . import capture_cpu_runtime as _cpu_runtime
from . import capture_cuda_runtime as _cuda_runtime
from . import capture_hardware as _hardware
from . import capture_provenance as _capture_provenance
from . import provenance as _provenance


# ---------------------------------------------------------------------------
# Evidence taxonomy
# ---------------------------------------------------------------------------

_CANONICAL_EVIDENCE_CLASSES = frozenset(("SIMULATION", "OBSERVATION"))

# execute_capture() and verify_capture_bundle() import this object only after this
# module has been loaded by capture.py.  A caller can no longer extend the accepted
# evidence taxonomy with set.add().
_common._ALLOWED_EVIDENCE = _CANONICAL_EVIDENCE_CLASSES

_ORIGINAL_VALIDATE_BACKEND_METADATA = _capture_provenance._validate_backend_metadata


def _validate_backend_metadata_round56(
    observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str
) -> None:
    # Do not let the older simulation fallback interpret an arbitrary non-observation
    # string as SIMULATION.  The public taxonomy has exactly two members.
    if evidence_class != "OBSERVATION" and evidence_class != "SIMULATION":
        raise CaptureContractError(
            f"unsupported capture evidence class: {evidence_class!r}"
        )
    _ORIGINAL_VALIDATE_BACKEND_METADATA(observed, request, evidence_class)


_capture_provenance._validate_backend_metadata = _validate_backend_metadata_round56


# ---------------------------------------------------------------------------
# Trusted Git executable
# ---------------------------------------------------------------------------


def _hash_executable(path: Path) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise _provenance.SourceIdentityError(
            f"unable to resolve trusted Git executable {path}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise _provenance.SourceIdentityError(
            f"trusted Git executable is not an executable regular file: {resolved}"
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _provenance.SourceIdentityError(
            f"unable to hash trusted Git executable {resolved}"
        ) from exc
    return resolved, digest.hexdigest()


def _windows_git_candidates() -> tuple[Path, ...]:
    if os.name != "nt":
        return ()
    candidates: list[Path] = []
    try:
        import winreg

        access_modes = [
            winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
            winreg.KEY_READ | getattr(winreg, "KEY_WOW64_32KEY", 0),
        ]
        for access in access_modes:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\GitForWindows",
                    0,
                    access,
                ) as key:
                    install_path, kind = winreg.QueryValueEx(key, "InstallPath")
            except OSError:
                continue
            if kind not in {
                getattr(winreg, "REG_SZ", object()),
                getattr(winreg, "REG_EXPAND_SZ", object()),
            }:
                continue
            if isinstance(install_path, str) and install_path.strip():
                root = Path(install_path)
                candidates.extend((root / "cmd" / "git.exe", root / "bin" / "git.exe"))
    except (ImportError, AttributeError):
        pass
    # Standard machine-wide Git for Windows location.  This is deliberately not
    # discovered from PATH or a caller-controlled per-user environment variable.
    candidates.append(Path(r"C:\Program Files\Git\cmd\git.exe"))
    return tuple(candidates)


def _system_git_candidates() -> tuple[Path, ...]:
    if os.name == "nt":
        return _windows_git_candidates()
    # System-managed locations only. /run/current-system is the machine-owned NixOS
    # system profile; resolving the symlink below binds its immutable store executable.
    return (
        Path("/usr/bin/git"),
        Path("/bin/git"),
        Path("/run/current-system/sw/bin/git"),
    )


def _make_trusted_git_resolver():
    baseline: tuple[Path, str] | None = None

    def resolve() -> Path:
        nonlocal baseline
        if baseline is None:
            last_error: BaseException | None = None
            for candidate in _system_git_candidates():
                try:
                    baseline = _hash_executable(candidate)
                    break
                except _provenance.SourceIdentityError as exc:
                    last_error = exc
            if baseline is None:
                raise _provenance.SourceIdentityError(
                    "canonical source identity requires a trusted system Git executable"
                ) from last_error
        path, expected_digest = baseline
        observed_path, observed_digest = _hash_executable(path)
        if observed_path != path or observed_digest != expected_digest:
            raise _provenance.SourceIdentityError(
                "trusted Git executable changed after source-identity initialization"
            )
        return path

    return resolve


_trusted_git_executable = _make_trusted_git_resolver()
del _make_trusted_git_resolver


def _git_run_round56(
    root: Path, *args: str, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    git = _trusted_git_executable()
    # Remove Git-specific caller overrides before invoking the absolute executable.
    # Repository-local config remains visible because source verification already
    # authenticates raw working-tree bytes independently of clean filters.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    completed = subprocess.run(
        [str(git), "-C", str(root), *args],
        env=environment,
        **kwargs,
    )
    # A machine-level executable change is itself a trust failure.  Recheck after
    # every command so source identity is never accepted across an executable swap.
    _trusted_git_executable()
    return completed


_provenance._git_run = _git_run_round56


# ---------------------------------------------------------------------------
# Native macOS sysctlbyname hardware identity
# ---------------------------------------------------------------------------


def _native_darwin_sysctl_text(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    if not isinstance(name, str) or not name or not name.isascii():
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        sysctlbyname = getattr(libc, "sysctlbyname")
        sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        sysctlbyname.restype = ctypes.c_int
        encoded = name.encode("ascii", errors="strict")
        size = ctypes.c_size_t(0)
        if sysctlbyname(encoded, None, ctypes.byref(size), None, 0) != 0:
            return None
        if size.value <= 1 or size.value > 1024 * 1024:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if sysctlbyname(
            encoded,
            ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.byref(size),
            None,
            0,
        ) != 0:
            return None
        raw = bytes(buffer.raw[: size.value]).rstrip(b"\x00")
        value = raw.decode("utf-8", errors="strict").strip()
    except (AttributeError, OSError, UnicodeError, ValueError):
        return None
    return value or None


def _sysctl_cpu_model_round56() -> str | None:
    if platform.system() != "Darwin":
        return None
    for name in ("machdep.cpu.brand_string", "hw.model"):
        value = _native_darwin_sysctl_text(name)
        if _hardware._is_concrete_cpu_identity(value):
            assert value is not None
            return value.strip()
    return None


_hardware._sysctl_cpu_model = _sysctl_cpu_model_round56
# capture_backend_core imported the earlier helper by name. Replace the module global
# it resolves at metadata time as well, so MPS hardware provenance never calls PATH
# resolved `sysctl`.
_core._sysctl_value = _native_darwin_sysctl_text


# ---------------------------------------------------------------------------
# Darwin mapped Mach-O identity
# ---------------------------------------------------------------------------

_LC_UUID = 0x1B
_MACHO_MAGICS: dict[bytes, tuple[str, int]] = {
    b"\xce\xfa\xed\xfe": ("<", 28),
    b"\xcf\xfa\xed\xfe": ("<", 32),
    b"\xfe\xed\xfa\xce": (">", 28),
    b"\xfe\xed\xfa\xcf": (">", 32),
}
_FAT_MAGICS: dict[bytes, tuple[str, int]] = {
    b"\xca\xfe\xba\xbe": (">", 20),
    b"\xca\xfe\xba\xbf": (">", 32),
    b"\xbe\xba\xfe\xca": ("<", 20),
    b"\xbf\xba\xfe\xca": ("<", 32),
}


class _DarwinMappedLibrary(_cuda_runtime._MappedLibrary):
    __slots__ = ("mapped_uuid",)

    def __init__(self, path: Path, mapped_uuid: str) -> None:
        super().__init__(path=path, content_path=path, device=None, inode=None)
        self.mapped_uuid = mapped_uuid


def _uuid_from_load_commands(
    commands: bytes, *, endian: str, ncmds: int
) -> str | None:
    offset = 0
    for _index in range(ncmds):
        if offset + 8 > len(commands):
            raise CaptureContractError("mapped Darwin image has truncated Mach-O load commands")
        command, command_size = struct.unpack_from(endian + "II", commands, offset)
        if command_size < 8 or offset + command_size > len(commands):
            raise CaptureContractError("mapped Darwin image has malformed Mach-O load commands")
        if command == _LC_UUID:
            if command_size < 24:
                raise CaptureContractError("mapped Darwin image has malformed LC_UUID")
            return commands[offset + 8 : offset + 24].hex()
        offset += command_size
    return None


def _mapped_macho_uuid(header_address: int) -> str:
    try:
        header_prefix = ctypes.string_at(header_address, 32)
    except (ValueError, OSError) as exc:
        raise CaptureContractError("unable to read mapped Darwin Mach-O header") from exc
    layout = _MACHO_MAGICS.get(header_prefix[:4])
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
    value = _uuid_from_load_commands(commands, endian=endian, ncmds=ncmds)
    if value is None:
        raise CaptureContractError("loaded Darwin runtime image has no LC_UUID identity")
    return value


def _thin_macho_uuid_from_fd(fd: int, base_offset: int) -> str | None:
    try:
        header_prefix = os.pread(fd, 32, base_offset)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime Mach-O header") from exc
    if len(header_prefix) < 28:
        return None
    layout = _MACHO_MAGICS.get(header_prefix[:4])
    if layout is None:
        return None
    endian, header_size = layout
    ncmds = struct.unpack_from(endian + "I", header_prefix, 16)[0]
    sizeofcmds = struct.unpack_from(endian + "I", header_prefix, 20)[0]
    if ncmds > 8192 or sizeofcmds > 16 * 1024 * 1024 or sizeofcmds < ncmds * 8:
        raise CaptureContractError("Darwin runtime file has unreasonable Mach-O command bounds")
    try:
        commands = os.pread(fd, sizeofcmds, base_offset + header_size)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime Mach-O load commands") from exc
    if len(commands) != sizeofcmds:
        raise CaptureContractError("Darwin runtime file has truncated Mach-O load commands")
    return _uuid_from_load_commands(commands, endian=endian, ncmds=ncmds)


def _macho_uuids_from_fd(fd: int) -> frozenset[str]:
    try:
        prefix = os.pread(fd, 8, 0)
    except OSError as exc:
        raise CaptureContractError("unable to read Darwin runtime library identity") from exc
    if len(prefix) < 4:
        return frozenset()
    if prefix[:4] in _MACHO_MAGICS:
        value = _thin_macho_uuid_from_fd(fd, 0)
        return frozenset((value,)) if value is not None else frozenset()

    fat_layout = _FAT_MAGICS.get(prefix[:4])
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
    values: set[str] = set()
    for index in range(nfat):
        offset = index * arch_size
        if arch_size == 20:
            slice_offset = struct.unpack_from(endian + "I", table, offset + 8)[0]
        else:
            slice_offset = struct.unpack_from(endian + "Q", table, offset + 8)[0]
        value = _thin_macho_uuid_from_fd(fd, int(slice_offset))
        if value is not None:
            values.add(value)
    return frozenset(values)


def _darwin_loaded_library_paths_round56(
    predicate: Any,
) -> list[Path | _DarwinMappedLibrary]:
    try:
        process = ctypes.CDLL(None)
        image_count = getattr(process, "_dyld_image_count")
        image_name = getattr(process, "_dyld_get_image_name")
        image_header = getattr(process, "_dyld_get_image_header")
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        image_header.argtypes = [ctypes.c_uint32]
        image_header.restype = ctypes.c_void_p
    except Exception as exc:
        raise CaptureContractError("unable to access Darwin dyld image identity APIs") from exc

    result: dict[tuple[str, str], _DarwinMappedLibrary] = {}
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
        mapped_uuid = _mapped_macho_uuid(int(header))
        key = (candidate, mapped_uuid)
        result.setdefault(key, _DarwinMappedLibrary(Path(candidate), mapped_uuid))
    return sorted(result.values(), key=lambda item: (str(item.path), item.mapped_uuid))


_ORIGINAL_RUNTIME_LIBRARY_MEASUREMENT = _cuda_runtime._runtime_library_measurement


def _runtime_library_measurement_round56(
    raw_path: Any,
    *,
    predicate: Any,
    label: str,
) -> tuple[str, str, str, tuple[int, int, int, int, int]] | None:
    if not isinstance(raw_path, _DarwinMappedLibrary):
        return _ORIGINAL_RUNTIME_LIBRARY_MEASUREMENT(
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
        file_uuids = _macho_uuids_from_fd(fd)
        if raw_path.mapped_uuid not in file_uuids:
            raise CaptureContractError(
                f"mapped {label} Darwin runtime image no longer matches its pathname: "
                f"{display_path}"
            )
        before_fingerprint = _cuda_runtime._stat_fingerprint(before)
        digest = _cuda_runtime._sha256_fd(fd, display_path)
        after_fingerprint = _cuda_runtime._stat_fingerprint(os.fstat(fd))
        if after_fingerprint != before_fingerprint:
            raise CaptureContractError(
                f"mapped {label} Darwin runtime library changed while hashing: {display_path}"
            )
        # Re-read the identity from the same descriptor after hashing.  This catches
        # in-place changes that alter the Mach-O identity even if a coarse filesystem
        # timestamp is preserved by the writer.
        if raw_path.mapped_uuid not in _macho_uuids_from_fd(fd):
            raise CaptureContractError(
                f"mapped {label} Darwin runtime library identity changed while hashing: "
                f"{display_path}"
            )
    finally:
        os.close(fd)
    return str(display_path), display_path.name, digest, before_fingerprint


_cuda_runtime._darwin_loaded_library_paths = _darwin_loaded_library_paths_round56
_cpu_runtime._darwin_loaded_library_paths = _darwin_loaded_library_paths_round56
_cuda_runtime._runtime_library_measurement = _runtime_library_measurement_round56


# ---------------------------------------------------------------------------
# PyTorch package re-authentication and hook-registry trust anchors
# ---------------------------------------------------------------------------


def _make_torch_package_baseline_vault():
    baselines: weakref.WeakKeyDictionary[Any, tuple[int, str]] = weakref.WeakKeyDictionary()

    def remember(instance: Any, value: tuple[int, str]) -> None:
        if instance in baselines:
            raise CaptureContractError(
                "PyTorch package provenance baseline was already initialized"
            )
        baselines[instance] = value

    def recall(instance: Any) -> tuple[int, str] | None:
        return baselines.get(instance)

    return remember, recall


(
    _remember_round56_torch_package_baseline,
    _recall_round56_torch_package_baseline,
) = _make_torch_package_baseline_vault()
del _make_torch_package_baseline_vault


def _trusted_hook_registry_names(
    _module_registries: tuple[str, ...] = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
    ),
    _global_registries: tuple[str, ...] = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_forward_pre_hooks_with_kwargs",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_hooks_always_called",
    ),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Immutable defaults live on a source-authenticated callable rather than a
    # caller-writable backend class attribute.
    return _module_registries, _global_registries


class HuggingFacePyTorchBackend(_Round46Backend):
    """Canonical Round-56 backend with post-load package and hook trust binding."""

    def __init__(self, request: Mapping[str, Any]):
        super().__init__(request)
        self._initialize_round56_torch_package_baseline()

    def _real_round56_torch_runtime(self) -> bool:
        torch = getattr(self, "_torch", None)
        return (
            isinstance(torch, types.ModuleType)
            and isinstance(getattr(torch, "__name__", None), str)
            and (
                torch.__name__ == "torch"
                or torch.__name__.startswith("torch.")
            )
        )

    def _observe_round56_torch_package(self) -> tuple[int, str]:
        torch = getattr(self, "_torch", None)
        observed = _python_package_provenance(torch, "PyTorch")
        count = observed.get("file_count")
        receipt = observed.get("receipt_sha256")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CaptureContractError("live PyTorch package file count is malformed")
        if (
            not isinstance(receipt, str)
            or len(receipt) != 64
            or any(character not in "0123456789abcdef" for character in receipt)
        ):
            raise CaptureContractError("live PyTorch package receipt is malformed")
        return count, receipt

    def _initialize_round56_torch_package_baseline(self) -> None:
        if not self._real_round56_torch_runtime():
            return
        build = getattr(self, "_torch_build_provenance", None)
        if not isinstance(build, Mapping):
            raise CaptureContractError(
                "canonical PyTorch build provenance is missing after model loading"
            )
        expected = (
            build.get("torch_package_file_count"),
            build.get("torch_package_receipt_sha256"),
        )
        observed = self._observe_round56_torch_package()
        if expected != observed:
            raise CaptureContractError(
                "imported PyTorch package changed during authenticated model loading"
            )
        _remember_round56_torch_package_baseline(self, observed)

    def _assert_round56_torch_package_provenance(self) -> None:
        baseline = _recall_round56_torch_package_baseline(self)
        if baseline is None:
            if self._real_round56_torch_runtime():
                raise CaptureContractError(
                    "canonical PyTorch package baseline is missing"
                )
            return
        if self._observe_round56_torch_package() != baseline:
            raise CaptureContractError(
                "imported PyTorch package changed after authenticated backend construction"
            )

    def _assert_round56_hook_registry_binding(self) -> None:
        module_registries, global_registries = _trusted_hook_registry_names()
        if (
            tuple(getattr(type(self), "_EXECUTION_HOOK_REGISTRIES", ()))
            != module_registries
            or tuple(getattr(type(self), "_GLOBAL_EXECUTION_HOOK_REGISTRIES", ()))
            != global_registries
            or tuple(getattr(self, "_EXECUTION_HOOK_REGISTRIES", ()))
            != module_registries
            or tuple(getattr(self, "_GLOBAL_EXECUTION_HOOK_REGISTRIES", ()))
            != global_registries
        ):
            raise CaptureContractError(
                "canonical PyTorch hook-registry allowlist changed after import"
            )

    def _assert_no_global_module_hooks(self) -> None:
        self._assert_round56_hook_registry_binding()
        _module_registries, global_registries = _trusted_hook_registry_names()
        torch = getattr(self, "_torch", None)
        if torch is None:
            return
        nn = getattr(torch, "nn", None)
        if nn is None:
            # Preserve the established dependency-free source-audit fixture lane.
            return
        module_api = getattr(getattr(nn, "modules", None), "module", None)
        if module_api is None:
            raise CaptureContractError(
                "canonical OBSERVATION requires access to PyTorch global module-hook registries"
            )
        found_registry = False
        for registry_name in global_registries:
            registry = getattr(module_api, registry_name, None)
            if registry is None:
                continue
            found_registry = True
            try:
                populated = bool(registry)
            except Exception as exc:
                raise CaptureContractError(
                    f"unable to authenticate process-global hook registry {registry_name}"
                ) from exc
            if populated:
                raise CaptureContractError(
                    "canonical OBSERVATION forbids process-global PyTorch execution hooks; "
                    f"registry={registry_name!r}"
                )
        if not found_registry:
            raise CaptureContractError(
                "canonical OBSERVATION cannot authenticate PyTorch global module-hook state"
            )

    def _assert_no_registered_module_hooks(self) -> None:
        # Bind both registry-name sets before delegating to the inherited complete
        # graph audit, which also rejects compiled call implementations and other
        # per-module execution substitutions.
        self._assert_round56_hook_registry_binding()
        super()._assert_no_registered_module_hooks()

    def _begin_runtime_library_stability_window(self) -> None:
        # production.begin_observation() calls this only after the exclusive Python
        # thread boundary and ambient-state receipt are active.  A failure therefore
        # follows the established retryable restoration path.
        self._assert_round56_torch_package_provenance()
        super()._begin_runtime_library_stability_window()

    def metadata(self) -> Mapping[str, Any]:
        # This is the post-execution package check.  It detects package updates and
        # model-specific lazy imports from changed bytes before a manifest can be
        # published with the construction-time receipt.
        self._assert_round56_torch_package_provenance()
        return super().metadata()


HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]
