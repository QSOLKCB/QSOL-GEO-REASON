from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_section(text: str, start: str, end: str, replacement: str, label: str) -> str:
    start_at = text.find(start)
    if start_at < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end_at = text.find(end, start_at)
    if end_at < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start_at] + replacement + text[end_at:]


# 1. Make the authenticated-loader stage own patches before installation starts,
#    so interrupts cannot land in a return-value ownership gap.
rel = "src/qsol_geo_reason/capture_backend_round46.py"
text = read(rel)
install_start = "def _install_authenticated_loader_redirects(\n"
install_end = "\n\nclass HuggingFacePyTorchBackend(_Round45Backend):"
install_replacement = '''def _install_authenticated_loader_redirects(
    module: Any,
    *,
    model_source: Path,
    tokenizer_source: Path,
    model_stage: Path,
    tokenizer_stage: Path,
    patches: list[_LoaderPatch] | None = None,
) -> list[_LoaderPatch]:
    """Redirect exactly the two authenticated core loads to private staged bytes.

    When ``patches`` is supplied, ownership exists in the caller before the first
    process-global loader mutation. This removes the post-return handoff window in
    which an interrupt could otherwise strand installed redirects outside the vault.
    """
    installed = patches if patches is not None else []
    specifications = (
        ("AutoTokenizer", "tokenizer", tokenizer_source.resolve(), tokenizer_stage),
        ("AutoModelForCausalLM", "model", model_source.resolve(), model_stage),
    )
    try:
        for owner_name, label, expected_source, staged_path in specifications:
            owner = getattr(module, owner_name, None)
            original = getattr(owner, "from_pretrained", None) if owner is not None else None
            if owner is None or not callable(original):
                raise CaptureContractError(
                    f"canonical Transformers {owner_name}.from_pretrained is unavailable"
                )

            def redirected(
                cls: Any,
                pretrained_model_name_or_path: Any,
                *args: Any,
                _original: Any = original,
                _expected: Path = expected_source,
                _stage: Path = staged_path,
                _label: str = label,
                **kwargs: Any,
            ) -> Any:
                del cls
                supplied = _loader_argument_path(
                    pretrained_model_name_or_path, _label
                )
                if supplied != _expected:
                    raise CaptureContractError(
                        f"canonical Transformers {_label} loader path changed after authentication"
                    )
                if kwargs.get("local_files_only") is not True:
                    raise CaptureContractError(
                        f"canonical Transformers {_label} loader must remain local-files-only"
                    )
                if kwargs.get("trust_remote_code") is not False:
                    raise CaptureContractError(
                        f"canonical Transformers {_label} loader must keep trust_remote_code=false"
                    )
                return _original(str(_stage), *args, **kwargs)

            installed.append(_LoaderPatch(owner, redirected))
        return installed
    except BaseException:
        for patch in reversed(installed):
            patch.restore()
        installed.clear()
        raise


def _install_round46_stage(
    instance: Any,
    module: Any,
    *,
    tempdir: tempfile.TemporaryDirectory[str],
    model_source: Path,
    tokenizer_source: Path,
    model_stage: Path,
    tokenizer_stage: Path,
) -> None:
    """Install redirects with local ownership until the stage vault takes over."""
    stage = _AuthenticatedLoadStage(tempdir, [])
    try:
        _install_authenticated_loader_redirects(
            module,
            model_source=model_source,
            tokenizer_source=tokenizer_source,
            model_stage=model_stage,
            tokenizer_stage=tokenizer_stage,
            patches=stage.patches,
        )
        _remember_round46_stage(instance, stage)
    except BaseException:
        # The stage object existed before the first loader mutation, so it can
        # restore every installed redirect even if interruption occurs immediately
        # after installation returns or during the vault handoff itself.
        stage.cleanup()
        raise
'''
text = replace_section(text, install_start, install_end, install_replacement, "round46 install section")
old_handoff = '''            patches = _install_authenticated_loader_redirects(
                module,
                model_source=model_snapshot,
                tokenizer_source=tokenizer_snapshot,
                model_stage=model_stage,
                tokenizer_stage=tokenizer_stage,
            )
            _remember_round46_stage(
                self, _AuthenticatedLoadStage(tempdir, patches)
            )
        except Exception:
            tempdir.cleanup()
            raise
'''
new_handoff = '''            _install_round46_stage(
                self,
                module,
                tempdir=tempdir,
                model_source=model_snapshot,
                tokenizer_source=tokenizer_snapshot,
                model_stage=model_stage,
                tokenizer_stage=tokenizer_stage,
            )
        except BaseException:
            # Failures before stage installation still own only the temporary tree;
            # stage-install failures have already restored redirects and cleanup is
            # intentionally idempotent.
            tempdir.cleanup()
            raise
'''
text = replace_once(text, old_handoff, new_handoff, "round46 handoff")
write(rel, text)


# 2. Keep thread-patch restoration state reachable until every original callable is
#    restored. SIGINT/SystemExit are deferred until restoration succeeds.
rel = "src/qsol_geo_reason/capture_backend_production.py"
text = read(rel)
leave_start = "    def _leave_exclusive_python_thread_boundary(self) -> None:\n"
leave_end = "\n    def _assert_live_state_authentication(self) -> None:"
leave_replacement = '''    def _leave_exclusive_python_thread_boundary(self) -> None:
        state = getattr(self, "_exclusive_thread_boundary_state", None)
        if state is None:
            return
        lock = state["lock"]
        patches = state["patches"]
        interrupted: BaseException | None = None
        while True:
            try:
                with lock:
                    for owner, name, original in reversed(patches):
                        setattr(owner, name, original)
                    # Clear the receipt only after all originals are restored, while
                    # still inside the protected registry region. If an interrupt
                    # lands earlier, the same local/state receipt remains retryable.
                    self._exclusive_thread_boundary_state = None
                break
            except (KeyboardInterrupt, SystemExit) as exc:
                if interrupted is None:
                    interrupted = exc
                # Retry the complete idempotent restoration before propagating the
                # asynchronous interruption to the caller.
                continue
            except BaseException as exc:
                # Deterministic restoration failure leaves the state installed so a
                # later cleanup attempt can retry rather than losing ownership.
                raise CaptureContractError(
                    "unable to restore ambient Python thread-start policy after canonical observation"
                ) from exc
        if interrupted is not None:
            raise interrupted
'''
text = replace_section(text, leave_start, leave_end, leave_replacement, "thread leave section")
write(rel, text)


# 3. Linux runtime receipts must authenticate the inode that is actually mapped, not
#    a pathname that can be atomically replaced after /proc/self/maps is sampled.
rel = "src/qsol_geo_reason/capture_cuda_runtime.py"
text = read(rel)
text = replace_once(
    text,
    "import ctypes\nimport hashlib\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n",
    "import ctypes\nimport hashlib\nimport os\nimport re\nimport stat\nimport sys\nfrom dataclasses import dataclass\nfrom pathlib import Path\n",
    "cuda runtime imports",
)
text = replace_once(
    text,
    "_LibraryPredicate = Callable[[str], bool]\n\n\n",
    '''_LibraryPredicate = Callable[[str], bool]\n\n\n@dataclass(frozen=True)\nclass _MappedLibrary:\n    \"\"\"A loaded library plus the stable object used to read its mapped bytes.\"\"\"\n\n    path: Path\n    content_path: Path\n    device: int | None = None\n    inode: int | None = None\n\n\n''',
    "mapped library type",
)
linux_start = "def _linux_loaded_library_paths(\n"
linux_end = "\n\ndef _configure_windows_module_api"
linux_replacement = '''def _linux_loaded_library_mappings(
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
        map_file = Path("/proc/self/map_files") / mapping_range
        key = (device, inode, candidate)
        result.setdefault(
            key,
            _MappedLibrary(
                path=Path(candidate),
                content_path=map_file,
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
'''
text = replace_section(text, linux_start, linux_end, linux_replacement, "linux runtime mapping enumeration")
loaded_start = "def _loaded_cuda_library_paths() -> list[Path]:\n"
loaded_end = "\n\ndef _sha256_file"
loaded_replacement = '''def _loaded_cuda_library_entries() -> list[Path | _MappedLibrary]:
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
'''
text = replace_section(text, loaded_start, loaded_end, loaded_replacement, "loaded cuda entries")
hash_start = "def _sha256_file(path: Path) -> str:\n"
hash_end = "\n\ndef loaded_cuda_runtime_library_provenance()"
hash_replacement = '''def _sha256_file(path: Path) -> str:
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
'''
text = replace_section(text, hash_start, hash_end, hash_replacement, "runtime digest section")
text = replace_once(
    text,
    "    count, receipt = _cuda_runtime_library_receipt(_loaded_cuda_library_paths())\n",
    "    count, receipt = _cuda_runtime_library_receipt(_loaded_cuda_library_entries())\n",
    "cuda provenance uses mapped entries",
)
write(rel, text)


# CPU runtime provenance shares the same Linux mapped-object authentication.
rel = "src/qsol_geo_reason/capture_cpu_runtime.py"
cpu_text = '''"""Content receipts for external CPU math/runtime libraries used by capture."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_cuda_runtime import (
    _MappedLibrary,
    _darwin_loaded_library_paths,
    _linux_loaded_library_mappings,
    _runtime_library_digest,
    _windows_loaded_library_paths,
)


_LINUX_CPU_LIBRARY = re.compile(
    r"^(?:libmkl[^/]*|libdnnl[^/]*|libmkldnn[^/]*|libiomp5[^/]*|libgomp[^/]*|libomp[^/]*|libopenblas[^/]*|libblas[^/]*|libcblas[^/]*|liblapack[^/]*)[.]so(?:[.].*)?$",
    re.IGNORECASE,
)
_WINDOWS_CPU_LIBRARY = re.compile(
    r"^(?:mkl[^/]*|dnnl[^/]*|mkldnn[^/]*|libiomp5(?:md)?|libgomp[^/]*|libomp[^/]*|openblas[^/]*|libopenblas[^/]*|blas[^/]*|lapack[^/]*|vcomp[0-9]+)[.]dll$",
    re.IGNORECASE,
)
_DARWIN_CPU_LIBRARY = re.compile(
    r"^(?:libmkl[^/]*|libdnnl[^/]*|libmkldnn[^/]*|libiomp5[^/]*|libomp[^/]*|libopenblas[^/]*|libblas[^/]*|liblapack[^/]*)[.]dylib$|^(?:Accelerate|vecLib)$",
    re.IGNORECASE,
)


def _is_cpu_runtime_library(path: str) -> bool:
    name = Path(path).name
    return bool(
        _LINUX_CPU_LIBRARY.fullmatch(name)
        or _WINDOWS_CPU_LIBRARY.fullmatch(name)
        or _DARWIN_CPU_LIBRARY.fullmatch(name)
    )


def _loaded_cpu_library_entries() -> list[Path | _MappedLibrary]:
    if sys.platform.startswith("linux"):
        return list(_linux_loaded_library_mappings(_is_cpu_runtime_library))
    if os.name == "nt":
        return list(_windows_loaded_library_paths(_is_cpu_runtime_library))
    if sys.platform == "darwin":
        return list(_darwin_loaded_library_paths(_is_cpu_runtime_library))
    raise CaptureContractError(
        "canonical CPU observation cannot enumerate loaded CPU math/runtime shared objects on this platform"
    )


def _loaded_cpu_library_paths() -> list[Path]:
    """Compatibility view for callers that only need display pathnames."""
    entries = _loaded_cpu_library_entries()
    return [entry.path if isinstance(entry, _MappedLibrary) else Path(entry) for entry in entries]


def _cpu_runtime_library_receipt(
    paths: Iterable[Path | _MappedLibrary],
) -> tuple[int, str]:
    """Hash mapped external CPU math/runtime libraries by basename and content."""
    by_name: dict[str, str] = {}
    for raw_path in paths:
        item = _runtime_library_digest(
            raw_path,
            predicate=_is_cpu_runtime_library,
            label="CPU",
        )
        if item is None:
            continue
        name, digest = item
        prior = by_name.get(name)
        if prior is not None and prior != digest:
            raise CaptureContractError(
                f"multiple loaded CPU runtime libraries share basename {name!r} with different content"
            )
        by_name[name] = digest
    # A statically linked PyTorch build can legitimately have no external CPU
    # math library in this set. The empty-set receipt is still an explicit,
    # canonical statement that enumeration succeeded and found none.
    return len(by_name), sha256_json(dict(sorted(by_name.items())))


def loaded_cpu_runtime_library_provenance() -> dict[str, int | str]:
    count, receipt = _cpu_runtime_library_receipt(_loaded_cpu_library_entries())
    return {
        "cpu_runtime_library_file_count": count,
        "cpu_runtime_library_receipt_sha256": receipt,
    }


__all__ = ["loaded_cpu_runtime_library_provenance"]
'''
write(rel, cpu_text)


# Round-51 behavioral regressions.
test_text = '''"""Round 51 regressions for interrupt ownership and mapped runtime receipts."""
from __future__ import annotations

import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend_round46 as round46
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_cpu_runtime import _cpu_runtime_library_receipt
from qsol_geo_reason.capture_cuda_runtime import (
    _MappedLibrary,
    _cuda_runtime_library_receipt,
)
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound51RegressionTests(unittest.TestCase):
    def test_loader_stage_restores_if_interrupt_lands_during_vault_handoff(self):
        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class Instance:
            pass

        module = types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )
        tokenizer_descriptor = vars(AutoTokenizer)["from_pretrained"]
        model_descriptor = vars(AutoModelForCausalLM)["from_pretrained"]
        instance = Instance()
        real_remember = round46._remember_round46_stage

        def remember_then_interrupt(inst, stage):
            real_remember(inst, stage)
            raise KeyboardInterrupt

        tempdir = tempfile.TemporaryDirectory(prefix="round51-stage-")
        root = Path(tempdir.name)
        for name in ("model-source", "tokenizer-source", "model-stage", "tokenizer-stage"):
            (root / name).mkdir()
        try:
            with patch.object(
                round46,
                "_remember_round46_stage",
                side_effect=remember_then_interrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    round46._install_round46_stage(
                        instance,
                        module,
                        tempdir=tempdir,
                        model_source=root / "model-source",
                        tokenizer_source=root / "tokenizer-source",
                        model_stage=root / "model-stage",
                        tokenizer_stage=root / "tokenizer-stage",
                    )
            self.assertIs(vars(AutoTokenizer)["from_pretrained"], tokenizer_descriptor)
            self.assertIs(vars(AutoModelForCausalLM)["from_pretrained"], model_descriptor)
        finally:
            # The constructor's outer finally performs this second, idempotent vault
            # cleanup in production if the handoff stored the stage before interrupting.
            round46._finish_round46_stage(instance)
            tempdir.cleanup()

    def test_thread_restore_retries_interrupt_before_clearing_state(self):
        class Owner:
            def __init__(self):
                object.__setattr__(self, "interrupt_next_restore", False)
                object.__setattr__(self, "start", "blocked")

            def __setattr__(self, name, value):
                if name == "start" and self.interrupt_next_restore:
                    object.__setattr__(self, "interrupt_next_restore", False)
                    raise KeyboardInterrupt
                object.__setattr__(self, name, value)

        backend = object.__new__(ProductionBackend)
        owner = Owner()
        original = object()
        owner.interrupt_next_restore = True
        backend._exclusive_thread_boundary_state = {
            "lock": threading.RLock(),
            "patches": [(owner, "start", original)],
        }

        with self.assertRaises(KeyboardInterrupt):
            backend._leave_exclusive_python_thread_boundary()

        self.assertIs(owner.start, original)
        self.assertIsNone(backend._exclusive_thread_boundary_state)

    def test_linux_mapped_cuda_receipt_hashes_mapped_inode_not_replaced_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_path = root / "libcuda.so.1"
            mapped_handle = root / "mapped-handle"
            control_dir = root / "control"
            control_dir.mkdir()
            control = control_dir / "libcuda.so.1"

            live_path.write_bytes(b"mapped-old-bytes")
            os.link(live_path, mapped_handle)
            mapped_stat = mapped_handle.stat()
            control.write_bytes(b"mapped-old-bytes")

            replacement = root / "replacement"
            replacement.write_bytes(b"new-path-bytes")
            os.replace(replacement, live_path)

            mapped = _MappedLibrary(
                path=live_path,
                content_path=mapped_handle,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            count, mapped_receipt = _cuda_runtime_library_receipt([mapped])
            _, control_receipt = _cuda_runtime_library_receipt([control])
            _, pathname_receipt = _cuda_runtime_library_receipt([live_path])

            self.assertEqual(count, 1)
            self.assertEqual(mapped_receipt, control_receipt)
            self.assertNotEqual(mapped_receipt, pathname_receipt)

            wrong_identity = _MappedLibrary(
                path=live_path,
                content_path=live_path,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            with self.assertRaisesRegex(CaptureContractError, "identity changed"):
                _cuda_runtime_library_receipt([wrong_identity])

    def test_cpu_runtime_receipt_uses_same_mapped_identity_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_path = root / "libgomp.so.1"
            mapped_handle = root / "mapped-cpu-handle"
            control_dir = root / "control"
            control_dir.mkdir()
            control = control_dir / "libgomp.so.1"

            live_path.write_bytes(b"old-openmp")
            os.link(live_path, mapped_handle)
            mapped_stat = mapped_handle.stat()
            control.write_bytes(b"old-openmp")
            replacement = root / "replacement"
            replacement.write_bytes(b"new-openmp")
            os.replace(replacement, live_path)

            mapped = _MappedLibrary(
                path=live_path,
                content_path=mapped_handle,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            _, mapped_receipt = _cpu_runtime_library_receipt([mapped])
            _, control_receipt = _cpu_runtime_library_receipt([control])
            _, pathname_receipt = _cpu_runtime_library_receipt([live_path])
            self.assertEqual(mapped_receipt, control_receipt)
            self.assertNotEqual(mapped_receipt, pathname_receipt)


if __name__ == "__main__":
    unittest.main()
'''
write("tests/test_capture_codex_round51.py", test_text)
