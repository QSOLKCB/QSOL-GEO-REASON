"""Round-46 hardening for immutable loads and ambient CPU policy restoration.

The canonical loader now redirects the already source-authenticated Transformers
Auto* calls to a private byte-for-byte authenticated copy of the frozen Hub snapshot.
This closes the shared-cache ABA window without weakening the existing pre/post Hub
receipts. The same boundary also snapshots/restores the CPU denormal mode and derives
Linux NVIDIA driver identity from the loaded kernel module instead of PATH.
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import stat
import tempfile
import weakref
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .capture_backend_final import _recall_final_construction_preflight
from .capture_backend_round45 import HuggingFacePyTorchBackend as _Round45Backend
from .capture_common import CaptureContractError
from .capture_provenance import _is_canonical_snapshot_path
from . import capture_backend_production as _production


_CPU_FLUSH_DENORMAL_STATE_KEY = "cpu_flush_denormal"
_NVIDIA_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_NVIDIA_PROC_VERSION = re.compile(
    r"\bKernel Module\s+([0-9]+(?:\.[0-9]+)+)\b"
)


def _cpu_flush_denormal_state(torch: Any) -> bool | None:
    """Observe the current CPU FTZ/DAZ policy without mutating it.

    PyTorch exposes a setter but no getter. A multiplication from the smallest
    normal float32 into the subnormal range distinguishes gradual underflow from
    flush-to-zero on the current thread. Tiny software doubles that do not expose
    tensor/finfo return ``None`` and retain the older dependency-free test contract.
    """
    setter = getattr(torch, "set_flush_denormal", None)
    tensor = getattr(torch, "tensor", None)
    finfo = getattr(torch, "finfo", None)
    float32 = getattr(torch, "float32", None)
    if not callable(setter):
        return None
    if not callable(tensor) or not callable(finfo) or float32 is None:
        return None
    try:
        tiny = float(finfo(float32).tiny)
        normal = tensor([tiny], dtype=float32, device="cpu")
        probe = normal * 0.5
        value = float(probe.item())
    except Exception as exc:
        raise CaptureContractError(
            "unable to inspect ambient CPU flush-denormal policy"
        ) from exc
    if value == 0.0:
        return True
    if 0.0 < abs(value) < tiny:
        return False
    raise CaptureContractError(
        "CPU flush-denormal probe produced a noncanonical result"
    )


def _trusted_linux_nvidia_driver_version(
    *,
    sysfs_path: Path = Path("/sys/module/nvidia/version"),
    proc_path: Path = Path("/proc/driver/nvidia/version"),
) -> str | None:
    """Read the loaded NVIDIA kernel-module version without executable lookup."""
    if platform.system() != "Linux":
        return None

    try:
        value = sysfs_path.read_text(encoding="ascii", errors="strict").strip()
    except (OSError, UnicodeError):
        value = ""
    if _NVIDIA_VERSION.fullmatch(value):
        return value

    try:
        text = proc_path.read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError):
        return None
    match = _NVIDIA_PROC_VERSION.search(text)
    if match is None:
        return None
    value = match.group(1)
    return value if _NVIDIA_VERSION.fullmatch(value) else None


def _copy_authenticated_snapshot(
    source: Path,
    destination: Path,
    expected_hashes: Mapping[str, str],
    where: str,
) -> dict[str, str]:
    """Copy exactly the pre-authenticated bytes into a private regular-file tree.

    Each source file is opened through a file descriptor and the bytes actually
    copied are hashed before they become eligible for deserialization. Source cache
    mutation during the copy therefore either reproduces the authenticated bytes or
    fails closed. Destination files are created exclusively and are never symlinks.
    """
    if not expected_hashes:
        raise CaptureContractError(f"authenticated {where} snapshot is empty")
    try:
        destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to create private authenticated {where} snapshot"
        ) from exc

    observed: dict[str, str] = {}
    for rel, expected_digest in sorted(expected_hashes.items()):
        if not _is_canonical_snapshot_path(rel):
            raise CaptureContractError(
                f"authenticated {where} snapshot contains noncanonical path {rel!r}"
            )
        if not (
            isinstance(expected_digest, str)
            and len(expected_digest) == 64
            and all(character in "0123456789abcdef" for character in expected_digest)
        ):
            raise CaptureContractError(
                f"authenticated {where} snapshot contains invalid SHA-256 for {rel!r}"
            )

        parts = PurePosixPath(rel).parts
        source_path = source.joinpath(*parts)
        target_path = destination.joinpath(*parts)
        try:
            target_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            read_path = source_path.resolve(strict=True) if source_path.is_symlink() else source_path
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            source_fd = os.open(read_path, flags)
        except (OSError, RuntimeError) as exc:
            raise CaptureContractError(
                f"unable to open authenticated {where} snapshot artifact {rel}"
            ) from exc

        digest = hashlib.sha256()
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise CaptureContractError(
                    f"authenticated {where} snapshot artifact is not regular: {rel}"
                )
            write_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
            )
            target_fd = os.open(target_path, write_flags, 0o400)
            try:
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(target_fd, view)
                        if written <= 0:
                            raise OSError("short write while staging authenticated snapshot")
                        view = view[written:]
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
        except CaptureContractError:
            raise
        except OSError as exc:
            raise CaptureContractError(
                f"unable to copy authenticated {where} snapshot artifact {rel}"
            ) from exc
        finally:
            os.close(source_fd)

        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise CaptureContractError(
                f"{where} snapshot changed while creating the authenticated private load copy: {rel}"
            )
        observed[rel] = actual_digest

    if observed != dict(sorted(expected_hashes.items())):
        raise CaptureContractError(
            f"private authenticated {where} snapshot does not match frozen file receipts"
        )
    return observed


class _LoaderPatch:
    def __init__(self, owner: Any, replacement: Any):
        self.owner = owner
        namespace = vars(owner)
        self.had_own = "from_pretrained" in namespace
        self.original_descriptor = namespace.get("from_pretrained")
        setattr(owner, "from_pretrained", classmethod(replacement))

    def restore(self) -> None:
        if self.had_own:
            # Reassigning the original descriptor is idempotent if an interrupt
            # arrives immediately after setattr succeeds.
            setattr(self.owner, "from_pretrained", self.original_descriptor)
        else:
            # Likewise, retry after an interrupt that landed just after delattr.
            if "from_pretrained" not in vars(self.owner):
                return
            delattr(self.owner, "from_pretrained")


class _AuthenticatedLoadStage:
    def __init__(
        self,
        tempdir: tempfile.TemporaryDirectory[str],
        patches: list[_LoaderPatch],
    ):
        self.tempdir = tempdir
        self.patches = patches

    def cleanup(self) -> BaseException | None:
        """Restore all global loaders before releasing stage ownership.

        KeyboardInterrupt/SystemExit are deferred until restoration completes.
        Ordinary restoration failures leave the patch receipt intact so cleanup can
        be retried instead of losing ownership of a partially restored boundary.
        """
        interrupted: BaseException | None = None
        error: Exception | None = None

        for patch in reversed(tuple(self.patches)):
            while True:
                try:
                    patch.restore()
                except (KeyboardInterrupt, SystemExit) as exc:
                    if interrupted is None:
                        interrupted = exc
                    continue
                except Exception as exc:  # pragma: no cover - defensive aggregation
                    error = error or exc
                break

        if error is not None:
            if isinstance(error, CaptureContractError):
                raise error
            raise CaptureContractError(
                "unable to restore authenticated Transformers loader boundary"
            ) from error

        # Only discard the loader receipts after every redirect has been restored.
        self.patches.clear()
        while True:
            try:
                self.tempdir.cleanup()
            except (KeyboardInterrupt, SystemExit) as exc:
                if interrupted is None:
                    interrupted = exc
                continue
            except Exception as exc:  # pragma: no cover - platform cleanup failure
                raise CaptureContractError(
                    "unable to remove authenticated Transformers temporary snapshot"
                ) from exc
            break
        return interrupted


def _make_round46_stage_vault():
    active: weakref.WeakKeyDictionary[Any, _AuthenticatedLoadStage] = (
        weakref.WeakKeyDictionary()
    )
    completed: weakref.WeakSet[Any] = weakref.WeakSet()

    def is_completed(instance: Any) -> bool:
        return instance in completed

    def remember(instance: Any, stage: _AuthenticatedLoadStage) -> None:
        if instance in active:
            raise CaptureContractError("authenticated load stage was already installed")
        active[instance] = stage

    def finish(instance: Any) -> None:
        # Retain ownership in the active vault until every redirect is restored and
        # the private staged snapshot has been cleaned up successfully.
        stage = active.get(instance)
        if stage is None:
            return
        interrupted = stage.cleanup()
        active.pop(instance, None)
        completed.add(instance)
        if interrupted is not None:
            raise interrupted

    return is_completed, remember, finish


_round46_stage_completed, _remember_round46_stage, _finish_round46_stage = (
    _make_round46_stage_vault()
)
del _make_round46_stage_vault


def _loader_argument_path(value: Any, where: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise CaptureContractError(
            f"canonical Transformers {where} loader received a non-filesystem snapshot"
        )
    try:
        return Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CaptureContractError(
            f"canonical Transformers {where} loader snapshot is unavailable"
        ) from exc


def _install_authenticated_loader_redirects(
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


class HuggingFacePyTorchBackend(_Round45Backend):
    """Newest canonical boundary for Round-46 state and deserialization hardening."""

    def __init__(self, request: Mapping[str, Any]):
        try:
            super().__init__(request)
        finally:
            # The private files and temporary Auto* redirects exist only while the
            # inherited constructor is performing its two authenticated loads. This
            # finally path also restores them on partial initialization or interruption.
            _finish_round46_stage(self)

    @classmethod
    def _snapshot_torch_execution_policy_state(cls, torch: Any) -> dict[str, Any]:
        state = super()._snapshot_torch_execution_policy_state(torch)
        denormal = _cpu_flush_denormal_state(torch)
        if denormal is not None:
            state[_CPU_FLUSH_DENORMAL_STATE_KEY] = denormal
        return state

    @classmethod
    def _restore_torch_execution_policy_state(
        cls, torch: Any, state: Mapping[str, Any]
    ) -> None:
        # Restore denormal mode before the inherited exact-equality check dynamically
        # snapshots the complete Round-46 policy map.
        if _CPU_FLUSH_DENORMAL_STATE_KEY in state:
            expected = state[_CPU_FLUSH_DENORMAL_STATE_KEY]
            if type(expected) is not bool:
                raise CaptureContractError(
                    "ambient CPU flush-denormal policy snapshot is malformed"
                )
            setter = getattr(torch, "set_flush_denormal", None)
            if not callable(setter):
                raise CaptureContractError(
                    "canonical process isolation cannot restore CPU flush-denormal policy"
                )
            try:
                supported = setter(expected)
            except Exception as exc:
                raise CaptureContractError(
                    "unable to restore ambient CPU flush-denormal policy"
                ) from exc
            if supported is not True or _cpu_flush_denormal_state(torch) is not expected:
                raise CaptureContractError(
                    "unable to restore ambient CPU flush-denormal policy exactly"
                )
        super()._restore_torch_execution_policy_state(torch, state)

    @staticmethod
    def _nvidia_driver_version() -> str | None:
        # Never execute PATH-resolved nvidia-smi. Linux exposes the version of the
        # actually loaded NVIDIA kernel module directly through sysfs/procfs.
        return _trusted_linux_nvidia_driver_version()

    def _assert_autocast_disabled(self) -> None:
        super()._assert_autocast_disabled()
        if _round46_stage_completed(self):
            return
        preflight = _recall_final_construction_preflight(self)
        if preflight is None:
            return
        (
            _model_tree_receipt,
            _tokenizer_tree_receipt,
            model_snapshot,
            tokenizer_snapshot,
            model_before_items,
            tokenizer_before_items,
        ) = preflight

        tempdir = tempfile.TemporaryDirectory(prefix="qsol-geo-authenticated-load-")
        try:
            root = Path(tempdir.name)
            try:
                os.chmod(root, 0o700)
            except OSError:
                # TemporaryDirectory already requests a private directory; chmod is an
                # additional POSIX tightening and is not portable to every platform.
                pass
            model_stage = root / "model" / self._model_revision
            tokenizer_stage = root / "tokenizer" / self._tokenizer_revision
            _copy_authenticated_snapshot(
                model_snapshot,
                model_stage,
                dict(model_before_items),
                "model",
            )
            _copy_authenticated_snapshot(
                tokenizer_snapshot,
                tokenizer_stage,
                dict(tokenizer_before_items),
                "tokenizer",
            )
            module = getattr(self, "_transformers", None)
            if module is None:
                raise CaptureContractError(
                    "canonical authenticated load stage requires Transformers"
                )
            _install_round46_stage(
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


HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]
