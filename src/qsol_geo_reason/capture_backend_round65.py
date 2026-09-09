"""Round-65 construction and deterministic-policy trust hardening.

This layer closes construction/runtime trust gaps without rewriting the older
source-audited backend bodies. It reasserts a closure-bound snapshot verifier before
any inherited construction can trust Hub bytes, retains a pre-load stat/change-time
receipt for Transformers across model loading, and binds PyTorch deterministic-policy
callables after the pristine import boundary has established the real runtime.
"""
from __future__ import annotations

import importlib.util
import os
import stat
import types
import weakref
from pathlib import Path
from typing import Any, Mapping

from . import capture_backend_core as _core
from . import capture_backend_final as _final
from . import capture_backend_round64 as _round64
from . import capture_snapshot as _snapshot
from .capture_backend_round64 import HuggingFacePyTorchBackend as _Round64Backend
from .capture_common import CaptureContractError


_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
_DETERMINISM_CALLABLE_NAMES = (
    "use_deterministic_algorithms",
    "are_deterministic_algorithms_enabled",
    "is_deterministic_algorithms_warn_only_enabled",
)


def _make_snapshot_verifier_reassertion_round65(
    trusted_snapshot_verifier: Any,
    core_module: Any,
    final_module: Any,
):
    """Return a closure that restores the trusted snapshot verifier bindings."""
    if not callable(trusted_snapshot_verifier):
        raise RuntimeError("canonical snapshot verifier is not callable at Round-65 import")

    def reassert() -> None:
        core_module.__dict__["_snapshot_file_hashes"] = trusted_snapshot_verifier
        final_module.__dict__["_snapshot_file_hashes"] = trusted_snapshot_verifier
        if (
            core_module.__dict__.get("_snapshot_file_hashes") is not trusted_snapshot_verifier
            or final_module.__dict__.get("_snapshot_file_hashes") is not trusted_snapshot_verifier
        ):
            raise CaptureContractError(
                "canonical snapshot verifier binding could not be restored before construction"
            )

    return reassert


_reassert_snapshot_verifier_bindings_round65 = _make_snapshot_verifier_reassertion_round65(
    _snapshot._snapshot_file_hashes,
    _core,
    _final,
)
del _make_snapshot_verifier_reassertion_round65
_reassert_snapshot_verifier_bindings_round65()


def _package_file_stat_fingerprint_round65(
    path: Path, label: str
) -> tuple[int, ...]:
    try:
        link = os.lstat(path)
        target = os.stat(path)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to stat pre-load {label} package artifact: {path}"
        ) from exc
    if stat.S_ISDIR(target.st_mode):
        raise CaptureContractError(
            f"pre-load {label} package stability expected a file, found directory: {path}"
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


def _package_stability_from_root_round65(
    root: Path, label: str = "package"
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    try:
        canonical_root = root.resolve(strict=True)
        paths = sorted(canonical_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CaptureContractError(
            f"unable to enumerate {label} package before authenticated model loading"
        ) from exc
    if not canonical_root.is_dir():
        raise CaptureContractError(f"pre-load {label} package root is not a directory")

    entries: list[tuple[str, tuple[int, ...]]] = []
    for path in paths:
        try:
            relative = path.relative_to(canonical_root)
        except ValueError as exc:
            raise CaptureContractError(
                f"pre-load {label} package artifact escapes its package root"
            ) from exc
        if path.is_dir():
            if path.is_symlink():
                raise CaptureContractError(
                    f"{label} package contains an unenumerated symlinked directory: {path}"
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
                f"{label} package contains a noncanonical pre-load stability path: {rel!r}"
            )
        entries.append((rel, _package_file_stat_fingerprint_round65(path, label)))
    if not entries:
        raise CaptureContractError(
            f"canonical capture found no {label} files for pre-load stability receipt"
        )
    return str(canonical_root), tuple(entries)


def _preimport_package_stability_round65(
    package_name: str, label: str
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]] | None:
    """Fingerprint an importable package without executing package code."""
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not isinstance(spec.origin, str) or not spec.origin.strip():
        return None
    try:
        origin = Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to resolve {label} package before authenticated model loading"
        ) from exc
    return _package_stability_from_root_round65(origin.parent, label)


def _loaded_package_stability_round65(
    module: Any, label: str
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise CaptureContractError(
            f"canonical capture cannot locate loaded {label} for pre-load stability verification"
        )
    try:
        origin = Path(module_file).resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to resolve loaded {label} after authenticated model loading"
        ) from exc
    return _package_stability_from_root_round65(origin.parent, label)


def _make_round65_policy_vault():
    baselines: weakref.WeakKeyDictionary[Any, tuple[Any, Any, Any]] = (
        weakref.WeakKeyDictionary()
    )
    constructing: weakref.WeakSet[Any] = weakref.WeakSet()

    def enter(instance: Any) -> None:
        constructing.add(instance)

    def leave(instance: Any) -> None:
        constructing.discard(instance)

    def is_constructing(instance: Any) -> bool:
        return instance in constructing

    def remember(instance: Any, value: tuple[Any, Any, Any]) -> None:
        if instance in baselines:
            raise CaptureContractError(
                "deterministic-policy callable baseline was already initialized"
            )
        baselines[instance] = value

    def recall(instance: Any) -> tuple[Any, Any, Any] | None:
        return baselines.get(instance)

    return enter, leave, is_constructing, remember, recall


(
    _enter_round65_construction,
    _leave_round65_construction,
    _is_round65_constructing,
    _remember_round65_determinism_callables,
    _recall_round65_determinism_callables,
) = _make_round65_policy_vault()
del _make_round65_policy_vault


def _determinism_callables_round65(
    torch_module: Any,
    _names: tuple[str, str, str] = _DETERMINISM_CALLABLE_NAMES,
) -> tuple[Any, Any, Any]:
    values = tuple(getattr(torch_module, name, None) for name in _names)
    missing = [name for name, value in zip(_names, values) if not callable(value)]
    if missing:
        raise CaptureContractError(
            "canonical deterministic-policy callable is unavailable: " + ", ".join(missing)
        )
    return values  # type: ignore[return-value]


def _real_transformers_runtime_round65(instance: Any) -> bool:
    module = getattr(instance, "_transformers", None)
    return (
        isinstance(module, types.ModuleType)
        and isinstance(getattr(module, "__name__", None), str)
        and (module.__name__ == "transformers" or module.__name__.startswith("transformers."))
    )


def _make_round65_constructor(
    original_init: Any,
    reassert_snapshot_verifiers: Any,
    preimport_stability: Any,
    loaded_stability: Any,
    real_transformers_runtime: Any,
    enter_construction: Any,
    leave_construction: Any,
    remember_policy: Any,
    policy_callables: Any,
):
    """Closure-bind every new construction-time trust dependency."""

    def __init__(self: Any, request: Mapping[str, Any]) -> None:
        transformers_before = preimport_stability("transformers", "Transformers")
        enter_construction(self)
        try:
            reassert_snapshot_verifiers()
            original_init(self, request)
            reassert_snapshot_verifiers()

            if real_transformers_runtime(self):
                if transformers_before is None:
                    raise CaptureContractError(
                        "canonical OBSERVATION could not establish a pre-load Transformers package baseline"
                    )
                transformers_after = loaded_stability(self._transformers, "Transformers")
                if transformers_after != transformers_before:
                    raise CaptureContractError(
                        "imported Transformers package changed transiently during authenticated model loading"
                    )

            torch_module = getattr(self, "_torch", None)
            if isinstance(torch_module, types.ModuleType) and (
                torch_module.__name__ == "torch" or torch_module.__name__.startswith("torch.")
            ):
                remember_policy(self, policy_callables(torch_module))
        finally:
            leave_construction(self)

    __init__.__name__ = "__init__"
    __init__.__qualname__ = f"{_Round64Backend.__qualname__}.__init__"
    __init__.__module__ = _round64.__name__
    __init__.__doc__ = original_init.__doc__
    return __init__


def _make_round65_policy_methods(
    original_state: Any,
    original_warn_state: Any,
    original_force: Any,
    recall_policy: Any,
    is_constructing: Any,
    policy_callables: Any,
):
    def authenticated_policy(self: Any) -> tuple[Any, Any, Any] | None:
        baseline = recall_policy(self)
        if baseline is None:
            if is_constructing(self):
                return None
            torch_module = getattr(self, "_torch", None)
            if isinstance(torch_module, types.ModuleType) and (
                torch_module.__name__ == "torch" or torch_module.__name__.startswith("torch.")
            ):
                raise CaptureContractError(
                    "canonical deterministic-policy callable baseline is missing"
                )
            return None
        observed = policy_callables(getattr(self, "_torch", None))
        if any(observed[index] is not baseline[index] for index in range(3)):
            raise CaptureContractError(
                "PyTorch deterministic-policy callable changed after authenticated construction"
            )
        return baseline

    def deterministic_state(self: Any) -> bool:
        baseline = authenticated_policy(self)
        if baseline is None:
            return original_state(self)
        enabled = baseline[1]()
        if not isinstance(enabled, bool):
            raise CaptureContractError("deterministic algorithm state must be boolean")
        return enabled

    def warn_state(self: Any) -> bool:
        baseline = authenticated_policy(self)
        if baseline is None:
            return original_warn_state(self)
        enabled = baseline[2]()
        if not isinstance(enabled, bool):
            raise CaptureContractError("deterministic warn-only state must be boolean")
        return enabled

    def force_policy(self: Any) -> None:
        baseline = authenticated_policy(self)
        if baseline is None:
            original_force(self)
            return
        expected = getattr(self, "_canonical_deterministic_algorithms_enabled", None)
        if expected is None:
            mode = getattr(self, "_determinism_mode", None)
            if mode is None:
                return
            expected = True if mode == "required" else deterministic_state(self)

        warn_expected = getattr(self, "_canonical_deterministic_warn_only_enabled", None)
        enabled_now = deterministic_state(self)
        setter = baseline[0]
        if warn_expected is None:
            if enabled_now is not expected:
                setter(expected)
        else:
            warn_now = warn_state(self)
            if enabled_now is not expected or warn_now is not warn_expected:
                try:
                    setter(expected, warn_only=warn_expected)
                except TypeError as exc:
                    raise CaptureContractError(
                        "canonical capture requires warn_only control on torch.use_deterministic_algorithms"
                    ) from exc
        self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()

    return deterministic_state, warn_state, force_policy


_Round64Backend.__init__ = _make_round65_constructor(
    _Round64Backend.__init__,
    _reassert_snapshot_verifier_bindings_round65,
    _preimport_package_stability_round65,
    _loaded_package_stability_round65,
    _real_transformers_runtime_round65,
    _enter_round65_construction,
    _leave_round65_construction,
    _remember_round65_determinism_callables,
    _determinism_callables_round65,
)
(
    _Round64Backend._deterministic_algorithms_state,
    _Round64Backend._deterministic_warn_only_state,
    _Round64Backend._force_canonical_determinism_policy,
) = _make_round65_policy_methods(
    _Round64Backend._deterministic_algorithms_state,
    _Round64Backend._deterministic_warn_only_state,
    _Round64Backend._force_canonical_determinism_policy,
    _recall_round65_determinism_callables,
    _is_round65_constructing,
    _determinism_callables_round65,
)
del _make_round65_constructor
del _make_round65_policy_methods


HuggingFacePyTorchBackend = _Round64Backend

__all__ = ["HuggingFacePyTorchBackend"]
