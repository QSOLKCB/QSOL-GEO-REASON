"""Round-44 hardening for checkout, verification, CUDA identity, and loaders.

This layer intentionally leaves the established production/core constructors unchanged.
It installs two canonical validation wrappers before capture execution imports them, and
extends the current exact-type OBSERVATION backend through the same subclass/rebind
pattern used by the preceding hardening layers.
"""
from __future__ import annotations

import sys
import types
import weakref
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .capture_backend_round39 import HuggingFacePyTorchBackend as _Round39Backend
from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance
from .provenance import SourceIdentityError, _stable_code_identity
from . import capture_backend_production as _production
from . import capture_provenance as _capture_provenance
from . import provenance as _provenance


_NATIVE_IMPORT_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
_LEGACY_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt")
_TREE_RECEIPT_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")


def _is_importable_package_native_artifact(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return (
        len(parts) >= 3
        and tuple(parts[:2]) == ("src", "qsol_geo_reason")
        and normalized.lower().endswith(_NATIVE_IMPORT_SUFFIXES)
    )


def _ignored_importable_native_artifacts(root: Path) -> tuple[str, ...]:
    """Return ignored native files capable of participating in package execution."""
    try:
        result = _provenance._git_run(
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        raise SourceIdentityError(
            "unable to inspect ignored native artifacts for checkout-bound execution"
        ) from exc
    return tuple(
        sorted(
            path.replace("\\", "/")
            for path in result.stdout.split("\0")
            if path and _is_importable_package_native_artifact(path)
        )
    )


def _assert_no_ignored_importable_native_artifacts(root: Path) -> None:
    paths = _ignored_importable_native_artifacts(root)
    if paths:
        preview = ", ".join(paths[:3])
        raise SourceIdentityError(
            "ignored native artifact exists inside the importable qsol_geo_reason package: "
            + preview
        )


_ORIGINAL_GIT_SOURCE_REVISION = _provenance.git_source_revision


def _git_source_revision_round44(
    *, require_clean: bool = True, reject_importable_bytecode: bool = False
) -> str | None:
    observed = _ORIGINAL_GIT_SOURCE_REVISION(
        require_clean=require_clean,
        reject_importable_bytecode=reject_importable_bytecode,
    )
    if observed is not None and require_clean and reject_importable_bytecode:
        _assert_no_ignored_importable_native_artifacts(_provenance.source_repo_root())
    return observed


# resolve_implementation_revision() performs a module-global lookup of
# git_source_revision at call time, so this wrapper also protects the public
# checkout-bound OBSERVATION path without replacing the resolver itself.
_provenance.git_source_revision = _git_source_revision_round44


def _require_observation_tree_receipts(request: Mapping[str, Any]) -> None:
    model = request.get("model")
    if not isinstance(model, Mapping):
        raise CaptureContractError("OBSERVATION request model provenance is missing")
    missing = []
    for field in _TREE_RECEIPT_FIELDS:
        value = model.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            missing.append(field)
    if missing:
        raise CaptureContractError(
            "verified OBSERVATION requires frozen Hub commit-tree SHA-256 receipts: "
            + ", ".join(missing)
        )


def _assert_observation_safetensors_policy(observed: Mapping[str, Any]) -> None:
    if observed.get("safetensors_deserializer_active") is not True:
        raise CaptureContractError(
            "verified OBSERVATION requires the Safetensors checkpoint deserializer"
        )
    hashes = observed.get("model_snapshot_file_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise CaptureContractError("verified OBSERVATION model snapshot provenance is missing")
    paths = [path.lower() for path in hashes if isinstance(path, str)]
    if not any(path.endswith(".safetensors") for path in paths):
        raise CaptureContractError(
            "verified OBSERVATION requires at least one authenticated Safetensors weight artifact"
        )
    legacy = sorted(
        path for path in paths if path.endswith(_LEGACY_PICKLE_SUFFIXES)
    )
    if legacy:
        raise CaptureContractError(
            "verified OBSERVATION forbids pickle-capable checkpoint artifacts: "
            + ", ".join(legacy[:3])
        )


_ORIGINAL_VALIDATE_BACKEND_METADATA = _capture_provenance._validate_backend_metadata


def _validate_backend_metadata_round44(
    observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str
) -> None:
    if evidence_class == "OBSERVATION":
        _require_observation_tree_receipts(request)
    _ORIGINAL_VALIDATE_BACKEND_METADATA(observed, request, evidence_class)
    if evidence_class == "OBSERVATION":
        _assert_observation_safetensors_policy(observed)


# capture_execute and capture_verify are imported only after this module by capture.py,
# so their from-import bindings receive the hardened verifier directly.
_capture_provenance._validate_backend_metadata = _validate_backend_metadata_round44


def _unwrap_python_function(value: Any, where: str) -> types.FunctionType:
    if isinstance(value, types.MethodType):
        value = value.__func__
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    if not isinstance(value, types.FunctionType):
        raise CaptureContractError(f"{where} is not a receipt-backed Python function")
    return value


def _compiled_source_index(source_path: Path) -> dict[tuple[str, int], types.CodeType]:
    try:
        module_code = compile(
            source_path.read_bytes(),
            str(source_path),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        raise CaptureContractError(
            f"unable to compile receipt-backed Transformers source: {source_path}"
        ) from exc
    index: dict[tuple[str, int], types.CodeType] = {}

    def visit(code: types.CodeType) -> None:
        key = (code.co_qualname, code.co_firstlineno)
        if key in index:
            raise CaptureContractError(
                f"receipt-backed Transformers source has ambiguous code identity: {source_path}"
            )
        index[key] = code
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                visit(constant)

    visit(module_code)
    return index


def _assert_package_callable_matches_source(
    package_module: Any,
    value: Any,
    where: str,
    *,
    source_cache: dict[Path, dict[tuple[str, int], types.CodeType]] | None = None,
) -> None:
    """Bind a live Python callable to source inside the receipt-backed package tree."""
    module_file = getattr(package_module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise CaptureContractError("canonical capture cannot locate imported Transformers")
    try:
        package_root = Path(module_file).resolve(strict=True).parent
    except OSError as exc:
        raise CaptureContractError("canonical capture cannot resolve imported Transformers") from exc

    function = _unwrap_python_function(value, where)
    filename = function.__code__.co_filename
    if not filename or filename.startswith("<"):
        raise CaptureContractError(f"{where} has no receipt-backed package source filename")
    try:
        source_path = Path(filename).resolve(strict=True)
        source_path.relative_to(package_root)
    except (OSError, ValueError) as exc:
        raise CaptureContractError(f"{where} executes outside the imported Transformers package") from exc
    if source_path.suffix != ".py":
        raise CaptureContractError(f"{where} is not backed by Transformers Python source")

    cache = source_cache if source_cache is not None else {}
    index = cache.get(source_path)
    if index is None:
        index = _compiled_source_index(source_path)
        cache[source_path] = index
    code = function.__code__
    expected = index.get((code.co_qualname, code.co_firstlineno))
    if expected is None or _stable_code_identity(code) != _stable_code_identity(expected):
        raise CaptureContractError(
            f"{where} does not match the receipt-backed Transformers package source"
        )


def _transformers_receipt(module: Any) -> tuple[int, str]:
    provenance = _python_package_provenance(module, "Transformers")
    count = provenance.get("file_count")
    receipt = provenance.get("receipt_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError("Transformers package file count is invalid")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise CaptureContractError("Transformers package receipt is invalid")
    return count, receipt


def _assert_transformers_loaders_source_bound(module: Any) -> tuple[int, str]:
    receipt = _transformers_receipt(module)
    cache: dict[Path, dict[tuple[str, int], types.CodeType]] = {}
    for owner_name in ("AutoTokenizer", "AutoModelForCausalLM"):
        owner = getattr(module, owner_name, None)
        loader = getattr(owner, "from_pretrained", None) if owner is not None else None
        if loader is None:
            raise CaptureContractError(
                f"Transformers {owner_name}.from_pretrained is unavailable"
            )
        _assert_package_callable_matches_source(
            module,
            loader,
            f"Transformers {owner_name}.from_pretrained",
            source_cache=cache,
        )
    return receipt


def _assert_transformers_model_source_bound(module: Any, model: Any) -> None:
    root_module = getattr(type(model), "__module__", "")
    if not isinstance(root_module, str) or not (
        root_module == "transformers" or root_module.startswith("transformers.")
    ):
        raise CaptureContractError(
            "canonical model returned by Transformers loader is not a Transformers implementation"
        )
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise CaptureContractError("canonical Transformers model does not expose named_modules()")

    cache: dict[Path, dict[tuple[str, int], types.CodeType]] = {}
    checked: set[tuple[str, int]] = set()
    try:
        modules = tuple(named_modules())
    except Exception as exc:
        raise CaptureContractError("unable to enumerate the loaded Transformers model graph") from exc
    for _name, child in modules:
        child_module = getattr(type(child), "__module__", "")
        if not isinstance(child_module, str) or not (
            child_module == "transformers" or child_module.startswith("transformers.")
        ):
            continue
        forward = getattr(type(child), "forward", None)
        function = _unwrap_python_function(
            forward,
            f"Transformers model forward {type(child).__qualname__}",
        )
        identity = (function.__module__, id(function))
        if identity in checked:
            continue
        checked.add(identity)
        _assert_package_callable_matches_source(
            module,
            function,
            f"Transformers model forward {type(child).__qualname__}",
            source_cache=cache,
        )


def _make_round44_vault():
    cuda_identities: weakref.WeakKeyDictionary[
        Any, tuple[str, str, str | None]
    ] = weakref.WeakKeyDictionary()
    transformers_receipts: weakref.WeakKeyDictionary[
        Any, tuple[int, str]
    ] = weakref.WeakKeyDictionary()

    def remember_cuda(instance: Any, identity: tuple[str, str, str | None]) -> None:
        if instance in cuda_identities:
            if cuda_identities[instance] != identity:
                raise CaptureContractError("canonical CUDA hardware identity changed during construction")
            return
        cuda_identities[instance] = identity

    def recall_cuda(instance: Any) -> tuple[str, str, str | None] | None:
        return cuda_identities.get(instance)

    def remember_transformers(instance: Any, receipt: tuple[int, str]) -> None:
        if instance in transformers_receipts:
            if transformers_receipts[instance] != receipt:
                raise CaptureContractError("Transformers package receipt changed during construction")
            return
        transformers_receipts[instance] = receipt

    def recall_transformers(instance: Any) -> tuple[int, str] | None:
        return transformers_receipts.get(instance)

    return remember_cuda, recall_cuda, remember_transformers, recall_transformers


(
    _remember_round44_cuda_identity,
    _recall_round44_cuda_identity,
    _remember_round44_transformers_receipt,
    _recall_round44_transformers_receipt,
) = _make_round44_vault()
del _make_round44_vault


def _cuda_identity_tuple(identity: Any) -> tuple[str, str, str | None]:
    if not isinstance(identity, Mapping):
        raise CaptureContractError("canonical CUDA hardware identity is missing")
    name = identity.get("cuda_device_name")
    capability = identity.get("cuda_device_capability")
    uuid = identity.get("cuda_device_uuid")
    if not isinstance(name, str) or not name.strip():
        raise CaptureContractError("canonical CUDA hardware identity lacks device name")
    if not isinstance(capability, str) or not capability.strip():
        raise CaptureContractError("canonical CUDA hardware identity lacks device capability")
    if uuid is not None and (not isinstance(uuid, str) or not uuid.strip()):
        raise CaptureContractError("canonical CUDA hardware UUID must be nonblank or null")
    return name, capability, uuid


class HuggingFacePyTorchBackend(_Round39Backend):
    """Newest canonical boundary for Round-44 provenance hardening."""

    def _assert_round44_cuda_hardware_identity(self) -> None:
        expected = _recall_round44_cuda_identity(self)
        if expected is None:
            return
        observed = _cuda_identity_tuple(getattr(self, "_cuda_hardware_identity", None))
        if observed != expected:
            raise CaptureContractError(
                "CUDA hardware identity changed after authenticated construction"
            )

    def _assert_round44_transformers_source_identity(self) -> None:
        expected = _recall_round44_transformers_receipt(self)
        if expected is None:
            return
        module = getattr(self, "_transformers", None)
        if module is None or _transformers_receipt(module) != expected:
            raise CaptureContractError(
                "Transformers package changed after pre-load loader authentication"
            )
        model = getattr(self, "_model", None)
        if model is not None:
            _assert_transformers_model_source_bound(module, model)

    def _assert_autocast_disabled(self) -> None:
        # Core has already resolved the CUDA device identity at this point, but no
        # snapshot or model/tokenizer deserializer has run. Freeze that identity in a
        # closure-owned vault before any later instance state can redefine it.
        if getattr(self, "_device_type", None) == "cuda":
            _remember_round44_cuda_identity(
                self,
                _cuda_identity_tuple(getattr(self, "_cuda_hardware_identity", None)),
            )

        # Preserve the established Hub-tree and Safetensors preflight first. It may
        # fail before package work when the request lacks its production receipts.
        super()._assert_autocast_disabled()

        # The core imports Transformers and its Auto classes before this dynamic hook,
        # but invokes from_pretrained only afterward. Authenticate the exact in-memory
        # loader methods against the receipt-backed package source in that gap.
        module = getattr(self, "_transformers", None)
        if module is not None:
            _remember_round44_transformers_receipt(
                self,
                _assert_transformers_loaders_source_bound(module),
            )

    def _assert_cpu_dispatch_policy(self) -> None:
        # Production construction invokes this after the real model/tokenizer load.
        super()._assert_cpu_dispatch_policy()
        self._assert_round44_cuda_hardware_identity()
        self._assert_round44_transformers_source_identity()

    def _assert_final_construction_baseline(self) -> None:
        super()._assert_final_construction_baseline()
        self._assert_round44_cuda_hardware_identity()
        self._assert_round44_transformers_source_identity()


# Rebind the exact-type OBSERVATION gate before capture_execute imports production.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]
