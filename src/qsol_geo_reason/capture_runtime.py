"""Device-scoped RNG operations and content-bound runtime identity for capture."""
from __future__ import annotations

import hashlib
import json
import re
import types
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_package import (
    _python_package_provenance,
    _validate_python_package_provenance,
)

_CUDA_RUNTIME_CONFIG_PREFIX = "QSOL_GEO_CUDA_RUNTIME="
_CUDA_RUNTIME_PROVENANCE_KEYS = frozenset(
    {
        "cuda_runtime_library_file_count",
        "cuda_runtime_library_receipt_sha256",
    }
)
_CPU_RUNTIME_CONFIG_PREFIX = "QSOL_GEO_CPU_RUNTIME="
_CPU_RUNTIME_PROVENANCE_KEYS = frozenset(
    {
        "cpu_runtime_library_file_count",
        "cpu_runtime_library_receipt_sha256",
    }
)


def _capture_device_type(device: str) -> str:
    if not isinstance(device, str):
        raise CaptureContractError("capture RNG device must be a string")
    if device in {"cpu", "mps"}:
        return device
    if isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device):
        return "cuda"
    raise CaptureContractError("capture RNG state requires cpu, mps, or an explicit cuda:N")


def _is_real_torch_module(torch: Any) -> bool:
    """Distinguish the imported PyTorch module from intentionally tiny test doubles."""
    return isinstance(torch, types.ModuleType) and getattr(torch, "__name__", None) == "torch"


def _assert_torch_execution_surface(torch: Any) -> None:
    """Authenticate public PyTorch factories against their immutable C-level anchors.

    The production wrapper already binds ``self._torch`` to the exact module imported
    at construction. This closes the remaining gap where code could mutate members of
    that same module, such as replacing ``torch.tensor`` after construction while the
    module identity itself remained unchanged.
    """
    if not _is_real_torch_module(torch):
        return

    c_api = getattr(torch, "_C", None)
    variable_functions = getattr(c_api, "_VariableFunctions", None)
    if variable_functions is None:
        raise CaptureContractError(
            "canonical OBSERVATION cannot authenticate PyTorch C factory bindings"
        )
    for name in ("tensor", "ones_like", "zeros", "isfinite"):
        public = getattr(torch, name, None)
        anchored = getattr(variable_functions, name, None)
        if not callable(public) or public is not anchored:
            raise CaptureContractError(
                "canonical OBSERVATION PyTorch execution callable changed after import: "
                f"torch.{name}"
            )

    # These aliases are executable inputs to the canonical producer: float64
    # controls pooling/accumulation and long controls token materialization.
    # Bind both canonical spellings so replacing either public module attribute
    # cannot silently change the produced vectors or token domain.
    dtype_names = ("float64", "double", "long", "int64")
    # Minimal dependency-free fixtures intentionally omit the dtype surface.
    # A real torch module exposes all four names; partial presence is drift.
    dtype_bindings = (
        (("float64", "double"), ("long", "int64"))
        if any(hasattr(torch, name) for name in dtype_names)
        else ()
    )
    for name, alias in dtype_bindings:
        public = getattr(torch, name, None)
        canonical_alias = getattr(torch, alias, None)
        if public is None or public is not canonical_alias:
            raise CaptureContractError(
                "canonical OBSERVATION PyTorch dtype binding changed after import: "
                f"torch.{name}"
            )

    autograd = getattr(torch, "autograd", None)
    grad_mode = getattr(autograd, "grad_mode", None) if autograd is not None else None
    public_inference_mode = getattr(torch, "inference_mode", None)
    anchored_inference_mode = (
        getattr(grad_mode, "inference_mode", None) if grad_mode is not None else None
    )
    if (
        not callable(public_inference_mode)
        or public_inference_mode is not anchored_inference_mode
    ):
        raise CaptureContractError(
            "canonical OBSERVATION PyTorch execution callable changed after import: "
            "torch.inference_mode"
        )


def _seed_capture_generators(torch: Any, device: str, seed: int) -> None:
    """Seed the CPU generator and only the accelerator selected by the request.

    torch.manual_seed also seeds other accelerators, including their lazy queues.
    Calling the CPU generator directly avoids changing a future CUDA/MPS session.
    """
    _assert_torch_execution_surface(torch)
    device_type = _capture_device_type(device)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise CaptureContractError("capture seed must be an unsigned 64-bit integer")
    generator = getattr(torch, "default_generator", None)
    cpu_seed = getattr(generator, "manual_seed", None)
    if not callable(cpu_seed):
        raise CaptureContractError("canonical capture requires the default CPU generator")
    try:
        cpu_seed(seed)
        if device_type == "cuda":
            # manual_seed targets the current device; restore the caller's device
            # after seeding the explicit index, rather than seeding every GPU.
            with torch.cuda.device(int(device.split(":", 1)[1])):
                torch.cuda.manual_seed(seed)
        elif device_type == "mps":
            torch.mps.manual_seed(seed)
    except Exception as exc:
        raise CaptureContractError("unable to seed the requested capture generators") from exc
    _assert_torch_execution_surface(torch)


def _snapshot_accelerator_rng(torch: Any, device: str) -> dict[str, Any] | None:
    """Never probe or initialize an accelerator on the CPU-only path."""
    device_type = _capture_device_type(device)
    if device_type == "cpu":
        return None
    try:
        if device_type == "cuda":
            state = torch.cuda.get_rng_state(int(device.split(":", 1)[1]))
        else:
            state = torch.mps.get_rng_state()
        clone = getattr(state, "clone", None)
        return {"device": device, "state": clone() if callable(clone) else state}
    except Exception as exc:
        raise CaptureContractError("unable to snapshot the requested accelerator RNG") from exc


def _restore_accelerator_rng(torch: Any, receipt: Mapping[str, Any] | None) -> None:
    if receipt is None:
        return
    if not isinstance(receipt, Mapping) or set(receipt) != {"device", "state"}:
        raise CaptureContractError("capture accelerator RNG snapshot is malformed")
    device = receipt["device"]
    device_type = _capture_device_type(device)
    try:
        if device_type == "cuda":
            torch.cuda.set_rng_state(receipt["state"], int(device.split(":", 1)[1]))
        elif device_type == "mps":
            torch.mps.set_rng_state(receipt["state"])
        else:
            raise CaptureContractError("CPU-only captures cannot carry an accelerator RNG snapshot")
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError("unable to restore the requested accelerator RNG") from exc


def _require_torch_build_config(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureContractError("torch_build_config must be a non-whitespace string")
    # Preserve build information verbatim; the receipt authenticates these UTF-8 bytes.
    return value


def _cuda_runtime_library_provenance_from_build_config(
    config: str,
) -> dict[str, int | str] | None:
    """Parse and validate the canonical CUDA shared-library receipt extension.

    The producer appends exactly one compact JSON record as the final build-config
    line. Canonical verification must understand that record rather than treating it
    as opaque text, otherwise a caller can replace the complete config and merely
    recompute its outer SHA-256.
    """
    record_lines = [
        line
        for line in config.splitlines()
        if line.startswith(_CUDA_RUNTIME_CONFIG_PREFIX)
    ]
    if not record_lines:
        return None
    if len(record_lines) != 1:
        raise CaptureContractError(
            "torch_build_config must contain exactly one CUDA runtime provenance record"
        )
    record_line = record_lines[0]
    if config.rstrip("\n").split("\n")[-1] != record_line:
        raise CaptureContractError(
            "CUDA runtime provenance record must be the final torch_build_config line"
        )
    try:
        payload = json.loads(record_line[len(_CUDA_RUNTIME_CONFIG_PREFIX) :])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureContractError("CUDA runtime provenance record is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"loaded_cuda_runtime_libraries"}
        or not isinstance(payload["loaded_cuda_runtime_libraries"], dict)
    ):
        raise CaptureContractError("CUDA runtime provenance record is malformed")
    provenance = payload["loaded_cuda_runtime_libraries"]
    if set(provenance) != _CUDA_RUNTIME_PROVENANCE_KEYS:
        raise CaptureContractError("CUDA runtime provenance record is malformed")
    count = provenance["cuda_runtime_library_file_count"]
    receipt = provenance["cuda_runtime_library_receipt_sha256"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(
            "CUDA runtime library file count must be a positive integer"
        )
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(character not in "0123456789abcdef" for character in receipt)
    ):
        raise CaptureContractError(
            "CUDA runtime library receipt must be a lowercase SHA-256 digest"
        )
    return {
        "cuda_runtime_library_file_count": count,
        "cuda_runtime_library_receipt_sha256": receipt,
    }


def _cpu_runtime_library_provenance_from_build_config(
    config: str,
) -> dict[str, int | str] | None:
    """Parse and validate the canonical external CPU pooling-runtime receipt."""
    record_lines = [
        line
        for line in config.splitlines()
        if line.startswith(_CPU_RUNTIME_CONFIG_PREFIX)
    ]
    if not record_lines:
        return None
    if len(record_lines) != 1:
        raise CaptureContractError(
            "torch_build_config must contain exactly one CPU runtime provenance record"
        )
    record_line = record_lines[0]
    terminal_lines = config.rstrip("\n").split("\n")
    record_index = terminal_lines.index(record_line)
    if record_index != len(terminal_lines) - 1:
        immediately_before_cuda = (
            record_index == len(terminal_lines) - 2
            and terminal_lines[-1].startswith(_CUDA_RUNTIME_CONFIG_PREFIX)
        )
        if not immediately_before_cuda:
            raise CaptureContractError(
                "CPU runtime provenance record must terminate torch_build_config or "
                "immediately precede the final CUDA runtime provenance record"
            )
    try:
        payload = json.loads(record_line[len(_CPU_RUNTIME_CONFIG_PREFIX) :])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureContractError("CPU runtime provenance record is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"loaded_cpu_runtime_libraries"}
        or not isinstance(payload["loaded_cpu_runtime_libraries"], dict)
    ):
        raise CaptureContractError("CPU runtime provenance record is malformed")
    provenance = payload["loaded_cpu_runtime_libraries"]
    if set(provenance) != _CPU_RUNTIME_PROVENANCE_KEYS:
        raise CaptureContractError("CPU runtime provenance record is malformed")
    count = provenance["cpu_runtime_library_file_count"]
    receipt = provenance["cpu_runtime_library_receipt_sha256"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CaptureContractError(
            "CPU runtime library file count must be a non-negative integer"
        )
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(character not in "0123456789abcdef" for character in receipt)
    ):
        raise CaptureContractError(
            "CPU runtime library receipt must be a lowercase SHA-256 digest"
        )
    return {
        "cpu_runtime_library_file_count": count,
        "cpu_runtime_library_receipt_sha256": receipt,
    }


def _torch_build_metadata(torch: Any) -> dict[str, Any]:
    show = getattr(getattr(torch, "__config__", None), "show", None)
    if not callable(show):
        raise CaptureContractError("canonical capture requires torch.__config__.show()")
    try:
        config = _require_torch_build_config(show())
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError("unable to record the PyTorch build configuration") from exc
    metadata: dict[str, Any] = {
        "torch_build_config": config,
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }
    if _is_real_torch_module(torch):
        package = _python_package_provenance(torch, "PyTorch")
        metadata["torch_package_file_count"] = package["file_count"]
        metadata["torch_package_receipt_sha256"] = package["receipt_sha256"]
    return metadata


def _validate_torch_build_metadata(observed: Mapping[str, Any]) -> None:
    config = _require_torch_build_config(observed.get("torch_build_config"))
    expected = hashlib.sha256(config.encode("utf-8")).hexdigest()
    if observed.get("torch_build_config_sha256") != expected:
        raise CaptureContractError("torch_build_config_sha256 does not authenticate the recorded build")

    cuda_runtime = _cuda_runtime_library_provenance_from_build_config(config)
    cpu_runtime = _cpu_runtime_library_provenance_from_build_config(config)
    device = observed.get("device")
    if isinstance(device, str):
        cuda_active = re.fullmatch(r"cuda:[0-9]+", device) is not None
        cpu_active = device == "cpu"
        production_device = cpu_active or device == "mps" or cuda_active
        cpu_pooling_active = observed.get("pool_accumulation_device") == "cpu"
        if cuda_active and cuda_runtime is None:
            raise CaptureContractError(
                "CUDA torch_build_config is missing the authenticated CUDA runtime library receipt"
            )
        if not cuda_active and cuda_runtime is not None:
            raise CaptureContractError(
                "CUDA runtime library provenance must be absent outside CUDA"
            )
        if (cpu_active or (production_device and cpu_pooling_active)) and cpu_runtime is None:
            raise CaptureContractError(
                "CPU pooling torch_build_config is missing the authenticated CPU runtime library receipt"
            )
        if cpu_runtime is not None and not (cpu_active or cpu_pooling_active):
            raise CaptureContractError(
                "CPU runtime library provenance must be absent outside CPU pooling"
            )

    # Standalone build-receipt tests may intentionally exercise only __config__.
    # Canonical production metadata exact-key validation requires both package
    # fields, so when either is present validate the pair as a complete receipt.
    if (
        "torch_package_file_count" in observed
        or "torch_package_receipt_sha256" in observed
    ):
        _validate_python_package_provenance(
            observed,
            count_field="torch_package_file_count",
            receipt_field="torch_package_receipt_sha256",
            where="PyTorch",
        )


def _cuda_device_identity(torch: Any, index: int) -> dict[str, str | None]:
    """Require device name/capability; keep only the optional UUID best-effort."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CaptureContractError("CUDA hardware identity requires an explicit non-negative device index")
    try:
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
    except Exception as exc:
        raise CaptureContractError("unable to record required CUDA hardware identity") from exc
    if not isinstance(name, str) or not name.strip():
        raise CaptureContractError("cuda_device_name must be a non-whitespace string for CUDA")
    if (
        not isinstance(capability, (tuple, list)) or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability)
    ):
        raise CaptureContractError("CUDA device capability must contain two non-negative integers")
    uuid = None
    try:
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = getattr(properties, "uuid", None)
        if raw_uuid is not None:
            uuid = str(raw_uuid).strip() or None
    except Exception:
        # Not all supported PyTorch builds expose a device UUID. This does not
        # make the mandatory device name and capability optional.
        pass
    return {
        "cuda_device_name": name,
        "cuda_device_capability": f"{capability[0]}.{capability[1]}",
        "cuda_device_uuid": uuid,
    }
