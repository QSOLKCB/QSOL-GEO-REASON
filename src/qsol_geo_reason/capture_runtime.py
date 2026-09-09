"""Runtime/build provenance and deterministic device helpers for capture."""

from __future__ import annotations

import hashlib
import json
import re
import types
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance, _validate_python_package_provenance


def _require_torch_build_config(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureContractError("torch_build_config must be a non-whitespace string")
    return value


def _is_real_torch_module(torch: Any) -> bool:
    return (
        isinstance(torch, types.ModuleType)
        and isinstance(getattr(torch, "__name__", None), str)
        and (torch.__name__ == "torch" or torch.__name__.startswith("torch."))
    )


def _cuda_runtime_library_provenance_from_build_config(
    config: str,
) -> dict[str, int | str] | None:
    prefix = "QSOL_GEO_CUDA_RUNTIME="
    records = [line for line in config.rstrip("\n").split("\n") if line.startswith(prefix)]
    if not records:
        return None
    if len(records) != 1:
        raise CaptureContractError(
            "torch_build_config must contain exactly one CUDA runtime provenance record"
        )
    lines = config.rstrip("\n").split("\n")
    if records[0] != lines[-1]:
        raise CaptureContractError(
            "CUDA runtime provenance record must be the final torch_build_config line"
        )
    try:
        payload = json.loads(records[0][len(prefix) :])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureContractError("CUDA runtime provenance record is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"loaded_cuda_runtime_libraries"}
        or not isinstance(payload["loaded_cuda_runtime_libraries"], dict)
    ):
        raise CaptureContractError("CUDA runtime provenance record is malformed")
    provenance = payload["loaded_cuda_runtime_libraries"]
    expected = {
        "cuda_runtime_library_file_count",
        "cuda_runtime_library_receipt_sha256",
    }
    if set(provenance) != expected:
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
    prefix = "QSOL_GEO_CPU_RUNTIME="
    records = [line for line in config.rstrip("\n").split("\n") if line.startswith(prefix)]
    if not records:
        return None
    if len(records) != 1:
        raise CaptureContractError(
            "torch_build_config must contain exactly one CPU runtime provenance record"
        )
    lines = config.rstrip("\n").split("\n")
    index = lines.index(records[0])
    if index < 1 or lines[index - 1] != "QSOL_GEO_CPU_FLUSH_DENORMAL=false":
        raise CaptureContractError(
            "CPU runtime provenance record must immediately follow the canonical CPU denormal receipt"
        )
    if index != len(lines) - 1 and not (
        index == len(lines) - 2 and lines[-1].startswith("QSOL_GEO_CUDA_RUNTIME=")
    ):
        raise CaptureContractError(
            "CPU runtime provenance record must terminate the CPU/MPS build receipt or immediately precede the final CUDA runtime receipt"
        )
    try:
        payload = json.loads(records[0][len(prefix) :])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureContractError("CPU runtime provenance record is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"loaded_cpu_runtime_libraries"}
        or not isinstance(payload["loaded_cpu_runtime_libraries"], dict)
    ):
        raise CaptureContractError("CPU runtime provenance record is malformed")
    provenance = payload["loaded_cpu_runtime_libraries"]
    expected = {
        "cpu_runtime_library_file_count",
        "cpu_runtime_library_receipt_sha256",
    }
    if set(provenance) != expected:
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


def _seed_capture_generators(torch: Any, device: str, seed: int) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda:"):
        try:
            torch.cuda.manual_seed(seed)
        except Exception as exc:
            raise CaptureContractError("unable to seed requested CUDA generator") from exc
    elif device == "mps":
        mps = getattr(torch, "mps", None)
        manual_seed = getattr(mps, "manual_seed", None) if mps is not None else None
        if not callable(manual_seed):
            raise CaptureContractError("canonical MPS capture requires torch.mps.manual_seed")
        try:
            manual_seed(seed)
        except Exception as exc:
            raise CaptureContractError("unable to seed requested MPS generator") from exc


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
        production_device = device == "cpu" or device == "mps" or cuda_active
        if cuda_active and cuda_runtime is None:
            raise CaptureContractError(
                "CUDA torch_build_config is missing the authenticated CUDA runtime library receipt"
            )
        if not cuda_active and cuda_runtime is not None:
            raise CaptureContractError(
                "CUDA runtime library provenance must be absent outside CUDA"
            )
        if production_device and cpu_runtime is None:
            raise CaptureContractError(
                "production torch_build_config is missing the authenticated CPU pooling runtime library receipt"
            )
        if cpu_runtime is not None and not production_device:
            raise CaptureContractError(
                "CPU runtime library provenance must be absent outside production capture devices"
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
        candidate = getattr(properties, "uuid", None)
        if candidate is not None:
            text = str(candidate).strip()
            uuid = text or None
    except Exception:
        uuid = None
    return {
        "cuda_device_name": name.strip(),
        "cuda_device_capability": f"{capability[0]}.{capability[1]}",
        "cuda_device_uuid": uuid,
    }
