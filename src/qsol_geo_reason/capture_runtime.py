"""Device-scoped RNG operations and content-bound runtime identity for capture."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from .capture_common import CaptureContractError


def _capture_device_type(device: str) -> str:
    if not isinstance(device, str):
        raise CaptureContractError("capture RNG device must be a string")
    if device in {"cpu", "mps"}:
        return device
    if isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device):
        return "cuda"
    raise CaptureContractError("capture RNG state requires cpu, mps, or an explicit cuda:N")


def _seed_capture_generators(torch: Any, device: str, seed: int) -> None:
    """Seed the CPU generator and only the accelerator selected by the request.

    torch.manual_seed also seeds other accelerators, including their lazy queues.
    Calling the CPU generator directly avoids changing a future CUDA/MPS session.
    """
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


def _torch_build_metadata(torch: Any) -> dict[str, str]:
    show = getattr(getattr(torch, "__config__", None), "show", None)
    if not callable(show):
        raise CaptureContractError("canonical capture requires torch.__config__.show()")
    try:
        config = _require_torch_build_config(show())
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError("unable to record the PyTorch build configuration") from exc
    return {
        "torch_build_config": config,
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }


def _validate_torch_build_metadata(observed: Mapping[str, Any]) -> None:
    config = _require_torch_build_config(observed.get("torch_build_config"))
    expected = hashlib.sha256(config.encode("utf-8")).hexdigest()
    if observed.get("torch_build_config_sha256") != expected:
        raise CaptureContractError("torch_build_config_sha256 does not authenticate the recorded build")


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
