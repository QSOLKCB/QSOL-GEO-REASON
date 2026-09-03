"""Backend provenance, hardware identity, and snapshot authentication helpers."""
from __future__ import annotations
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
from .canonical import sha256_json
from .capture_common import (
    _BLOCK_CONTAINER_PATHS,
    _CAPTURE_PHASE,
    _PRODUCTION_BACKEND,
    _PRODUCTION_BACKEND_KEYS,
    _SIMULATION_BACKEND,
    _SIMULATION_BACKEND_KEYS,
    CaptureContractError,
    _require_exact_keys,
)


def _validate_backend_identity(
    observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str
) -> None:
    expected_name = _PRODUCTION_BACKEND if evidence_class == "OBSERVATION" else _SIMULATION_BACKEND
    if observed.get("name") != expected_name:
        raise CaptureContractError(f"backend metadata name must be {expected_name!r} for {evidence_class}")
    if observed.get("observed_model_commit") != request["model"]["revision"]:
        raise CaptureContractError("observed model commit does not match the frozen request revision")
    if observed.get("observed_tokenizer_commit") != request["model"]["tokenizer_revision"]:
        raise CaptureContractError("observed tokenizer commit does not match the frozen request revision")


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_snapshot_receipt(observed: Mapping[str, Any], prefix: str) -> None:
    hashes = observed.get(f"{prefix}_snapshot_file_sha256")
    count = observed.get(f"{prefix}_snapshot_file_count")
    receipt = observed.get(f"{prefix}_snapshot_receipt_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise CaptureContractError(f"{prefix} snapshot artifact hashes are missing")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(f"{prefix} snapshot file count is invalid")
    if count != len(hashes):
        raise CaptureContractError(f"{prefix} snapshot file count does not match artifact hashes")
    for path, digest in hashes.items():
        if not isinstance(path, str) or not path or not _is_lower_sha256(digest):
            raise CaptureContractError(f"{prefix} snapshot artifact hashes are malformed")
    if not _is_lower_sha256(receipt) or receipt != sha256_json(hashes):
        raise CaptureContractError(f"{prefix} snapshot receipt SHA-256 is invalid")


def _validate_required_determinism(observed: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    enabled = observed.get("deterministic_algorithms_enabled")
    if not isinstance(enabled, bool):
        raise CaptureContractError("deterministic_algorithms_enabled must be boolean")
    if request["determinism"]["mode"] == "required" and enabled is not True:
        raise CaptureContractError(
            "required determinism requires deterministic_algorithms_enabled=true"
        )


def _validate_nullable_string(value: Any, where: str) -> None:
    if value is not None and not isinstance(value, str):
        raise CaptureContractError(f"{where} must be string or null")


def _validate_nullable_integer(value: Any, where: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise CaptureContractError(f"{where} must be integer or null")


def _validate_nullable_boolean(value: Any, where: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise CaptureContractError(f"{where} must be boolean or null")


def _env_flag_enabled(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in {"", "0", "false", "no", "off"}


def _validate_production_metadata_shape(observed: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Mirror the production run-manifest schema's field types and domains."""
    nonempty_strings = (
        "python_version", "platform", "torch_version", "transformers_version",
        "model_class", "tokenizer_class", "device",
    )
    for field in nonempty_strings:
        value = observed.get(field)
        if not isinstance(value, str) or not value:
            raise CaptureContractError(f"production backend field {field} must be a non-empty string")

    nullable_strings = (
        "tokenizers_version", "huggingface_hub_version", "attention_implementation",
        "cpu_machine", "cpu_processor", "cpu_instruction_flags", "omp_num_threads",
        "mkl_num_threads", "cuda_device_name", "cuda_device_capability", "cuda_device_uuid",
        "cuda_visible_devices", "cuda_build_version", "nvidia_driver_version",
        "float32_matmul_precision", "nvidia_tf32_override", "torch_allow_tf32_cublas_override",
        "cublas_workspace_config", "mps_mac_model", "mps_cpu_brand", "mps_macos_version",
        "mps_fallback_env", "mps_fast_math_env",
    )
    for field in nullable_strings:
        _validate_nullable_string(observed.get(field), f"production backend field {field}")

    for field in ("torch_num_threads", "torch_num_interop_threads", "cudnn_version", "cuda_resolved_device_index"):
        _validate_nullable_integer(observed.get(field), f"production backend field {field}")
    for field in (
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "sdpa_flash_enabled",
        "sdpa_mem_efficient_enabled", "sdpa_math_enabled", "sdpa_cudnn_enabled",
    ):
        _validate_nullable_boolean(observed.get(field), f"production backend field {field}")
    for field in ("mps_device_active", "mps_built", "mps_available", "autocast_disabled"):
        if not isinstance(observed.get(field), bool):
            raise CaptureContractError(f"production backend field {field} must be boolean")

    block_path = observed.get("hidden_state_block_path")
    if block_path not in _BLOCK_CONTAINER_PATHS:
        raise CaptureContractError("hidden_state_block_path is not a recognized canonical decoder block path")
    hidden_state_count = observed.get("hidden_state_count")
    if isinstance(hidden_state_count, bool) or not isinstance(hidden_state_count, int) or hidden_state_count < 2:
        raise CaptureContractError("hidden_state_count must be an integer >= 2")
    requested_layers = request["capture"]["layers"]
    if any(layer >= hidden_state_count for layer in requested_layers):
        raise CaptureContractError("hidden_state_count does not cover every requested capture layer")

    dtype_map = observed.get("observed_hidden_state_dtypes")
    if not isinstance(dtype_map, dict):
        raise CaptureContractError("observed_hidden_state_dtypes must be an object")
    expected_keys = {str(layer) for layer in requested_layers}
    if set(dtype_map) != expected_keys:
        raise CaptureContractError("observed_hidden_state_dtypes keys do not match requested layers")
    for layer, values in dtype_map.items():
        if (
            not isinstance(layer, str)
            or not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise CaptureContractError("observed_hidden_state_dtypes has an invalid layer dtype set")

    device = request["backend"]["device"]
    cuda_active = device.startswith("cuda:")
    expected_cuda_index = int(device.split(":", 1)[1]) if cuda_active else None
    if observed.get("cuda_resolved_device_index") != expected_cuda_index:
        raise CaptureContractError("cuda_resolved_device_index does not match the explicit request device")
    attention = observed.get("attention_implementation")
    sdpa_fields = {
        "sdpa_flash_enabled": observed.get("sdpa_flash_enabled"),
        "sdpa_mem_efficient_enabled": observed.get("sdpa_mem_efficient_enabled"),
        "sdpa_math_enabled": observed.get("sdpa_math_enabled"),
        "sdpa_cudnn_enabled": observed.get("sdpa_cudnn_enabled"),
    }
    if cuda_active and attention == "sdpa":
        if sdpa_fields["sdpa_flash_enabled"] is not False:
            raise CaptureContractError("canonical CUDA SDPA requires Flash SDPA disabled")
        if sdpa_fields["sdpa_mem_efficient_enabled"] is not False:
            raise CaptureContractError("canonical CUDA SDPA requires memory-efficient SDPA disabled")
        if sdpa_fields["sdpa_math_enabled"] is not True:
            raise CaptureContractError("canonical CUDA SDPA requires math SDPA enabled")
        if sdpa_fields["sdpa_cudnn_enabled"] is True:
            raise CaptureContractError("canonical CUDA SDPA requires cuDNN SDPA disabled")
    elif any(value is not None for value in sdpa_fields.values()):
        raise CaptureContractError("SDPA policy fields must be null outside the canonical CUDA SDPA lane")

    mps_active = device.startswith("mps")
    if observed.get("mps_device_active") is not mps_active:
        raise CaptureContractError("mps_device_active does not match the requested device")
    if mps_active:
        if _env_flag_enabled(observed.get("mps_fallback_env")):
            raise CaptureContractError("canonical MPS provenance forbids fallback enablement")
        if _env_flag_enabled(observed.get("mps_fast_math_env")):
            raise CaptureContractError("canonical MPS provenance forbids fast-math enablement")


def _validate_backend_metadata(observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str) -> None:
    required = _PRODUCTION_BACKEND_KEYS if evidence_class == "OBSERVATION" else _SIMULATION_BACKEND_KEYS
    _require_exact_keys(observed, required=set(required), where=f"{evidence_class.lower()} backend_observed")
    _validate_backend_identity(observed, request, evidence_class)
    expected = {
        "device": request["backend"]["device"],
        "dtype": request["backend"]["dtype"],
        "quantization": "none",
        "local_files_only": True,
        "trust_remote_code": False,
        "use_cache": False,
        "capture_phase": _CAPTURE_PHASE,
        "kv_cache_reuse": False,
        "determinism_mode": request["determinism"]["mode"],
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise CaptureContractError(f"backend provenance field {key} does not match the capture contract")
    if evidence_class == "OBSERVATION":
        _validate_production_metadata_shape(observed, request)
        production_constants = {
            "checkpoint_loading_clean": True, "quantization_config_present": False,
            "model_reports_quantized": False, "offloading": "none",
            "pool_accumulation_dtype": "float64", "pool_accumulation_device": "cpu",
            "hidden_state_capture_strategy": "selective_forward_hooks",
            "snapshot_authentication": "sha256_all_snapshot_files_pre_and_post_load",
            "autocast_disabled": True,
        }
        for key, value in production_constants.items():
            if observed.get(key) != value:
                raise CaptureContractError(f"observation backend provenance field {key} is invalid")
        _validate_required_determinism(observed, request)
        for prefix in ("model", "tokenizer"):
            _validate_snapshot_receipt(observed, prefix)


def _cpu_hardware_metadata(torch_module: Any) -> dict[str, Any]:
    cpu_model: str | None = platform.processor() or None
    cpu_flags: str | None = None
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        first = text.split("\n\n", 1)[0]
        fields = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip() for line in first.splitlines() if ":" in line}
        cpu_model = fields.get("model name") or fields.get("Processor") or fields.get("Hardware") or cpu_model
        cpu_flags = fields.get("flags") or fields.get("Features")
    except OSError:
        pass
    def thread_value(name: str) -> int | None:
        fn = getattr(torch_module, name, None)
        if fn is None:
            return None
        try:
            value = fn()
        except Exception:
            return None
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    return {
        "cpu_machine": platform.machine() or None, "cpu_processor": cpu_model,
        "cpu_instruction_flags": cpu_flags, "torch_num_threads": thread_value("get_num_threads"),
        "torch_num_interop_threads": thread_value("get_num_interop_threads"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _get_dotted_attr(root: Any, path: str) -> Any:
    value = root
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def _config_hidden_layer_count(config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise CaptureContractError("canonical selective hidden-state capture requires an explicit positive layer count in model config")


def _resolve_hidden_state_layout(model: Any) -> tuple[Any, str, Any]:
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise CaptureContractError("model does not expose a canonical base_model")
    expected = _config_hidden_layer_count(getattr(model, "config", None))
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path in _BLOCK_CONTAINER_PATHS:
        blocks = _get_dotted_attr(base_model, path)
        if blocks is None:
            continue
        try:
            length = len(blocks)
            blocks[0]
        except (TypeError, IndexError, KeyError):
            continue
        if length == expected and id(blocks) not in seen:
            seen.add(id(blocks))
            candidates.append((path, blocks))
    if len(candidates) != 1:
        raise CaptureContractError(
            "canonical selective hidden-state capture requires exactly one recognized "
            f"decoder block sequence of length {expected}; candidates={[p for p, _ in candidates]!r}"
        )
    path, blocks = candidates[0]
    return base_model, path, blocks


def _extract_hidden_tensor(value: Any) -> Any:
    if value is None:
        return None
    hidden = getattr(value, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(value, Mapping) and value.get("last_hidden_state") is not None:
        return value["last_hidden_state"]
    if isinstance(value, (tuple, list)) and value:
        return _extract_hidden_tensor(value[0])
    if hasattr(value, "ndim") and hasattr(value, "shape") and hasattr(value, "detach"):
        return value
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_file_hashes(snapshot: Path, expected_commit: str, where: str) -> dict[str, str]:
    snapshot = snapshot.resolve()
    if snapshot.name.lower() != expected_commit.lower():
        raise CaptureContractError(f"{where} snapshot path is not bound to requested commit {expected_commit}")
    hashes: dict[str, str] = {}
    for path in sorted(snapshot.rglob("*"), key=lambda p: p.as_posix()):
        if path.is_dir():
            continue
        try:
            if not path.is_file():
                raise CaptureContractError(f"{where} snapshot contains a non-regular artifact: {path}")
            rel = path.relative_to(snapshot).as_posix()
            hashes[rel] = _sha256_file(path)
        except OSError as exc:
            raise CaptureContractError(f"unable to hash {where} snapshot artifact {path}: {exc}") from exc
    if not hashes:
        raise CaptureContractError(f"{where} snapshot contains no files")
    return hashes


def _sysctl_value(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        completed = subprocess.run(["sysctl", "-n", name], check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None
