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
from .capture_common import (_BLOCK_CONTAINER_PATHS, _CAPTURE_PHASE, _PRODUCTION_BACKEND, _PRODUCTION_BACKEND_KEYS, _SIMULATION_BACKEND_KEYS, CaptureContractError, _require_exact_keys)

def _validate_backend_identity(observed: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if observed.get("name") != _PRODUCTION_BACKEND:
        raise CaptureContractError(f"backend metadata name must be {_PRODUCTION_BACKEND!r}")
    if observed.get("observed_model_commit") != request["model"]["revision"]:
        raise CaptureContractError("observed model commit does not match the frozen request revision")
    if observed.get("observed_tokenizer_commit") != request["model"]["tokenizer_revision"]:
        raise CaptureContractError("observed tokenizer commit does not match the frozen request revision")


def _validate_backend_metadata(observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str) -> None:
    required = _PRODUCTION_BACKEND_KEYS if evidence_class == "OBSERVATION" else _SIMULATION_BACKEND_KEYS
    _require_exact_keys(observed, required=set(required), where=f"{evidence_class.lower()} backend_observed")
    _validate_backend_identity(observed, request)
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
        production_constants = {
            "checkpoint_loading_clean": True, "quantization_config_present": False,
            "model_reports_quantized": False, "offloading": "none",
            "pool_accumulation_dtype": "float64", "pool_accumulation_device": "cpu",
            "hidden_state_capture_strategy": "selective_forward_hooks",
            "snapshot_authentication": "sha256_all_snapshot_files",
        }
        for key, value in production_constants.items():
            if observed.get(key) != value:
                raise CaptureContractError(f"observation backend provenance field {key} is invalid")
        for prefix in ("model", "tokenizer"):
            hashes = observed.get(f"{prefix}_snapshot_file_sha256")
            count = observed.get(f"{prefix}_snapshot_file_count")
            receipt = observed.get(f"{prefix}_snapshot_receipt_sha256")
            if not isinstance(hashes, dict) or not hashes:
                raise CaptureContractError(f"{prefix} snapshot artifact hashes are missing")
            if count != len(hashes):
                raise CaptureContractError(f"{prefix} snapshot file count does not match artifact hashes")
            if any(not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64 for path, digest in hashes.items()):
                raise CaptureContractError(f"{prefix} snapshot artifact hashes are malformed")
            if receipt != sha256_json(hashes):
                raise CaptureContractError(f"{prefix} snapshot receipt SHA-256 is invalid")


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
