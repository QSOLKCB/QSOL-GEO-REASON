"""Concrete local Hugging Face/PyTorch backend for GEO-CAP-001."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .capture_common import _CAPTURE_PHASE, _PRODUCTION_BACKEND, CaptureBackendUnavailable, CaptureContractError
from .capture_validation import _quantization_reasons, _validate_loading_info, validate_capture_request
from .capture_provenance import (
    _cpu_hardware_metadata, _extract_hidden_tensor, _resolve_hidden_state_layout,
    _snapshot_file_hashes, _sysctl_value,
)


class HuggingFacePyTorchBackend:
    """Direct local-only Hugging Face / PyTorch replay backend."""

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise CaptureBackendUnavailable("canonical capture requires optional capture dependencies; install qsol-geo-reason[capture]") from exc
        self._torch = torch
        self._transformers = transformers
        self._observed_hidden_state_dtypes: dict[int, set[str]] = {}
        self._last_sdpa_policy: dict[str, bool | None] | None = None
        backend, model_cfg, determinism = validated["backend"], validated["model"], validated["determinism"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self._device = backend["device"]
        self._device_type = self._device.split(":", 1)[0]
        self._dtype_name = backend["dtype"]
        self._determinism_mode = determinism["mode"]
        self._applied_seed = determinism["seed"]
        self._model_identifier = model_cfg["identifier"]
        self._model_revision = model_cfg["revision"]
        self._tokenizer_identifier = model_cfg["tokenizer_identifier"]
        self._tokenizer_revision = model_cfg["tokenizer_revision"]
        self._cuda_resolved_device_index: int | None = None
        if self._device_type == "cuda":
            self._cuda_resolved_device_index = int(self._device.split(":", 1)[1])
            if not torch.cuda.is_available():
                raise CaptureContractError("canonical CUDA capture requested but CUDA is unavailable")
            if self._cuda_resolved_device_index >= int(torch.cuda.device_count()):
                raise CaptureContractError(
                    f"canonical CUDA device index {self._cuda_resolved_device_index} is outside available range"
                )
        self._assert_mps_execution_policy()
        self._assert_autocast_disabled()

        torch.manual_seed(self._applied_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._applied_seed)
        if determinism["mode"] == "required":
            torch.use_deterministic_algorithms(True)

        model_snapshot = Path(snapshot_download(repo_id=model_cfg["identifier"], revision=model_cfg["revision"], local_files_only=True))
        tokenizer_snapshot = Path(snapshot_download(repo_id=model_cfg["tokenizer_identifier"], revision=model_cfg["tokenizer_revision"], local_files_only=True))
        model_hashes_before = _snapshot_file_hashes(model_snapshot, model_cfg["revision"], "model")
        tokenizer_hashes_before = _snapshot_file_hashes(tokenizer_snapshot, model_cfg["tokenizer_revision"], "tokenizer")

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_snapshot), local_files_only=True, trust_remote_code=False)
        loaded = AutoModelForCausalLM.from_pretrained(
            str(model_snapshot), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype_map[backend["dtype"]], output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise CaptureContractError("Transformers did not return (model, loading_info) for canonical load")
        self._model, loading_info = loaded
        _validate_loading_info(loading_info)
        quantization = _quantization_reasons(self._model)
        if quantization:
            raise CaptureContractError("canonical capture forbids checkpoint/config quantization; detected: " + ", ".join(quantization))

        model_hashes_after = _snapshot_file_hashes(model_snapshot, model_cfg["revision"], "model")
        tokenizer_hashes_after = _snapshot_file_hashes(tokenizer_snapshot, model_cfg["tokenizer_revision"], "tokenizer")
        if model_hashes_before != model_hashes_after:
            raise CaptureContractError("model snapshot changed while canonical checkpoint was loading")
        if tokenizer_hashes_before != tokenizer_hashes_after:
            raise CaptureContractError("tokenizer snapshot changed while canonical tokenizer was loading")
        self._model_snapshot_hashes = model_hashes_after
        self._tokenizer_snapshot_hashes = tokenizer_hashes_after
        self._base_model, self._block_path, self._blocks = _resolve_hidden_state_layout(self._model)
        self._hidden_state_count = len(self._blocks) + 1
        self._checkpoint_loading_clean = True
        self._attention_implementation = getattr(self._model.config, "_attn_implementation", None)
        if self._device_type == "cuda" and self._attention_implementation == "sdpa":
            self._force_sdpa_math_policy()
        self._model.to(self._device)
        self._model.eval()

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        """Refuse reuse when construction-bound request identity differs."""
        determinism = request["determinism"]
        if determinism["seed"] != self._applied_seed:
            raise CaptureContractError(
                f"backend applied seed {self._applied_seed} does not match execution request seed {determinism['seed']}"
            )
        if determinism["mode"] != self._determinism_mode:
            raise CaptureContractError("backend determinism mode does not match execution request")
        model_cfg = request["model"]
        expected_model = (
            self._model_identifier, self._model_revision,
            self._tokenizer_identifier, self._tokenizer_revision,
        )
        requested_model = (
            model_cfg["identifier"], model_cfg["revision"],
            model_cfg["tokenizer_identifier"], model_cfg["tokenizer_revision"],
        )
        if requested_model != expected_model:
            raise CaptureContractError("backend model/tokenizer identity does not match execution request")
        backend = request["backend"]
        if backend["device"] != self._device or backend["dtype"] != self._dtype_name:
            raise CaptureContractError("backend device/dtype does not match execution request")

    @staticmethod
    def _env_flag_enabled(value: str | None) -> bool:
        return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}

    def _assert_mps_execution_policy(self) -> None:
        if self._device_type != "mps":
            return
        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        fast_math = os.environ.get("PYTORCH_MPS_FAST_MATH")
        if self._env_flag_enabled(fallback):
            raise CaptureContractError("canonical MPS capture forbids PYTORCH_ENABLE_MPS_FALLBACK")
        if self._env_flag_enabled(fast_math):
            raise CaptureContractError("canonical MPS capture forbids PYTORCH_MPS_FAST_MATH")

    def _autocast_enabled(self) -> bool:
        torch = self._torch
        checker = getattr(torch, "is_autocast_enabled", None)
        if callable(checker):
            try:
                return bool(checker(self._device_type))
            except TypeError:
                if self._device_type == "cuda":
                    return bool(checker())
        if self._device_type == "cpu":
            legacy = getattr(torch, "is_autocast_cpu_enabled", None)
            if callable(legacy):
                return bool(legacy())
        return False

    def _assert_autocast_disabled(self) -> None:
        if self._autocast_enabled():
            raise CaptureContractError("canonical capture forbids ambient torch autocast")

    def _sdpa_policy_state(self) -> dict[str, bool | None]:
        cuda_backend = getattr(self._torch.backends, "cuda", None)
        if cuda_backend is None:
            raise CaptureContractError("CUDA SDPA backend controls are unavailable")
        result: dict[str, bool | None] = {}
        for key, name in (
            ("flash", "flash_sdp_enabled"),
            ("mem_efficient", "mem_efficient_sdp_enabled"),
            ("math", "math_sdp_enabled"),
            ("cudnn", "cudnn_sdp_enabled"),
        ):
            query = getattr(cuda_backend, name, None)
            if key != "cudnn" and not callable(query):
                raise CaptureContractError(f"canonical CUDA SDPA policy requires torch.backends.cuda.{name}")
            result[key] = bool(query()) if callable(query) else None
        return result

    def _assert_sdpa_math_policy(self) -> dict[str, bool | None]:
        state = self._sdpa_policy_state()
        if state["flash"] is not False or state["mem_efficient"] is not False or state["math"] is not True:
            raise CaptureContractError(f"canonical CUDA SDPA policy drifted from math-only state: {state!r}")
        if state["cudnn"] is True:
            raise CaptureContractError("canonical CUDA SDPA policy requires cuDNN SDPA disabled")
        return state

    def _force_sdpa_math_policy(self) -> None:
        """Force canonical CUDA SDPA policy to math-only for reproducible capture."""
        cuda_backend = getattr(self._torch.backends, "cuda", None)
        if cuda_backend is None:
            raise CaptureContractError("CUDA SDPA backend controls are unavailable")
        required_toggles = (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        )
        for name, enabled in required_toggles:
            toggle = getattr(cuda_backend, name, None)
            if not callable(toggle):
                raise CaptureContractError(f"canonical CUDA SDPA policy requires torch.backends.cuda.{name}")
            toggle(enabled)
        cudnn_toggle = getattr(cuda_backend, "enable_cudnn_sdp", None)
        if callable(cudnn_toggle):
            cudnn_toggle(False)
        self._last_sdpa_policy = self._assert_sdpa_math_policy()

    def tokenize(self, text: str) -> list[int]:
        encoded = self._tokenizer(text, add_special_tokens=True, return_attention_mask=False)
        return [int(v) for v in encoded["input_ids"]]

    def _pool_tensor_record(self, tensor: Any, *, layer_index: int, token_count: int, pool_span: tuple[int, int]) -> Mapping[str, Any]:
        torch = self._torch
        if tensor is None:
            raise CaptureContractError(f"selective hook for layer {layer_index} produced no tensor")
        if tensor.ndim == 3:
            if int(tensor.shape[0]) != 1:
                raise CaptureContractError(f"layer {layer_index} hidden-state batch dimension must be 1")
            matrix = tensor[0]
        elif tensor.ndim == 2:
            matrix = tensor
        else:
            raise CaptureContractError(f"layer {layer_index} hidden-state rank {tensor.ndim} is unsupported")
        if int(matrix.shape[0]) != token_count or int(matrix.shape[1]) < 1:
            raise CaptureContractError(f"layer {layer_index} hidden-state shape is incompatible with token capture")
        start, end = pool_span
        observed_dtype = str(matrix.dtype).removeprefix("torch.")
        self._observed_hidden_state_dtypes.setdefault(layer_index, set()).add(observed_dtype)
        dimension = int(matrix.shape[1])
        if end - start == 1:
            pooled = matrix[start].detach().to(device="cpu").to(dtype=torch.float64)
        else:
            accumulator = torch.zeros(dimension, dtype=torch.float64, device="cpu")
            for chunk_start in range(start, end, 256):
                chunk = matrix[chunk_start:min(end, chunk_start + 256)].detach().to(device="cpu").to(dtype=torch.float64)
                accumulator.add_(chunk.sum(dim=0, dtype=torch.float64))
            pooled = accumulator / (end - start)
        if not bool(torch.isfinite(pooled).all().item()):
            raise CaptureContractError(f"layer {layer_index} pooled representation contains non-finite values")
        return {"vector": pooled.tolist(), "vector_dimension": dimension, "observed_dtype": observed_dtype}

    def hidden_states(self, input_ids: Sequence[int], layer_indices: Sequence[int], *, pool_span: tuple[int, int]) -> Mapping[int, Mapping[str, Any]]:
        """Capture requested states from the base model without LM-head logits."""
        torch = self._torch
        requested = tuple(layer_indices)
        if any(i < 0 or i >= self._hidden_state_count for i in requested):
            bad = next(i for i in requested if i < 0 or i >= self._hidden_state_count)
            raise CaptureContractError(f"requested layer {bad} outside backend hidden-state range [0, {self._hidden_state_count - 1}]")
        self._assert_mps_execution_policy()
        self._assert_autocast_disabled()
        if self._device_type == "cuda" and self._attention_implementation == "sdpa":
            self._force_sdpa_math_policy()
        selected: dict[int, Mapping[str, Any]] = {}
        handles: list[Any] = []
        token_count = len(input_ids)

        def capture(layer_index: int, value: Any) -> None:
            if layer_index in selected:
                raise CaptureContractError(f"selective hidden-state hook for layer {layer_index} fired more than once")
            selected[layer_index] = self._pool_tensor_record(_extract_hidden_tensor(value), layer_index=layer_index, token_count=token_count, pool_span=pool_span)

        def make_pre_hook(layer_index: int):
            def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
                capture(layer_index, kwargs.get("hidden_states") if kwargs.get("hidden_states") is not None else (args[0] if args else None))
            return hook

        for layer_index in requested:
            if layer_index < len(self._blocks):
                handles.append(self._blocks[layer_index].register_forward_pre_hook(make_pre_hook(layer_index), with_kwargs=True))
        if len(self._blocks) in requested:
            final_index = len(self._blocks)
            def final_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                capture(final_index, output)
            handles.append(self._base_model.register_forward_hook(final_hook))

        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self._device)
        mask = torch.ones_like(ids)
        try:
            with torch.inference_mode():
                self._base_model(
                    input_ids=ids, attention_mask=mask, output_hidden_states=False,
                    use_cache=False, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if self._device_type == "cuda" and self._attention_implementation == "sdpa":
            self._last_sdpa_policy = self._assert_sdpa_math_policy()
        if set(selected) != set(requested):
            raise CaptureContractError(f"selective hidden-state hooks did not capture requested layers: {sorted(set(requested) - set(selected))}")
        return selected

    @staticmethod
    def _installed_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _nvidia_driver_version() -> str | None:
        try:
            completed = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
        return ",".join(values) if completed.returncode == 0 and values else None

    def metadata(self) -> Mapping[str, Any]:
        torch = self._torch
        config = self._model.config
        model_commit = getattr(config, "_commit_hash", None) or Path(getattr(self._model, "name_or_path", "")).name
        tokenizer_commit = getattr(self._tokenizer, "_commit_hash", None) or self._tokenizer.init_kwargs.get("_commit_hash") or Path(getattr(self._tokenizer, "name_or_path", "")).name
        cuda_active = self._device_type == "cuda" and torch.cuda.is_available()
        mps_active = self._device_type == "mps"
        cuda_device = cuda_capability = cuda_uuid = None
        if cuda_active and self._cuda_resolved_device_index is not None:
            try:
                index = self._cuda_resolved_device_index
                cuda_device = torch.cuda.get_device_name(index)
                cap = torch.cuda.get_device_capability(index)
                cuda_capability = f"{cap[0]}.{cap[1]}"
                properties = torch.cuda.get_device_properties(index)
                raw_uuid = getattr(properties, "uuid", None)
                cuda_uuid = str(raw_uuid) if raw_uuid is not None else None
            except Exception:
                pass
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = None
        try:
            matmul_precision = torch.get_float32_matmul_precision()
        except Exception:
            matmul_precision = None
        try:
            cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        except Exception:
            cuda_tf32 = None
        try:
            cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        except Exception:
            cudnn_tf32 = None
        mps_backend = getattr(torch.backends, "mps", None)
        try:
            mps_built = bool(mps_backend.is_built()) if mps_backend is not None else False
            mps_available = bool(mps_backend.is_available()) if mps_backend is not None else False
        except Exception:
            mps_built = mps_available = False
        model_hashes = dict(sorted(self._model_snapshot_hashes.items()))
        tokenizer_hashes = dict(sorted(self._tokenizer_snapshot_hashes.items()))
        sdpa = self._last_sdpa_policy or {"flash": None, "mem_efficient": None, "math": None, "cudnn": None}
        return {
            "name": _PRODUCTION_BACKEND,
            "python_version": sys.version.split()[0], "platform": platform.platform(),
            "torch_version": torch.__version__, "transformers_version": self._transformers.__version__,
            "tokenizers_version": self._installed_version("tokenizers"), "huggingface_hub_version": self._installed_version("huggingface-hub"),
            "model_class": type(self._model).__name__, "tokenizer_class": type(self._tokenizer).__name__,
            "observed_model_commit": model_commit, "observed_tokenizer_commit": tokenizer_commit,
            "checkpoint_loading_clean": self._checkpoint_loading_clean,
            "quantization_config_present": getattr(config, "quantization_config", None) is not None,
            "model_reports_quantized": bool(getattr(self._model, "is_quantized", False)),
            "attention_implementation": self._attention_implementation,
            "device": str(self._device), **_cpu_hardware_metadata(torch),
            "cuda_device_name": cuda_device, "cuda_device_capability": cuda_capability,
            "cuda_resolved_device_index": self._cuda_resolved_device_index,
            "cuda_device_uuid": cuda_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_build_version": getattr(torch.version, "cuda", None), "cudnn_version": cudnn_version,
            "nvidia_driver_version": self._nvidia_driver_version() if cuda_active else None,
            "float32_matmul_precision": matmul_precision, "cuda_matmul_allow_tf32": cuda_tf32,
            "cudnn_allow_tf32": cudnn_tf32,
            "sdpa_flash_enabled": sdpa["flash"], "sdpa_mem_efficient_enabled": sdpa["mem_efficient"],
            "sdpa_math_enabled": sdpa["math"], "sdpa_cudnn_enabled": sdpa["cudnn"],
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "torch_allow_tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "mps_device_active": mps_active, "mps_built": mps_built, "mps_available": mps_available,
            "mps_mac_model": _sysctl_value("hw.model") if mps_active else None,
            "mps_cpu_brand": _sysctl_value("machdep.cpu.brand_string") if mps_active else None,
            "mps_macos_version": (platform.mac_ver()[0] or None) if mps_active else None,
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_fast_math_env": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "autocast_disabled": True,
            "dtype": self._dtype_name,
            "observed_hidden_state_dtypes": {str(k): sorted(v) for k, v in sorted(self._observed_hidden_state_dtypes.items())},
            "pool_accumulation_dtype": "float64", "pool_accumulation_device": "cpu",
            "hidden_state_capture_strategy": "selective_forward_hooks", "hidden_state_block_path": self._block_path,
            "hidden_state_count": self._hidden_state_count,
            "snapshot_authentication": "sha256_all_snapshot_files_pre_and_post_load",
            "model_snapshot_file_count": len(model_hashes), "model_snapshot_file_sha256": model_hashes,
            "model_snapshot_receipt_sha256": sha256_json(model_hashes),
            "tokenizer_snapshot_file_count": len(tokenizer_hashes), "tokenizer_snapshot_file_sha256": tokenizer_hashes,
            "tokenizer_snapshot_receipt_sha256": sha256_json(tokenizer_hashes),
            "quantization": "none", "offloading": "none", "local_files_only": True,
            "trust_remote_code": False, "use_cache": False, "capture_phase": _CAPTURE_PHASE,
            "kv_cache_reuse": False, "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "determinism_mode": self._determinism_mode,
        }
