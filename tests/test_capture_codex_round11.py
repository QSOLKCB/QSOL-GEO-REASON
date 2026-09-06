from __future__ import annotations

import contextlib
import copy
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend,
    execute_capture,
    verify_capture_bundle,
)
from qsol_geo_reason.capture_provenance import _validate_backend_metadata

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
REV = "e" * 40


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class FakeBackend:
    def __init__(self, request: dict):
        self.request = request

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def hidden_states(self, input_ids, layer_indices, *, pool_span):
        start, end = pool_span
        output = {}
        for layer in layer_indices:
            rows = [
                [float(token + layer), float(token * (layer + 1))]
                for token in input_ids[start:end]
            ]
            output[layer] = {
                "vector": [
                    math.fsum(row[index] for row in rows) / len(rows)
                    for index in range(2)
                ],
                "vector_dimension": 2,
                "observed_dtype": "float64",
            }
        return output

    def metadata(self):
        return {
            "name": "ignored-by-simulation-canonicalization",
            "observed_model_commit": self.request["model"]["revision"],
            "observed_tokenizer_commit": self.request["model"]["tokenizer_revision"],
            "device": self.request["backend"]["device"],
            "dtype": self.request["backend"]["dtype"],
            "quantization": "none",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_cache": False,
            "capture_phase": "replayed_prefix",
            "kv_cache_reuse": False,
            "determinism_mode": self.request["determinism"]["mode"],
        }


def execute_simulation(request: dict | None = None):
    request = request or fixture_request()
    manifest, trajectory = execute_capture(
        request,
        implementation_revision=REV,
        backend=FakeBackend(request),
    )
    return request, manifest, trajectory


def rebind_bundle(manifest: dict, trajectory: dict) -> None:
    identity_fields = (
        "schema_version", "protocol_id", "run_id", "repository_commit",
        "request_sha256", "model", "backend_request", "backend_observed",
        "capture", "determinism", "generation_parameters",
    )
    run_manifest_id = sha256_json(
        {field: manifest[field] for field in identity_fields}
    )
    manifest["run_manifest_id"] = run_manifest_id
    trajectory["run_manifest_id"] = run_manifest_id
    trajectory["trajectory_sha256"] = sha256_json(
        {key: value for key, value in trajectory.items() if key != "trajectory_sha256"}
    )
    manifest["artifacts"]["captured_trajectory_sha256"] = trajectory["trajectory_sha256"]
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def production_observed(request: dict) -> dict:
    model_hashes = {"config.json": "0" * 64}
    tokenizer_hashes = {"tokenizer.json": "1" * 64}
    return {
        "name": "huggingface-pytorch",
        "python_version": "3.13.0",
        "platform": "Linux",
        "torch_version": "2.9.0",
        "transformers_version": "4.56.0",
        "tokenizers_version": None,
        "huggingface_hub_version": None,
        "model_class": "Model",
        "tokenizer_class": "Tokenizer",
        "observed_model_commit": request["model"]["revision"],
        "observed_tokenizer_commit": request["model"]["tokenizer_revision"],
        "checkpoint_loading_clean": True,
        "quantization_config_present": False,
        "model_reports_quantized": False,
        "attention_implementation": "eager",
        "device": "cpu",
        "cpu_machine": None,
        "cpu_processor": None,
        "cpu_instruction_flags": None,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "omp_num_threads": None,
        "mkl_num_threads": None,
        "cpu_mkldnn_enabled": None,
        "cpu_mkldnn_matmul_fp32_precision": None,
        "cuda_device_name": None,
        "cuda_device_capability": None,
        "cuda_resolved_device_index": None,
        "cuda_device_uuid": None,
        "cuda_visible_devices": None,
        "cuda_build_version": None,
        "cudnn_version": None,
        "nvidia_driver_version": None,
        "float32_matmul_precision": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": None,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": None,
        "sdpa_flash_enabled": None,
        "sdpa_mem_efficient_enabled": None,
        "sdpa_math_enabled": None,
        "sdpa_cudnn_enabled": None,
        "nvidia_tf32_override": None,
        "torch_allow_tf32_cublas_override": None,
        "cublas_workspace_config": None,
        "mps_device_active": False,
        "mps_built": False,
        "mps_available": False,
        "mps_mac_model": None,
        "mps_cpu_brand": None,
        "mps_macos_version": None,
        "mps_fallback_env": None,
        "mps_fast_math_env": None,
        "autocast_disabled": True,
        "dtype": request["backend"]["dtype"],
        "observed_hidden_state_dtypes": {
            str(layer): ["float64"] for layer in request["capture"]["layers"]
        },
        "pool_accumulation_dtype": "float64",
        "pool_accumulation_device": "cpu",
        "hidden_state_capture_strategy": "selective_forward_hooks",
        "hidden_state_block_path": "layers",
        "hidden_state_count": max(request["capture"]["layers"]) + 2,
        "snapshot_authentication": "sha256_all_snapshot_files_pre_and_post_load",
        "model_snapshot_file_count": 1,
        "model_snapshot_file_sha256": model_hashes,
        "model_snapshot_receipt_sha256": sha256_json(model_hashes),
        "tokenizer_snapshot_file_count": 1,
        "tokenizer_snapshot_file_sha256": tokenizer_hashes,
        "tokenizer_snapshot_receipt_sha256": sha256_json(tokenizer_hashes),
        "quantization": "none",
        "offloading": "none",
        "local_files_only": True,
        "trust_remote_code": False,
        "use_cache": False,
        "capture_phase": "replayed_prefix",
        "kv_cache_reuse": False,
        "deterministic_algorithms_enabled": True,
        "determinism_mode": request["determinism"]["mode"],
    }


class HookHandle:
    def __init__(self, hooks: list, hook):
        self.hooks = hooks
        self.hook = hook
        self.removed = False

    def remove(self):
        if not self.removed:
            self.hooks.remove(self.hook)
            self.removed = True


class FakeBlock:
    def __init__(self):
        self.pre_hooks: list = []

    def register_forward_pre_hook(self, hook, *, with_kwargs=False):
        self.pre_hooks.append(hook)
        return HookHandle(self.pre_hooks, hook)


class FakeBaseModel:
    def __init__(self, block: FakeBlock):
        self.block = block
        self.forward_hooks: list = []

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)
        return HookHandle(self.forward_hooks, hook)

    def __call__(self, **kwargs):
        hidden = object()
        for hook in list(self.block.pre_hooks):
            hook(self.block, (hidden,), {})
        output = object()
        for hook in list(self.forward_hooks):
            hook(self, (), output)
        return output


class FakeTorch:
    long = object()

    def __init__(self):
        self.fail_next_tensor = True

    def is_autocast_enabled(self, *_args):
        return False

    def tensor(self, *_args, **_kwargs):
        if self.fail_next_tensor:
            self.fail_next_tensor = False
            raise RuntimeError("synthetic allocation failure")
        return object()

    @staticmethod
    def ones_like(_value):
        return object()

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class CaptureRound11RegressionTests(unittest.TestCase):
    def test_boolean_metadata_numeric_substitutes_are_rejected_after_rehash(self):
        for field, numeric in (
            ("local_files_only", 1),
            ("local_files_only", 1.0),
            ("use_cache", 0),
            ("use_cache", 0.0),
        ):
            request, manifest, trajectory = execute_simulation()
            manifest = copy.deepcopy(manifest)
            trajectory = copy.deepcopy(trajectory)
            manifest["backend_observed"][field] = numeric
            rebind_bundle(manifest, trajectory)
            with self.subTest(field=field, numeric=numeric):
                with self.assertRaisesRegex(CaptureContractError, "must be boolean"):
                    verify_capture_bundle(request, manifest, trajectory)

    def test_production_boolean_constants_reject_integer_substitutes(self):
        request = fixture_request()
        observed = production_observed(request)
        _validate_backend_metadata(observed, request, "OBSERVATION")
        for field, numeric in (
            ("checkpoint_loading_clean", 1),
            ("quantization_config_present", 0),
            ("model_reports_quantized", 0.0),
            ("autocast_disabled", 1.0),
        ):
            mutated = dict(observed)
            mutated[field] = numeric
            with self.subTest(field=field, numeric=numeric):
                with self.assertRaisesRegex(CaptureContractError, "must be boolean"):
                    _validate_backend_metadata(mutated, request, "OBSERVATION")

    def test_request_bound_nested_values_use_type_strict_json_equality(self):
        request, manifest, trajectory = execute_simulation()
        manifest = copy.deepcopy(manifest)
        trajectory = copy.deepcopy(trajectory)
        manifest["backend_request"]["local_files_only"] = 1
        rebind_bundle(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "backend_request does not match request"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_hook_cleanup_covers_allocation_failure_and_backend_retry(self):
        block = FakeBlock()
        base_model = FakeBaseModel(block)
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeTorch()
        backend._device = "cpu"
        backend._device_type = "cpu"
        backend._attention_implementation = "eager"
        backend._hidden_state_count = 2
        backend._blocks = [block]
        backend._base_model = base_model
        backend._observed_hidden_state_dtypes = {}
        backend._pool_tensor_record = lambda _tensor, **_kwargs: {
            "vector": [1.0],
            "vector_dimension": 1,
            "observed_dtype": "float32",
        }

        with self.assertRaisesRegex(RuntimeError, "allocation failure"):
            backend.hidden_states([1], [0], pool_span=(0, 1))
        self.assertEqual(block.pre_hooks, [])
        self.assertEqual(base_model.forward_hooks, [])

        result = backend.hidden_states([1], [0], pool_span=(0, 1))
        self.assertEqual(set(result), {0})
        self.assertEqual(block.pre_hooks, [])
        self.assertEqual(base_model.forward_hooks, [])

    def test_only_canonical_attention_implementations_are_accepted(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        for allowed in ("eager", "sdpa"):
            backend._attention_implementation = allowed
            backend._assert_attention_implementation()
        for rejected in ("flash_attention_2", "flash_attention_3", None, "xformers"):
            backend._attention_implementation = rejected
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(CaptureContractError, "attention implementation"):
                    backend._assert_attention_implementation()

    def test_cuda_reduced_precision_reductions_are_forced_and_verified(self):
        matmul = SimpleNamespace(
            allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,
        )
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul))
        )
        backend._force_cuda_reduced_precision_policy()
        self.assertIs(matmul.allow_fp16_reduced_precision_reduction, False)
        self.assertIs(matmul.allow_bf16_reduced_precision_reduction, False)
        self.assertEqual(
            backend._assert_cuda_reduced_precision_policy(),
            {"fp16": False, "bf16": False},
        )
        matmul.allow_fp16_reduced_precision_reduction = True
        with self.assertRaisesRegex(CaptureContractError, "reduced-precision"):
            backend._assert_cuda_reduced_precision_policy()

    def test_manifest_schema_binds_attention_and_cuda_reduction_policy(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        properties = production["properties"]
        self.assertEqual(
            properties["attention_implementation"]["enum"],
            ["eager", "sdpa"],
        )
        for field in (
            "cuda_matmul_allow_fp16_reduced_precision_reduction",
            "cuda_matmul_allow_bf16_reduced_precision_reduction",
        ):
            self.assertIn(field, required)
            self.assertEqual(properties[field]["$ref"], "#/$defs/nullableBoolean")


if __name__ == "__main__":
    unittest.main()
