from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_cli import main as capture_cli_main
from qsol_geo_reason.capture_execute import execute_capture
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from qsol_geo_reason.capture_validation import validate_capture_request

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
TEST_BUILD_CONFIG = "SYNTHETIC build fixture (not a measured runtime)\n"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def valid_production_shape(request: dict) -> dict:
    device = request["backend"]["device"]
    build_config = TEST_BUILD_CONFIG.rstrip("\n")
    if device == "mps":
        mps_extension = json.dumps(
            {
                "loaded_mps_runtime_libraries": {
                    "mps_runtime_library_file_count": 3,
                    "mps_runtime_library_receipt_sha256": "f" * 64,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        build_config += "\nQSOL_GEO_MPS_RUNTIME=" + mps_extension
    build_config += "\nQSOL_GEO_CPU_FLUSH_DENORMAL=false"
    cpu_extension = json.dumps(
        {
            "loaded_cpu_runtime_libraries": {
                "cpu_runtime_library_file_count": 0,
                "cpu_runtime_library_receipt_sha256": "d" * 64,
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    build_config += "\nQSOL_GEO_CPU_RUNTIME=" + cpu_extension
    if device.startswith("cuda:"):
        cuda_extension = json.dumps(
            {
                "loaded_cuda_runtime_libraries": {
                    "cuda_runtime_library_file_count": 1,
                    "cuda_runtime_library_receipt_sha256": "c" * 64,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        build_config += "\nQSOL_GEO_CUDA_RUNTIME=" + cuda_extension
    build_config += "\n"
    return {
        "python_version": "3.13.0",
        "platform": "Linux",
        "torch_version": "2.9.0",
        "transformers_version": "4.56.0",
        "torch_package_file_count": 2,
        "torch_package_receipt_sha256": "b" * 64,
        "transformers_package_file_count": 2,
        "transformers_package_receipt_sha256": "a" * 64,
        "huggingface_hub_package_file_count": 2,
        "huggingface_hub_package_receipt_sha256": "9" * 64,
        "tokenizers_native_backend_active": True,
        "tokenizers_package_file_count": 1,
        "tokenizers_package_receipt_sha256": "e" * 64,
        "safetensors_deserializer_active": False,
        "safetensors_package_file_count": None,
        "safetensors_package_receipt_sha256": None,
        "torch_build_config": build_config,
        "torch_build_config_sha256": hashlib.sha256(build_config.encode("utf-8")).hexdigest(),
        "model_class": "Model",
        "tokenizer_class": "Tokenizer",
        "device": device,
        "tokenizers_version": "0.22.0",
        "huggingface_hub_version": None,
        "attention_implementation": "eager",
        "cpu_machine": None,
        # CPU float64 pooling is part of every production device lane, including
        # CUDA/MPS source execution, so the pooling processor is always concrete.
        "cpu_processor": "SYNTHETIC CPU MODEL",
        "cpu_instruction_flags": None,
        "cpu_aten_capability": "DEFAULT",
        "aten_cpu_capability_env": None,
        "aten_cpu_capability_env_known": False,
        "cpu_math_dispatch_env_known": False if device == "cpu" else None,
        "onednn_max_cpu_isa": None,
        "dnnl_max_cpu_isa": None,
        "mkl_cbwr": None,
        "cuda_matmul_allow_fp16_accumulation": None,
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
        "nvidia_driver_version": None,
        "float32_matmul_precision": None,
        "nvidia_tf32_override": None,
        "torch_allow_tf32_cublas_override": None,
        "cublas_workspace_config": None,
        "mps_mac_model": "SYNTHETIC-MPS-MODEL" if device == "mps" else None,
        "mps_cpu_brand": None,
        "mps_macos_version": None,
        "mps_fallback_env": None,
        "mps_fast_math_env": None,
        "mps_prefer_metal_env": None,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "cudnn_version": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "cudnn_benchmark": False if device.startswith("cuda:") else None,
        "cudnn_deterministic": False if device.startswith("cuda:") else None,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": None,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": None,
        "sdpa_flash_enabled": None,
        "sdpa_mem_efficient_enabled": None,
        "sdpa_math_enabled": None,
        "sdpa_cudnn_enabled": None,
        "mps_device_active": device == "mps",
        "mps_built": device == "mps",
        "mps_available": device == "mps",
        "autocast_disabled": True,
        "deterministic_warn_only_enabled": False,
        "hidden_state_block_path": "layers",
        "hidden_state_count": max(request["capture"]["layers"]) + 2,
        "observed_hidden_state_dtypes": {
            str(layer): ["float32"] for layer in request["capture"]["layers"]
        },
    }


class CaptureRound6RegressionTests(unittest.TestCase):
    def test_production_hidden_state_count_and_block_path_are_validated(self):
        request = fixture_request()
        observed = valid_production_shape(request)
        _validate_production_metadata_shape(observed, request)

        bad_count = dict(observed)
        bad_count["hidden_state_count"] = "bogus"
        with self.assertRaisesRegex(CaptureContractError, "hidden_state_count"):
            _validate_production_metadata_shape(bad_count, request)

        too_small = dict(observed)
        too_small["hidden_state_count"] = max(request["capture"]["layers"])
        with self.assertRaisesRegex(CaptureContractError, "does not cover"):
            _validate_production_metadata_shape(too_small, request)

        bad_path = dict(observed)
        bad_path["hidden_state_block_path"] = "not.a.decoder.path"
        with self.assertRaisesRegex(CaptureContractError, "hidden_state_block_path"):
            _validate_production_metadata_shape(bad_path, request)

    def test_production_schema_typed_fields_fail_closed(self):
        request = fixture_request()
        observed = valid_production_shape(request)
        for field, bad in (
            ("python_version", 313),
            ("tokenizers_version", 1),
            ("huggingface_hub_package_file_count", True),
            ("huggingface_hub_package_receipt_sha256", "bad"),
            ("torch_num_threads", True),
            ("cuda_matmul_allow_tf32", "false"),
            ("mps_available", 1),
            ("observed_hidden_state_dtypes", []),
        ):
            mutated = dict(observed)
            mutated[field] = bad
            with self.subTest(field=field):
                with self.assertRaises(CaptureContractError):
                    _validate_production_metadata_shape(mutated, request)

    def test_backend_reuse_is_bound_to_applied_seed(self):
        request = fixture_request()
        backend = object.__new__(HuggingFacePyTorchBackend)
        # Synthetic construction binding; this fixture isolates request identity.
        backend._torch = backend._canonical_torch_module = object()
        backend._applied_seed = 17
        backend._determinism_mode = request["determinism"]["mode"]
        backend._model_identifier = request["model"]["identifier"]
        backend._model_revision = request["model"]["revision"]
        backend._tokenizer_identifier = request["model"]["tokenizer_identifier"]
        backend._tokenizer_revision = request["model"]["tokenizer_revision"]
        backend._device = request["backend"]["device"]
        backend._dtype_name = request["backend"]["dtype"]
        matching = json.loads(json.dumps(request))
        matching["determinism"]["seed"] = 17
        backend.assert_execution_request(matching)
        changed = json.loads(json.dumps(matching))
        changed["determinism"]["seed"] = 18
        with self.assertRaisesRegex(CaptureContractError, "applied seed"):
            backend.assert_execution_request(changed)
        self.assertIn(
            "backend.assert_execution_request(validated)",
            inspect.getsource(execute_capture),
        )

    def test_schema_declares_required_semantic_unique_step_validator(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        semantic = schema["x-qsol-semantic-validation"]
        self.assertIs(semantic["required"], True)
        self.assertIn("--validate-only", semantic["command"])
        self.assertIn("unique", semantic["constraints"][0])

        request = fixture_request()
        duplicate = dict(request["steps"][0])
        duplicate["text"] = "different text, same identifier"
        request["steps"].append(duplicate)
        with self.assertRaisesRegex(CaptureContractError, "duplicate step_id"):
            validate_capture_request(request)

    def test_capture_cli_exposes_no_model_validate_only_path(self):
        source = inspect.getsource(capture_cli_main)
        self.assertIn('\"--validate-only\"', source)
        self.assertIn("validate_capture_request(request)", source)
        self.assertIn("if args.validate_only:", source)


if __name__ == "__main__":
    unittest.main()
