from __future__ import annotations

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


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def valid_production_shape(request: dict) -> dict:
    return {
        "python_version": "3.13.0",
        "platform": "Linux",
        "torch_version": "2.9.0",
        "transformers_version": "4.56.0",
        "model_class": "Model",
        "tokenizer_class": "Tokenizer",
        "device": request["backend"]["device"],
        "tokenizers_version": None,
        "huggingface_hub_version": None,
        "attention_implementation": "eager",
        "cpu_machine": None,
        "cpu_processor": None,
        "cpu_instruction_flags": None,
        "omp_num_threads": None,
        "mkl_num_threads": None,
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
        "mps_mac_model": None,
        "mps_cpu_brand": None,
        "mps_macos_version": None,
        "mps_fallback_env": None,
        "mps_fast_math_env": None,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "cudnn_version": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "sdpa_flash_enabled": None,
        "sdpa_mem_efficient_enabled": None,
        "sdpa_math_enabled": None,
        "sdpa_cudnn_enabled": None,
        "mps_device_active": False,
        "mps_built": False,
        "mps_available": False,
        "autocast_disabled": True,
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
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
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
        self.assertIn('"--validate-only"', source)
        self.assertIn("validate_capture_request(request)", source)
        self.assertIn("if args.validate_only:", source)


if __name__ == "__main__":
    unittest.main()
