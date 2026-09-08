"""Round 47 regressions for tokenizer and execution-policy provenance."""
from __future__ import annotations

import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_provenance as provenance
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend
from qsol_geo_reason.capture_common import CaptureContractError, _PRODUCTION_BACKEND_KEYS


class CaptureRound47RegressionTests(unittest.TestCase):
    def test_impossible_non_native_tokenizer_observation_is_rejected(self):
        observed = {
            "python_version": "3.12", "platform": "test", "torch_version": "2.0",
            "transformers_version": "5.0", "model_class": "Model",
            "tokenizer_class": "Tokenizer", "device": "cpu",
            "tokenizers_native_backend_active": False, "tokenizers_version": None,
            "tokenizers_package_file_count": None,
            "tokenizers_package_receipt_sha256": None,
        }
        request = {"capture": {"layers": [0]}, "backend": {"device": "cpu"}, "determinism": {"mode": "required"}}
        with patch.object(provenance, "_validate_torch_build_metadata"), patch.object(provenance, "_validate_python_package_provenance"):
            with self.assertRaisesRegex(CaptureContractError, "requires tokenizers_native_backend_active=true"):
                provenance._validate_production_metadata_shape(observed, request)

    def test_warn_only_state_is_observed_and_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = types.SimpleNamespace(is_deterministic_algorithms_warn_only_enabled=lambda: True)
        self.assertIs(backend._deterministic_warn_only_state(), True)
        request = {"determinism": {"mode": "required"}}
        with self.assertRaisesRegex(CaptureContractError, "warn_only_enabled=false"):
            provenance._validate_required_determinism(
                {"deterministic_algorithms_enabled": True, "deterministic_warn_only_enabled": True},
                request,
            )
        provenance._validate_required_determinism(
            {"deterministic_algorithms_enabled": True, "deterministic_warn_only_enabled": False},
            request,
        )

    def test_schema_and_exact_key_contract_bind_new_receipts(self):
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "capture-run-manifest.schema.json"
        production = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        for field in ("mps_prefer_metal_env", "deterministic_warn_only_enabled"):
            self.assertIn(field, required)
            self.assertIn(field, _PRODUCTION_BACKEND_KEYS)
        self.assertEqual(production["properties"]["tokenizers_native_backend_active"], {"const": True})
        self.assertEqual(production["properties"]["deterministic_warn_only_enabled"], {"const": False})
        self.assertEqual(production["properties"]["tokenizers_package_file_count"]["type"], "integer")
        self.assertEqual(production["properties"]["tokenizers_package_receipt_sha256"]["$ref"], "#/$defs/sha256")

    def test_mps_metal_preference_is_rejected_by_verifier_shape(self):
        observed = {key: None for key in _PRODUCTION_BACKEND_KEYS}
        observed.update({
            "python_version": "3.12", "platform": "test", "torch_version": "2.0",
            "transformers_version": "5.0", "model_class": "Model", "tokenizer_class": "Tokenizer",
            "device": "mps", "tokenizers_native_backend_active": True,
            "tokenizers_version": "0.22", "tokenizers_package_file_count": 1,
            "tokenizers_package_receipt_sha256": "a" * 64,
            "safetensors_deserializer_active": True, "safetensors_package_file_count": 1,
            "safetensors_package_receipt_sha256": "b" * 64,
            "attention_implementation": "eager", "torch_num_threads": 1,
            "torch_num_interop_threads": 1, "mps_device_active": True,
            "mps_built": True, "mps_available": True, "autocast_disabled": True,
            "hidden_state_block_path": "layers", "hidden_state_count": 2,
            "observed_hidden_state_dtypes": {"0": ["float32"]},
            "mps_prefer_metal_env": "1",
        })
        request = {"capture": {"layers": [0]}, "backend": {"device": "mps"}, "determinism": {"mode": "required"}}
        with patch.object(provenance, "_validate_torch_build_metadata"), patch.object(provenance, "_validate_python_package_provenance"), patch.object(provenance, "_validate_dispatch_metadata"):
            with self.assertRaisesRegex(CaptureContractError, "Metal matmul preference"):
                provenance._validate_production_metadata_shape(observed, request)


if __name__ == "__main__":
    unittest.main()
