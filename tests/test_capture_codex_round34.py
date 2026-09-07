"""Round 34 regressions for PR #4 provenance and observation-boundary findings."""
from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture_backend_audit import (
    HuggingFacePyTorchBackend as AuditBackend,
)
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound34RegressionTests(unittest.TestCase):
    def test_sentencepiece_backed_slow_tokenizer_is_rejected(self):
        processor_type = type("SentencePieceProcessor", (), {})
        processor_type.__module__ = "sentencepiece"
        tokenizer = types.SimpleNamespace(sp_model=processor_type())
        backend = object.__new__(AuditBackend)
        backend._tokenizer = tokenizer
        backend._tokenizers_package_provenance_initialized = False
        backend._tokenizers_native_backend_active = False
        backend._tokenizers_package_provenance = None

        self.assertEqual(
            backend._native_slow_tokenizer_backend(tokenizer), "sentencepiece"
        )
        with self.assertRaisesRegex(CaptureContractError, "native slow-tokenizer"):
            backend._initialize_tokenizers_package_provenance()

    def test_snapshot_receipts_are_bound_outside_instance_state(self):
        backend = object.__new__(AuditBackend)
        backend._snapshot_provenance_baseline_initialized = False
        backend._model_snapshot_hashes = {"model.bin": "a" * 64}
        backend._tokenizer_snapshot_hashes = {"tokenizer.json": "b" * 64}
        backend._initialize_snapshot_provenance_baseline()
        backend._assert_snapshot_provenance_baseline()

        backend._model_snapshot_hashes = {"model.bin": "c" * 64}
        with self.assertRaisesRegex(CaptureContractError, "snapshot provenance changed"):
            backend._assert_snapshot_provenance_baseline()

    def test_safetensors_receipt_is_required_when_snapshot_uses_it(self):
        request = fixture_request()
        observed = valid_production_shape(request)
        observed.update(
            safetensors_deserializer_active=True,
            safetensors_package_file_count=3,
            safetensors_package_receipt_sha256="d" * 64,
            model_snapshot_file_sha256={"model.safetensors": "e" * 64},
        )
        _validate_production_metadata_shape(observed, request)

        observed["safetensors_deserializer_active"] = False
        observed["safetensors_package_file_count"] = None
        observed["safetensors_package_receipt_sha256"] = None
        with self.assertRaisesRegex(CaptureContractError, "activation does not match"):
            _validate_production_metadata_shape(observed, request)

    def test_python_trace_and_profile_callbacks_are_rejected(self):
        with patch.object(sys, "gettrace", return_value=object()), patch.object(
            sys, "getprofile", return_value=None
        ):
            with self.assertRaisesRegex(CaptureContractError, "trace/profile"):
                AuditBackend._assert_no_active_python_instrumentation()
        with patch.object(sys, "gettrace", return_value=None), patch.object(
            sys, "getprofile", return_value=object()
        ):
            with self.assertRaisesRegex(CaptureContractError, "trace/profile"):
                AuditBackend._assert_no_active_python_instrumentation()

        production_start = inspect.getsource(ProductionBackend.begin_observation)
        audit_modes = inspect.getsource(AuditBackend._assert_no_active_torch_override_modes)
        self.assertIn("self._assert_no_active_torch_override_modes()", production_start)
        self.assertIn("self._assert_no_active_python_instrumentation()", audit_modes)

    def test_manifest_schema_requires_layer_dtype_entries(self):
        schema = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        dtype_map = production["properties"]["observed_hidden_state_dtypes"]
        self.assertEqual(dtype_map["minProperties"], 1)
        self.assertEqual(
            dtype_map["propertyNames"]["pattern"], "^(?:0|[1-9][0-9]*)$"
        )
        self.assertIn("capture.layers exactly", dtype_map["description"])

    def test_manifest_schema_binds_cpu_unknown_and_safetensors_state(self):
        schema = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        cpu_rule = next(
            rule for rule in production["allOf"]
            if rule.get("if", {}).get("properties", {}).get("device", {}).get("const")
            == "cpu"
        )
        self.assertEqual(
            cpu_rule["then"]["properties"]["cpu_math_dispatch_env_known"],
            {"type": "boolean"},
        )
        unknown_rule = next(
            rule for rule in production["allOf"]
            if rule.get("if", {})
            .get("properties", {})
            .get("cpu_math_dispatch_env_known", {})
            .get("const")
            is False
        )
        for field in ("onednn_max_cpu_isa", "dnnl_max_cpu_isa", "mkl_cbwr"):
            self.assertEqual(
                unknown_rule["then"]["properties"][field], {"type": "null"}
            )

        safetensors_rule = next(
            rule for rule in production["allOf"]
            if rule.get("if", {})
            .get("properties", {})
            .get("safetensors_deserializer_active", {})
            .get("const")
            is True
        )
        self.assertEqual(
            safetensors_rule["then"]["properties"]["safetensors_package_file_count"],
            {"type": "integer", "minimum": 1},
        )


if __name__ == "__main__":
    unittest.main()
