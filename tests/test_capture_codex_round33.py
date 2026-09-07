"""Round 33 regressions for the PR #4 exact-head review findings."""
from __future__ import annotations

import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend as policy_module
from qsol_geo_reason.capture_backend import HuggingFacePyTorchBackend as PolicyBackend
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as ProductionBackend
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from qsol_geo_reason.capture_runtime import _assert_torch_execution_surface
from test_capture_codex_round32 import synthetic_torch_module
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound33RegressionTests(unittest.TestCase):
    def test_pooling_and_token_dtype_alias_replacement_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "torch"
            package.mkdir()
            module, _native = synthetic_torch_module(package)
            float64 = object()
            int64 = object()
            module.float64 = module.double = float64
            module.long = module.int64 = int64
            _assert_torch_execution_surface(module)
            module.float64 = object()
            with self.assertRaisesRegex(CaptureContractError, r"torch\.float64"):
                _assert_torch_execution_surface(module)

    def test_live_seal_uses_external_construction_baseline(self):
        backend = object.__new__(PolicyBackend)
        float64, long = object(), object()
        backend._torch = types.SimpleNamespace(float64=float64, long=long)
        backend._live_state_seal_initialized = True
        backend._canonical_model_live_state = "caller-replaced-live"
        backend._canonical_model_content_state = "caller-replaced-content"
        backend._canonical_tokenizer_live_state = "caller-replaced-tokenizer"
        backend._model_live_state_seal = lambda: "original-live"
        backend._model_content_state_seal = lambda: "changed-content"
        backend._tokenizer_live_state_seal = lambda: "original-tokenizer"
        policy_module._remember_live_state_baseline(backend, (
            "original-live", "original-content", "original-tokenizer", float64, long
        ))
        with self.assertRaisesRegex(CaptureContractError, "tensor contents changed"):
            backend._assert_live_state_authentication()

    def test_cpu_math_dispatch_environment_is_frozen(self):
        backend = object.__new__(ProductionBackend)
        backend._device_type = "cpu"
        backend._canonical_cpu_math_environment_known = True
        backend._canonical_cpu_math_environment = {
            "onednn_max_cpu_isa": "AVX2",
            "dnnl_max_cpu_isa": None,
            "mkl_cbwr": "COMPATIBLE",
        }
        environment = {
            "ONEDNN_MAX_CPU_ISA": "AVX2",
            "MKL_CBWR": "COMPATIBLE",
        }
        with patch.dict(os.environ, environment, clear=True):
            backend._assert_cpu_math_environment_policy()
            os.environ["MKL_CBWR"] = "AUTO"
            with self.assertRaisesRegex(CaptureContractError, "math-library dispatch"):
                backend._assert_cpu_math_environment_policy()

    def test_native_tokenizer_receipt_is_required_when_active(self):
        request = fixture_request()
        observed = valid_production_shape(request)
        observed.update(
            tokenizers_native_backend_active=True,
            tokenizers_version="0.22.0",
            tokenizers_package_file_count=2,
            tokenizers_package_receipt_sha256="c" * 64,
        )
        _validate_production_metadata_shape(observed, request)
        observed["tokenizers_package_receipt_sha256"] = None
        with self.assertRaisesRegex(CaptureContractError, "Tokenizers package"):
            _validate_production_metadata_shape(observed, request)

    def test_cuda_capability_requires_numeric_major_minor(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda:0"
        request["determinism"]["mode"] = "best_effort"
        observed = valid_production_shape(request)
        observed.update(
            cuda_device_name="SYNTHETIC GPU",
            cuda_device_capability="unknown",
            cuda_resolved_device_index=0,
            float32_matmul_precision="highest",
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cuda_matmul_allow_fp16_reduced_precision_reduction=False,
            cuda_matmul_allow_bf16_reduced_precision_reduction=False,
        )
        with self.assertRaisesRegex(CaptureContractError, "numeric major.minor"):
            _validate_production_metadata_shape(observed, request)

    def test_schema_parity_for_token_bounds_cpu_fields_and_commits(self):
        trajectory = json.loads(
            (ROOT / "schemas/captured-trajectory.schema.json").read_text(encoding="utf-8")
        )
        observation_id = trajectory["$defs"]["observationTokenId"]
        self.assertEqual(observation_id["maximum"], 2**63 - 1)
        self.assertEqual(
            trajectory["allOf"][0]["if"]["properties"]["evidence_class"]["const"],
            "OBSERVATION",
        )

        manifest = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["$defs"]["gitSha"]["pattern"], "^[0-9a-f]{40}$")
        production = manifest["$defs"]["backendObservedProduction"]
        cpu_rule = next(
            rule for rule in production["allOf"]
            if rule.get("if", {}).get("properties", {}).get("device", {}).get("const") == "cpu"
        )
        for field in (
            "cpu_mkldnn_enabled", "cpu_mkldnn_matmul_fp32_precision",
            "cpu_math_dispatch_env_known", "onednn_max_cpu_isa", "dnnl_max_cpu_isa", "mkl_cbwr",
        ):
            self.assertEqual(cpu_rule["else"]["properties"][field], {"type": "null"})


if __name__ == "__main__":
    unittest.main()
