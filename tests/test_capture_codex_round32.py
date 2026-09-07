"""Round 32 regressions for PyTorch identity, token domains, and schema parity."""
from __future__ import annotations

import inspect
import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.capture_common import (
    _TORCH_LONG_MAX,
    CaptureContractError,
    _validate_token_ids,
)
from qsol_geo_reason.capture_execute import _capture_steps
from qsol_geo_reason.capture_runtime import (
    _assert_torch_execution_surface,
    _torch_build_metadata,
    _validate_torch_build_metadata,
)
from qsol_geo_reason.capture_verify import verify_capture_bundle
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


def synthetic_torch_module(package_root: Path):
    module = types.ModuleType("torch")
    init = package_root / "__init__.py"
    init.write_text("# synthetic PyTorch package\n", encoding="utf-8")
    native = package_root / "libsynthetic.so"
    native.write_bytes(b"synthetic-native-kernel-v1")
    module.__file__ = str(init)
    module.__config__ = SimpleNamespace(show=lambda: "synthetic torch build\n")

    tensor = lambda *args, **kwargs: (args, kwargs)
    ones_like = lambda *args, **kwargs: (args, kwargs)
    zeros = lambda *args, **kwargs: (args, kwargs)
    isfinite = lambda *args, **kwargs: (args, kwargs)
    inference_mode = lambda *args, **kwargs: (args, kwargs)
    module.tensor = tensor
    module.ones_like = ones_like
    module.zeros = zeros
    module.isfinite = isfinite
    module.inference_mode = inference_mode
    module._C = SimpleNamespace(
        _VariableFunctions=SimpleNamespace(
            tensor=tensor,
            ones_like=ones_like,
            zeros=zeros,
            isfinite=isfinite,
        )
    )
    module.autograd = SimpleNamespace(
        grad_mode=SimpleNamespace(inference_mode=inference_mode)
    )
    return module, native


class CaptureRound32RegressionTests(unittest.TestCase):
    def test_pytorch_package_receipt_binds_python_and_native_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "torch"
            package.mkdir()
            module, native = synthetic_torch_module(package)
            first = _torch_build_metadata(module)
            self.assertGreaterEqual(first["torch_package_file_count"], 2)
            self.assertEqual(len(first["torch_package_receipt_sha256"]), 64)
            _validate_torch_build_metadata(first)

            native.write_bytes(b"synthetic-native-kernel-v2")
            second = _torch_build_metadata(module)
            self.assertNotEqual(
                first["torch_package_receipt_sha256"],
                second["torch_package_receipt_sha256"],
            )

        request = fixture_request()
        observed = valid_production_shape(request)
        _validate_torch_build_metadata(observed)
        for field, bad in (
            ("torch_package_file_count", 0),
            ("torch_package_receipt_sha256", "g" * 64),
        ):
            candidate = dict(observed)
            candidate[field] = bad
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "PyTorch package"):
                    _validate_torch_build_metadata(candidate)

        schema = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        production = schema["$defs"]["backendObservedProduction"]
        for field in ("torch_package_file_count", "torch_package_receipt_sha256"):
            self.assertIn(field, production["required"])
            self.assertIn(field, production["properties"])

    def test_pytorch_factory_surface_rejects_member_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "torch"
            package.mkdir()
            module, _native = synthetic_torch_module(package)
            _assert_torch_execution_surface(module)
            original = module.tensor
            module.tensor = lambda *args, **kwargs: None
            with self.assertRaisesRegex(CaptureContractError, r"torch\.tensor"):
                _assert_torch_execution_surface(module)
            module.tensor = original
            _assert_torch_execution_surface(module)

    def test_observation_token_ids_are_bounded_to_torch_long(self):
        self.assertEqual(
            _validate_token_ids([_TORCH_LONG_MAX], "test", max_value=_TORCH_LONG_MAX),
            [_TORCH_LONG_MAX],
        )
        with self.assertRaisesRegex(CaptureContractError, "supported integer domain"):
            _validate_token_ids([_TORCH_LONG_MAX + 1], "test", max_value=_TORCH_LONG_MAX)

        capture_source = inspect.getsource(_capture_steps)
        verify_source = inspect.getsource(verify_capture_bundle)
        self.assertIn("max_value=max_token_id", capture_source)
        self.assertIn("max_value=max_token_id", verify_source)
        self.assertIn('trajectory["evidence_class"] == "OBSERVATION"', verify_source)

    def test_manifest_schema_matches_runtime_device_policy_domains(self):
        schema = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        production = schema["$defs"]["backendObservedProduction"]
        rules = production["allOf"]

        mps_rule = next(
            rule for rule in rules
            if rule["if"].get("properties", {}).get("device", {}).get("const") == "mps"
        )
        self.assertIs(mps_rule["else"]["properties"]["mps_device_active"]["const"], False)

        cuda_rule = next(
            rule for rule in rules
            if "cuda_matmul_allow_fp16_reduced_precision_reduction"
            in rule.get("then", {}).get("properties", {})
        )
        cuda_then = cuda_rule["then"]["properties"]
        cuda_else = cuda_rule["else"]["properties"]
        self.assertEqual(
            cuda_then["float32_matmul_precision"]["enum"],
            ["highest", "high", "medium"],
        )
        for field in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
            self.assertEqual(cuda_then[field]["type"], "boolean")
        for field in (
            "float32_matmul_precision",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        ):
            self.assertEqual(cuda_else[field]["type"], "null")

        cpu_rule = next(
            rule for rule in rules
            if rule["if"].get("properties", {}).get("device", {}).get("const") == "cpu"
        )
        processor = cpu_rule["then"]["properties"]["cpu_processor"]
        generic = re.compile(processor["not"]["pattern"])
        for value in ("generic", "CPU", "x86_64", "AMD64", "arm64", "AaRcH64"):
            with self.subTest(value=value):
                self.assertIsNotNone(generic.fullmatch(value))
        for value in ("AMD Ryzen 9 5950X 16-Core Processor", "Intel(R) Xeon(R) Gold 6430"):
            with self.subTest(value=value):
                self.assertIsNone(generic.fullmatch(value))


if __name__ == "__main__":
    unittest.main()
