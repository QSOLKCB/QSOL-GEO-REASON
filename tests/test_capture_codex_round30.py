"""Round 30 regressions for package provenance, CPU identity, and PyTorch modes."""
from __future__ import annotations

import inspect
import json
import py_compile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_hardware import _is_concrete_cpu_identity
from qsol_geo_reason.capture_package import _python_package_provenance
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound30RegressionTests(unittest.TestCase):
    def test_transformers_package_receipt_binds_imported_package_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "transformers"
            package.mkdir()
            init = package / "__init__.py"
            model = package / "modeling_fixture.py"
            init.write_text('__version__ = "4.56.0"\n', encoding="utf-8")
            model.write_text("def forward(x): return x + 1\n", encoding="utf-8")
            module = SimpleNamespace(__file__=str(init), __version__="4.56.0")

            first = _python_package_provenance(module, "Transformers")
            self.assertEqual(first["file_count"], 2)
            self.assertEqual(len(first["receipt_sha256"]), 64)

            # Keep the advertised version identical while changing executable source.
            model.write_text("def forward(x): return x + 2\n", encoding="utf-8")
            second = _python_package_provenance(module, "Transformers")
            self.assertEqual(module.__version__, "4.56.0")
            self.assertNotEqual(first["receipt_sha256"], second["receipt_sha256"])

            # Only source-equivalent caches may be omitted from package identity.
            cache = Path(py_compile.compile(str(model), doraise=True, optimize=0))
            self.assertEqual(
                _python_package_provenance(module, "Transformers"),
                second,
            )
            cache.write_bytes(b"unverifiable bytecode")
            with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                _python_package_provenance(module, "Transformers")

        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        for field in (
            "transformers_package_file_count",
            "transformers_package_receipt_sha256",
        ):
            self.assertIn(field, production["required"])
            self.assertIn(field, production["properties"])

        request = fixture_request()
        observed = valid_production_shape(request)
        for field, bad in (
            ("transformers_package_file_count", 0),
            ("transformers_package_file_count", True),
            ("transformers_package_receipt_sha256", "g" * 64),
            ("transformers_package_receipt_sha256", "A" * 64),
        ):
            candidate = dict(observed)
            candidate[field] = bad
            with self.subTest(field=field, bad=bad):
                with self.assertRaisesRegex(CaptureContractError, "Transformers package"):
                    _validate_production_metadata_shape(candidate, request)

    def test_cpu_observation_requires_concrete_processor_model(self):
        request = fixture_request()
        observed = valid_production_shape(request)
        self.assertTrue(_is_concrete_cpu_identity(observed["cpu_processor"]))
        _validate_production_metadata_shape(observed, request)

        for bad in (None, "", "   ", "generic", "CPU", "x86_64", "AMD64", "arm64", "aarch64"):
            candidate = dict(observed)
            candidate["cpu_processor"] = bad
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(CaptureContractError, "concrete processor"):
                    _validate_production_metadata_shape(candidate, request)

        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        cpu_rule = next(
            rule for rule in production["allOf"]
            if rule["if"].get("properties", {}).get("device", {}).get("const") == "cpu"
        )
        processor = cpu_rule["then"]["properties"]["cpu_processor"]
        self.assertEqual(processor["type"], "string")
        self.assertEqual(processor["pattern"], r"\S")

    def test_active_torch_dispatch_and_function_modes_are_rejected(self):
        backend = object.__new__(Backend)
        for dispatch_count, function_count, label in (
            (1, 0, "__torch_dispatch__"),
            (0, 1, "__torch_function__"),
        ):
            backend._torch = SimpleNamespace(
                nn=object(),
                _C=SimpleNamespace(
                    _len_torch_dispatch_stack=lambda value=dispatch_count: value,
                    _len_torch_function_stack=lambda value=function_count: value,
                ),
            )
            with self.subTest(label=label):
                with self.assertRaisesRegex(CaptureContractError, label):
                    backend._assert_no_active_torch_override_modes()

        backend._torch = SimpleNamespace(
            nn=object(),
            _C=SimpleNamespace(
                _len_torch_dispatch_stack=lambda: 0,
                _len_torch_function_stack=lambda: 0,
            ),
        )
        backend._assert_no_active_torch_override_modes()

        backend._torch._C._len_torch_dispatch_stack = None
        with self.assertRaisesRegex(CaptureContractError, "cannot authenticate"):
            backend._assert_no_active_torch_override_modes()

        source = inspect.getsource(Backend.begin_observation)
        exclusion = source.index("self._enter_exclusive_python_thread_boundary()")
        mode_check = source.index("self._assert_no_active_torch_override_modes()")
        state_check = source.index("self._assert_model_runtime_attributes()")
        self.assertLess(exclusion, mode_check)
        self.assertLess(mode_check, state_check)


if __name__ == "__main__":
    unittest.main()
