from __future__ import annotations

import inspect
import json
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from test_capture_codex_round17 import fixture_request, production_shape

ROOT = Path(__file__).resolve().parents[1]


class FakeRuntimeModule:
    def __init__(self, scaling: float = 0.5):
        self.scaling = scaling
        self.training = False

    def forward(self, value=None):
        return value


class FakeRuntimeModel:
    def __init__(self):
        self.attention = FakeRuntimeModule()
        self.training = False

    def forward(self, value=None):
        return self.attention.forward(value)

    def named_modules(self):
        return [("", self), ("attention", self.attention)]


class CaptureRound22RegressionTests(unittest.TestCase):
    def test_process_global_pytorch_hooks_are_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeRuntimeModel()
        module_api = SimpleNamespace(_global_forward_hooks={})
        backend._torch = SimpleNamespace(
            nn=SimpleNamespace(modules=SimpleNamespace(module=module_api))
        )
        backend._assert_no_registered_module_hooks()

        module_api._global_forward_hooks = {1: object()}
        with self.assertRaisesRegex(CaptureContractError, "process-global PyTorch execution hooks"):
            backend._assert_no_registered_module_hooks()

    def test_mutable_module_execution_attributes_are_sealed_before_observation(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeRuntimeModel()
        backend._torch = SimpleNamespace(is_tensor=lambda _value: False)
        original = backend._model_runtime_attributes_seal()
        backend._canonical_model_runtime_attributes = original
        backend._assert_model_runtime_attributes()

        backend._model.attention.scaling = 0.75
        self.assertNotEqual(backend._model_runtime_attributes_seal(), original)
        with self.assertRaisesRegex(CaptureContractError, "runtime execution attributes changed"):
            backend._assert_model_runtime_attributes()

        source = inspect.getsource(HuggingFacePyTorchBackend.begin_observation)
        self.assertIn("self._enter_exclusive_python_thread_boundary()", source)
        self.assertIn("self._assert_model_runtime_attributes()", source)

    def test_runtime_identity_fields_reject_whitespace_and_schema_matches(self):
        request = fixture_request()
        observed = production_shape(request)
        identity_fields = (
            "python_version",
            "platform",
            "torch_version",
            "transformers_version",
            "model_class",
            "tokenizer_class",
        )
        for field in identity_fields:
            mutated = dict(observed)
            mutated[field] = "   "
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "non-whitespace string"):
                    _validate_production_metadata_shape(mutated, request)

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        properties = schema["$defs"]["backendObservedProduction"]["properties"]
        for field in identity_fields:
            self.assertEqual(properties[field]["pattern"], "\\S")

    def test_observation_boundary_rejects_concurrent_python_threads_and_blocks_new_ones(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._exclusive_thread_boundary_state = None
        started = threading.Event()
        release = threading.Event()

        def background():
            started.set()
            release.wait(timeout=5)

        worker = threading.Thread(target=background)
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            with self.assertRaisesRegex(CaptureContractError, "exclusive Python-thread execution"):
                backend._enter_exclusive_python_thread_boundary()
        finally:
            release.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

        backend._enter_exclusive_python_thread_boundary()
        try:
            with self.assertRaisesRegex(CaptureContractError, "starting another Python thread"):
                threading.Thread(target=lambda: None).start()
        finally:
            backend._leave_exclusive_python_thread_boundary()

        probe = threading.Thread(target=lambda: None)
        probe.start()
        probe.join(timeout=2)
        self.assertFalse(probe.is_alive())

    def test_mps_policy_and_import_pristineness_are_checked_before_torch_import(self):
        request = fixture_request()
        request["backend"]["device"] = "mps"

        with patch.dict(os.environ, {}, clear=True):
            HuggingFacePyTorchBackend._validate_pre_mps_environment(request)
        for variable in (
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "PYTORCH_MPS_FAST_MATH",
            "PYTORCH_MPS_PREFER_METAL",
        ):
            with self.subTest(variable=variable):
                with patch.dict(os.environ, {variable: "1"}, clear=True):
                    with self.assertRaisesRegex(CaptureContractError, variable):
                        HuggingFacePyTorchBackend._validate_pre_mps_environment(request)

        HuggingFacePyTorchBackend._assert_pristine_mps_import_state("mps", {})
        with self.assertRaisesRegex(CaptureContractError, "fresh PyTorch import boundary"):
            HuggingFacePyTorchBackend._assert_pristine_mps_import_state(
                "mps", {"torch": object()}
            )
        HuggingFacePyTorchBackend._assert_pristine_mps_import_state(
            "cpu", {"torch": object()}
        )

        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertLess(
            source.index("self._validate_pre_mps_environment(validated)"),
            source.index("import torch as process_torch"),
        )
        self.assertLess(
            source.index("self._assert_pristine_mps_import_state(device)"),
            source.index("import torch as process_torch"),
        )
        self.assertLess(
            source.index("import torch as process_torch"),
            source.index("self._snapshot_torch_process_state(process_torch, device)"),
        )


if __name__ == "__main__":
    unittest.main()
