from __future__ import annotations

import inspect
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_provenance import (
    _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS,
    _validate_production_metadata_shape,
)
from test_capture_codex_round6 import valid_production_shape

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def production_shape(request: dict) -> dict:
    return valid_production_shape(request)


class FakeTensor:
    def __init__(self):
        self._version = 0
        self.shape = (2, 2)
        self.dtype = "torch.float32"
        self.device = "cpu"


class FakeModel:
    def __init__(self):
        self.weight = FakeTensor()

    def named_parameters(self):
        return [("weight", self.weight)]

    def named_buffers(self):
        return []


class FakeTokenizerBackend:
    def __init__(self):
        self.state = '{"model":"fake"}'

    def to_str(self):
        return self.state


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "b": 1}
        self.backend_tokenizer = FakeTokenizerBackend()
        self.added_tokens_encoder = {}
        self.special_tokens_map = {"eos_token": "b"}
        self.all_special_tokens = ["b"]
        self.all_special_ids = [1]
        self.init_kwargs = {"clean_up_tokenization_spaces": False}

    def get_vocab(self):
        return dict(self.vocab)


class CaptureRound17RegressionTests(unittest.TestCase):
    def test_production_thread_counts_are_positive_and_schema_bound(self):
        request = fixture_request()
        observed = production_shape(request)
        _validate_production_metadata_shape(observed, request)

        for field in ("torch_num_threads", "torch_num_interop_threads"):
            for bad in (None, 0, -1, True, 1.0):
                mutated = dict(observed)
                mutated[field] = bad
                with self.subTest(field=field, bad=bad):
                    with self.assertRaisesRegex(CaptureContractError, "positive integer"):
                        _validate_production_metadata_shape(mutated, request)

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        properties = schema["$defs"]["backendObservedProduction"]["properties"]
        for field in ("torch_num_threads", "torch_num_interop_threads"):
            self.assertEqual(properties[field], {"type": "integer", "minimum": 1})

    def test_required_cuda_workspace_is_validated_before_core_initialization(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda:0"
        request["determinism"]["mode"] = "required"

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(CaptureContractError, "CUBLAS_WORKSPACE_CONFIG"):
                HuggingFacePyTorchBackend._validate_pre_cuda_environment(request)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":1:1"}, clear=True):
            with self.assertRaisesRegex(CaptureContractError, "CUBLAS_WORKSPACE_CONFIG"):
                HuggingFacePyTorchBackend._validate_pre_cuda_environment(request)
        for value in sorted(_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": value}, clear=True):
                    HuggingFacePyTorchBackend._validate_pre_cuda_environment(request)

        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertLess(
            source.index("self._validate_pre_cuda_environment(validated)"),
            source.index("super().__init__(validated)"),
        )

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        workspace_rule = production["allOf"][1]
        self.assertEqual(
            set(workspace_rule["then"]["properties"]["cublas_workspace_config"]["enum"]),
            _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS,
        )

    def _sealed_backend(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeModel()
        backend._tokenizer = FakeTokenizer()
        backend._canonical_model_live_state = backend._model_live_state_seal()
        backend._canonical_tokenizer_live_state = backend._tokenizer_live_state_seal()
        backend._live_state_seal_initialized = True
        return backend

    def test_live_model_and_tokenizer_mutation_is_rejected_before_reuse(self):
        backend = self._sealed_backend()
        backend._assert_live_state_authentication()
        backend._model.weight._version += 1
        with self.assertRaisesRegex(CaptureContractError, "live model state changed"):
            backend._assert_live_state_authentication()

        backend = self._sealed_backend()
        backend._tokenizer.vocab["c"] = 2
        with self.assertRaisesRegex(CaptureContractError, "live tokenizer state changed"):
            backend._assert_live_state_authentication()

        backend = self._sealed_backend()
        backend._model.weight = FakeTensor()
        with self.assertRaisesRegex(CaptureContractError, "live model state changed"):
            backend._assert_live_state_authentication()

        # Audit the complete dispatch chain instead of requiring each wrapper to
        # duplicate an expensive guard already supplied by its parent.
        request_guards = [
            inspect.getsource(cls.__dict__["assert_execution_request"])
            for cls in HuggingFacePyTorchBackend.__mro__
            if "assert_execution_request" in cls.__dict__
        ]
        self.assertEqual(
            sum(source.count("self._assert_live_state_authentication()") for source in request_guards),
            1,
        )
        capture_guard = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        self.assertGreaterEqual(capture_guard.count("self._assert_live_state_authentication()"), 2)


if __name__ == "__main__":
    unittest.main()
