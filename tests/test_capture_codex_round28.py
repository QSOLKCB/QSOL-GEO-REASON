"""Round 28 regressions for bytecode, adapter routing, class dispatch, and CUDA schema parity."""
from __future__ import annotations

import importlib.util
import json
import py_compile
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.capture_backend import _ExplicitNoAttentionBaseModel
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_execute import (
    _assert_observation_backend_execution_methods,
    _assert_observation_backend_routing_state,
)
from qsol_geo_reason.provenance import SourceIdentityError, git_source_revision

ROOT = Path(__file__).resolve().parents[1]


class _RoutingModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.base_model = SimpleNamespace(layers=[object(), object()])


class CaptureRound28RegressionTests(unittest.TestCase):
    def test_checkout_bound_observation_accepts_matching_self_generated_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            source = package / "capture.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

            cache_path = Path(importlib.util.cache_from_source(str(source)))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(source), cfile=str(cache_path), doraise=True)

            with patch("qsol_geo_reason.provenance.source_repo_root", return_value=root):
                expected = git_source_revision(
                    require_clean=True,
                    reject_importable_bytecode=True,
                )
                self.assertRegex(expected or "", r"^[0-9a-f]{40}$")

                payload = bytearray(cache_path.read_bytes())
                payload[-1] ^= 1
                cache_path.write_bytes(payload)
                with self.assertRaisesRegex(SourceIdentityError, "bytecode cache"):
                    git_source_revision(
                        require_clean=True,
                        reject_importable_bytecode=True,
                    )

    def test_observation_backend_class_method_replacement_is_pinned_to_import_receipt(self):
        backend = object.__new__(Backend)
        _assert_observation_backend_execution_methods(backend)
        had_local = "tokenize" in vars(Backend)
        original_local = vars(Backend).get("tokenize")
        try:
            Backend.tokenize = lambda self, text: [999]
            with self.assertRaisesRegex(CaptureContractError, "trusted import"):
                _assert_observation_backend_execution_methods(backend)
        finally:
            if had_local:
                Backend.tokenize = original_local
            else:
                delattr(Backend, "tokenize")
        _assert_observation_backend_execution_methods(backend)

    def test_observation_backend_routing_binds_blocks_and_base_model_identity(self):
        model = _RoutingModel()
        backend = object.__new__(Backend)
        backend._model = model
        backend._block_path = "layers"
        backend._blocks = model.base_model.layers
        backend._hidden_state_count = 3
        backend._device_type = "cuda"
        backend._attention_implementation = "eager"
        backend._base_model = _ExplicitNoAttentionBaseModel(model.base_model)

        _assert_observation_backend_routing_state(backend)
        original_blocks = backend._blocks
        backend._blocks = list(reversed(original_blocks))
        with self.assertRaisesRegex(CaptureContractError, "routing object changed"):
            _assert_observation_backend_routing_state(backend)
        backend._blocks = original_blocks

        backend._base_model = _ExplicitNoAttentionBaseModel(SimpleNamespace(layers=original_blocks))
        with self.assertRaisesRegex(CaptureContractError, "routing identity changed"):
            _assert_observation_backend_routing_state(backend)

    def test_cuda_reduced_precision_schema_requires_false_and_null_elsewhere(self):
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        fields = (
            "cuda_matmul_allow_fp16_reduced_precision_reduction",
            "cuda_matmul_allow_bf16_reduced_precision_reduction",
        )
        rule = next(
            item
            for item in production["allOf"]
            if all(field in item.get("then", {}).get("properties", {}) for field in fields)
        )
        self.assertEqual(
            rule["if"]["properties"]["device"]["pattern"],
            "^cuda:[0-9]+$",
        )
        for field in fields:
            self.assertEqual(rule["then"]["properties"][field], {"const": False})
            self.assertEqual(rule["else"]["properties"][field], {"type": "null"})


if __name__ == "__main__":
    unittest.main()
