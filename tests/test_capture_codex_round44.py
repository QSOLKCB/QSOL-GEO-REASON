"""Round 44 regressions for native imports, observation receipts, CUDA identity, and loaders."""
from __future__ import annotations

import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend_round44 as round44
from qsol_geo_reason.capture_backend_round44 import (
    HuggingFacePyTorchBackend as Round44Backend,
    _assert_no_ignored_importable_native_artifacts,
    _assert_observation_safetensors_policy,
    _assert_transformers_loaders_source_bound,
    _assert_transformers_model_source_bound,
    _cuda_identity_tuple,
    _ignored_importable_native_artifacts,
    _remember_round44_cuda_identity,
    _require_observation_tree_receipts,
    _validate_backend_metadata_round44,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.provenance import SourceIdentityError
from test_capture_codex_round6 import fixture_request

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound44RegressionTests(unittest.TestCase):
    def test_ignored_native_package_artifacts_are_rejected(self):
        listing = types.SimpleNamespace(
            stdout=(
                "src/qsol_geo_reason/override.so\0"
                "src/qsol_geo_reason/override.pyd\0"
                "src/qsol_geo_reason/__pycache__/ok.pyc\0"
                "build/qsol_geo_reason/ignored.so\0"
            )
        )
        with patch.object(round44._provenance, "_git_run", return_value=listing):
            self.assertEqual(
                _ignored_importable_native_artifacts(Path("/synthetic")),
                (
                    "src/qsol_geo_reason/override.pyd",
                    "src/qsol_geo_reason/override.so",
                ),
            )
            with self.assertRaisesRegex(SourceIdentityError, "ignored native artifact"):
                _assert_no_ignored_importable_native_artifacts(Path("/synthetic"))

        source = Path(round44.__file__).read_text(encoding="utf-8")
        self.assertIn("_assert_no_ignored_importable_native_artifacts", source)
        self.assertIn("_provenance.git_source_revision = _git_source_revision_round44", source)

    def test_verified_observation_requires_both_frozen_tree_receipts(self):
        request = fixture_request()
        request["model"].pop("revision_tree_sha256", None)
        request["model"].pop("tokenizer_revision_tree_sha256", None)
        with self.assertRaisesRegex(CaptureContractError, "frozen Hub commit-tree"):
            _require_observation_tree_receipts(request)

        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        _require_observation_tree_receipts(request)

        # Exercise the evidence-aware wrapper without needing a complete production
        # metadata fixture: receipt rejection happens before the established validator.
        missing = fixture_request()
        with patch.object(round44, "_ORIGINAL_VALIDATE_BACKEND_METADATA", return_value=None):
            with self.assertRaisesRegex(CaptureContractError, "frozen Hub commit-tree"):
                _validate_backend_metadata_round44({}, missing, "OBSERVATION")
            _validate_backend_metadata_round44({}, missing, "SIMULATION")

    def test_verified_observation_requires_safetensors_and_rejects_legacy_weights(self):
        with self.assertRaisesRegex(CaptureContractError, "requires the Safetensors"):
            _assert_observation_safetensors_policy(
                {
                    "safetensors_deserializer_active": False,
                    "model_snapshot_file_sha256": {"pytorch_model.bin": "a" * 64},
                }
            )

        _assert_observation_safetensors_policy(
            {
                "safetensors_deserializer_active": True,
                "model_snapshot_file_sha256": {"model.safetensors": "a" * 64},
            }
        )
        with self.assertRaisesRegex(CaptureContractError, "pickle-capable"):
            _assert_observation_safetensors_policy(
                {
                    "safetensors_deserializer_active": True,
                    "model_snapshot_file_sha256": {
                        "model.safetensors": "a" * 64,
                        "pytorch_model.bin": "b" * 64,
                    },
                }
            )

    def test_cuda_hardware_identity_is_bound_outside_instance_state(self):
        backend = object.__new__(Round44Backend)
        backend._cuda_hardware_identity = {
            "cuda_device_name": "NVIDIA L4",
            "cuda_device_capability": "8.9",
            "cuda_device_uuid": "GPU-synthetic",
        }
        _remember_round44_cuda_identity(
            backend, _cuda_identity_tuple(backend._cuda_hardware_identity)
        )
        backend._assert_round44_cuda_hardware_identity()

        backend._cuda_hardware_identity = {
            "cuda_device_name": "NVIDIA H200",
            "cuda_device_capability": "9.0",
            "cuda_device_uuid": "GPU-forged",
        }
        with self.assertRaisesRegex(CaptureContractError, "CUDA hardware identity changed"):
            backend._assert_round44_cuda_hardware_identity()

    def test_transformers_loaders_and_model_forwards_match_receipt_backed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "transformers"
            package.mkdir()
            init_file = package / "__init__.py"
            init_file.write_text("# synthetic receipt-backed package\n", encoding="utf-8")
            source_file = package / "synthetic.py"
            source = (
                "class AutoTokenizer:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *args, **kwargs):\n"
                "        return cls()\n\n"
                "class AutoModelForCausalLM:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *args, **kwargs):\n"
                "        return Model()\n\n"
                "class Model:\n"
                "    def named_modules(self):\n"
                "        return ((\"\", self),)\n"
                "    def forward(self, *args, **kwargs):\n"
                "        return None\n"
            )
            source_file.write_text(source, encoding="utf-8")
            namespace = {"__name__": "transformers.synthetic"}
            exec(compile(source, str(source_file), "exec"), namespace)
            fake_transformers = types.SimpleNamespace(
                __file__=str(init_file),
                AutoTokenizer=namespace["AutoTokenizer"],
                AutoModelForCausalLM=namespace["AutoModelForCausalLM"],
            )

            receipt = _assert_transformers_loaders_source_bound(fake_transformers)
            self.assertGreater(receipt[0], 0)
            self.assertEqual(len(receipt[1]), 64)
            model = namespace["Model"]()
            _assert_transformers_model_source_bound(fake_transformers, model)

            original_loader = namespace["AutoModelForCausalLM"].from_pretrained
            replacement_namespace = {"__name__": "transformers.synthetic"}
            exec(
                compile(
                    "def replacement(cls, *args, **kwargs):\n    return None\n",
                    "<string>",
                    "exec",
                ),
                replacement_namespace,
            )
            namespace["AutoModelForCausalLM"].from_pretrained = classmethod(
                replacement_namespace["replacement"]
            )
            with self.assertRaisesRegex(CaptureContractError, "receipt-backed"):
                _assert_transformers_loaders_source_bound(fake_transformers)
            namespace["AutoModelForCausalLM"].from_pretrained = original_loader.__func__

            def forged_forward(self, *args, **kwargs):
                return "forged"

            namespace["Model"].forward = forged_forward
            with self.assertRaisesRegex(CaptureContractError, "Transformers package"):
                _assert_transformers_model_source_bound(fake_transformers, model)

    def test_run_manifest_schema_requires_observation_receipts_and_safetensors(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production_rule = next(
            rule
            for rule in schema["allOf"]
            if rule.get("if", {})
            .get("properties", {})
            .get("backend_observed", {})
            .get("properties", {})
            .get("name", {})
            .get("const")
            == "huggingface-pytorch"
        )
        required = production_rule["then"]["properties"]["model"]["required"]
        self.assertIn("revision_tree_sha256", required)
        self.assertIn("tokenizer_revision_tree_sha256", required)

        production = schema["$defs"]["backendObservedProduction"]
        properties = production["properties"]
        self.assertEqual(properties["safetensors_deserializer_active"], {
            "const": True,
            "description": "Canonical OBSERVATION requires Safetensors checkpoint loading; legacy pickle-capable checkpoint lanes are forbidden.",
        })
        self.assertEqual(
            properties["safetensors_package_file_count"],
            {"type": "integer", "minimum": 1},
        )
        self.assertEqual(
            properties["safetensors_package_receipt_sha256"],
            {"$ref": "#/$defs/sha256"},
        )

        legacy_rule = next(
            rule
            for rule in production["allOf"]
            if "model_snapshot_file_sha256" in rule.get("properties", {})
        )
        pattern = legacy_rule["properties"]["model_snapshot_file_sha256"][
            "propertyNames"
        ]["pattern"]
        self.assertIsNone(re.fullmatch(pattern, "pytorch_model.bin"))
        self.assertIsNone(re.fullmatch(pattern, "weights.ckpt"))
        self.assertIsNotNone(re.fullmatch(pattern, "model.safetensors"))


if __name__ == "__main__":
    unittest.main()
