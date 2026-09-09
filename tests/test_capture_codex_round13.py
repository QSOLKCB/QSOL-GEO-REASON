from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_backend_core import (
    HuggingFacePyTorchBackend as CoreHuggingFacePyTorchBackend,
)
from qsol_geo_reason.capture_execute import execute_capture
from qsol_geo_reason.capture_provenance import (
    _is_canonical_snapshot_path,
    _validate_production_metadata_shape,
    _validate_snapshot_receipt,
)
from qsol_geo_reason.provenance import SourceIdentityError, resolve_implementation_revision
from test_capture_codex_round6 import valid_production_shape

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def production_shape(request: dict) -> dict:
    return valid_production_shape(request)


class FakeDeterminismTorch:
    def __init__(self):
        self.enabled = False

    def use_deterministic_algorithms(self, value):
        self.enabled = bool(value)

    def are_deterministic_algorithms_enabled(self):
        return self.enabled


class CaptureRound13RegressionTests(unittest.TestCase):
    def test_required_determinism_is_reasserted_and_rechecked_each_forward(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeDeterminismTorch()
        backend._determinism_mode = "required"
        backend._last_deterministic_algorithms_enabled = None
        backend._force_required_determinism_policy()
        self.assertIs(backend._torch.enabled, True)
        backend._torch.enabled = False
        with self.assertRaisesRegex(CaptureContractError, "determinism policy drifted"):
            backend._assert_required_determinism_policy()

        source = inspect.getsource(CoreHuggingFacePyTorchBackend.hidden_states)
        forward = source.index("self._base_model(")
        self.assertLess(source.index("self._force_required_determinism_policy()"), forward)
        self.assertGreater(source.index("self._assert_required_determinism_policy()"), forward)

    def test_mps_observation_requires_built_and_available_backend(self):
        request = fixture_request()
        request["backend"]["device"] = "mps"
        observed = production_shape(request)
        _validate_production_metadata_shape(observed, request)
        for field in ("mps_built", "mps_available"):
            mutated = dict(observed)
            mutated[field] = False
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "built.*available"):
                    _validate_production_metadata_shape(mutated, request)

    def test_trajectory_schema_rejects_whitespace_only_observed_dtype(self):
        schema = json.loads(
            (ROOT / "schemas" / "captured-trajectory.schema.json").read_text(
                encoding="utf-8"
            )
        )
        observed_dtype = schema["$defs"]["layer"]["properties"]["observed_dtype"]
        self.assertEqual(observed_dtype["pattern"], "\\S")

    def test_snapshot_receipts_require_normalized_relative_posix_paths(self):
        digest = "a" * 64
        valid = {"config.json": digest, "subdir/model.safetensors": digest}
        observed = {
            "model_snapshot_file_count": len(valid),
            "model_snapshot_file_sha256": valid,
            "model_snapshot_receipt_sha256": sha256_json(valid),
        }
        _validate_snapshot_receipt(observed, "model")
        for path in (
            "../outside.safetensors",
            "/absolute.safetensors",
            "subdir/../outside.safetensors",
            "subdir//file.bin",
            "C:/outside.bin",
            "subdir\\file.bin",
        ):
            self.assertFalse(_is_canonical_snapshot_path(path), path)
            hashes = {path: digest}
            mutated = {
                "model_snapshot_file_count": 1,
                "model_snapshot_file_sha256": hashes,
                "model_snapshot_receipt_sha256": sha256_json(hashes),
            }
            with self.subTest(path=path):
                with self.assertRaisesRegex(CaptureContractError, "artifact hashes are malformed"):
                    _validate_snapshot_receipt(mutated, "model")

    def test_cpu_mkldnn_policy_is_restored_and_recorded(self):
        matmul = SimpleNamespace(fp32_precision="ieee")
        mkldnn = SimpleNamespace(enabled=True, matmul=matmul)
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = SimpleNamespace(backends=SimpleNamespace(mkldnn=mkldnn))
        backend._canonical_cpu_matmul_policy = backend._cpu_matmul_policy_state()
        backend._last_cpu_matmul_policy = None

        mkldnn.enabled = False
        matmul.fp32_precision = "bf16"
        backend._force_cpu_matmul_policy()
        self.assertEqual(
            backend._last_cpu_matmul_policy,
            {
                "cpu_mkldnn_enabled": True,
                "cpu_mkldnn_matmul_fp32_precision": "ieee",
            },
        )
        metadata_source = inspect.getsource(CoreHuggingFacePyTorchBackend.metadata)
        self.assertIn('"cpu_mkldnn_enabled"', metadata_source)
        self.assertIn('"cpu_mkldnn_matmul_fp32_precision"', metadata_source)

    def test_observation_revision_requires_clean_executing_checkout(self):
        with patch("qsol_geo_reason.provenance.git_source_revision", return_value=None):
            with self.assertRaisesRegex(SourceIdentityError, "requires execution from"):
                resolve_implementation_revision("a" * 40, require_checkout=True)

        source = inspect.getsource(execute_capture)
        self.assertIn("resolve_implementation_revision(", source)
        self.assertIn("require_checkout=True", source)
        self.assertIn("OBSERVATION implementation revision is not bound", source)

    def test_manifest_schema_binds_snapshot_paths_cpu_policy_and_mps_availability(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        for field in ("cpu_mkldnn_enabled", "cpu_mkldnn_matmul_fp32_precision"):
            self.assertIn(field, required)
        self.assertEqual(
            schema["$defs"]["shaMap"]["propertyNames"]["$ref"],
            "#/$defs/snapshotPath",
        )
        conditional = production["allOf"][0]["then"]["properties"]
        self.assertIs(conditional["mps_built"]["const"], True)
        self.assertIs(conditional["mps_available"]["const"], True)


if __name__ == "__main__":
    unittest.main()
