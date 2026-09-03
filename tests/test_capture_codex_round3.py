from __future__ import annotations

import copy
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend,
    _snapshot_file_hashes,
    execute_capture,
    verify_capture_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
REV = "d" * 40


class FakeBackend:
    def __init__(self, request: dict, updates: dict | None = None):
        self.request = request
        self.updates = updates or {}

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def hidden_states(self, input_ids, layer_indices, *, pool_span):
        start, end = pool_span
        out = {}
        for layer in layer_indices:
            rows = [[float(token + layer), float(token * (layer + 1))] for token in input_ids[start:end]]
            out[layer] = {
                "vector": [math.fsum(row[i] for row in rows) / len(rows) for i in range(2)],
                "vector_dimension": 2,
                "observed_dtype": "float64",
            }
        return out

    def metadata(self):
        value = {
            "name": "huggingface-pytorch",
            "observed_model_commit": self.request["model"]["revision"],
            "observed_tokenizer_commit": self.request["model"]["tokenizer_revision"],
            "device": self.request["backend"]["device"],
            "dtype": self.request["backend"]["dtype"],
            "quantization": "none",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_cache": False,
            "capture_phase": "replayed_prefix",
            "kv_cache_reuse": False,
            "determinism_mode": self.request["determinism"]["mode"],
        }
        value.update(self.updates)
        return value


def request_fixture() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def execute(request: dict | None = None, backend: FakeBackend | None = None):
    request = request or request_fixture()
    manifest, trajectory = execute_capture(
        request,
        implementation_revision=REV,
        backend=backend or FakeBackend(request),
    )
    return request, manifest, trajectory


def rehash_manifest(manifest: dict) -> None:
    manifest["manifest_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def rehash_trajectory(manifest: dict, trajectory: dict) -> None:
    trajectory["trajectory_sha256"] = sha256_json({k: v for k, v in trajectory.items() if k != "trajectory_sha256"})
    manifest["artifacts"]["captured_trajectory_sha256"] = trajectory["trajectory_sha256"]
    rehash_manifest(manifest)


def rebind_run_manifest_id(manifest: dict, trajectory: dict) -> None:
    keys = (
        "schema_version", "protocol_id", "run_id", "repository_commit", "request_sha256",
        "model", "backend_request", "backend_observed", "capture", "determinism",
        "generation_parameters",
    )
    value = sha256_json({key: manifest[key] for key in keys})
    manifest["run_manifest_id"] = value
    trajectory["run_manifest_id"] = value
    rehash_trajectory(manifest, trajectory)


class CaptureRound3RegressionTests(unittest.TestCase):
    def test_base_model_is_invoked_without_language_model_logits(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        self.assertIn("self._base_model(", source)
        self.assertNotIn("self._model(", source)
        self.assertIn("output_hidden_states=False", source)

    def test_snapshot_receipt_hashes_every_regular_file(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / commit
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"weights")
            hashes = _snapshot_file_hashes(root, commit, "model")
            self.assertEqual(set(hashes), {"config.json", "weights.bin"})
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_simulation_metadata_constants_are_enforced(self):
        for field, bad in (
            ("local_files_only", False), ("trust_remote_code", True), ("use_cache", True),
            ("capture_phase", "decode"), ("kv_cache_reuse", True), ("quantization", "int8"),
            ("device", "cuda"), ("dtype", "float16"), ("determinism_mode", "best_effort"),
        ):
            request = request_fixture()
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "backend provenance field"):
                    execute_capture(request, implementation_revision=REV, backend=FakeBackend(request, {field: bad}))

    def test_request_run_id_cannot_be_rewritten_by_self_consistent_artifacts(self):
        request, manifest, trajectory = execute()
        manifest["run_id"] = trajectory["run_id"] = "other-run"
        rebind_run_manifest_id(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "run_id does not match capture request"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_first_and_isolated_spans_are_recomputed_from_prefix_identity(self):
        request, manifest, trajectory = execute()
        trajectory["steps"][0]["changed_token_span"] = [2, 4]
        rehash_trajectory(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "changed token span mismatch at step 0"):
            verify_capture_bundle(request, manifest, trajectory)

        request = request_fixture()
        request["capture"]["context_mode"] = "isolated"
        request, manifest, trajectory = execute(request)
        trajectory["steps"][1]["changed_token_span"] = [2, 4]
        rehash_trajectory(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "changed token span mismatch at step 1"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_vector_dimension_must_remain_stable_across_steps(self):
        request, manifest, trajectory = execute()
        record = trajectory["steps"][1]["layers"][0]
        record["vector_dimension"] = 3
        record["vector"] = [1.0, 2.0, 3.0]
        record["vector_sha256"] = sha256_json(record["vector"])
        rehash_trajectory(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "vector dimension changed"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_production_metadata_records_precision_mps_and_snapshot_receipts(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.metadata)
        for marker in (
            "get_float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
            "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
            "mps_mac_model", "mps_cpu_brand", "mps_macos_version",
            "model_snapshot_file_sha256", "tokenizer_snapshot_file_sha256",
        ):
            self.assertIn(marker, source)

    def test_trajectory_schema_binds_prefix_token_identity(self):
        schema = json.loads((ROOT / "schemas" / "captured-trajectory.schema.json").read_text(encoding="utf-8"))
        required = set(schema["properties"]["representation_definition"]["required"])
        self.assertIn("prefix_input_ids", required)
        self.assertIn("prefix_input_ids_sha256", required)


if __name__ == "__main__":
    unittest.main()
