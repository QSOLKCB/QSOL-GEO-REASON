from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend, execute_capture, verify_capture_bundle

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
REVISION = "d" * 40


class FakeBackend:
    def __init__(self, request: dict):
        self.request = request

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def hidden_states(self, input_ids, layer_indices, *, pool_span):
        start, end = pool_span
        selected = {}
        for layer_index in layer_indices:
            rows = [
                [float(token_id + layer_index), float(token_id * (layer_index + 1))]
                for token_id in input_ids[start:end]
            ]
            selected[layer_index] = {
                "vector": [
                    math.fsum(row[axis] for row in rows) / len(rows)
                    for axis in range(2)
                ],
                "vector_dimension": 2,
                "observed_dtype": "float64",
            }
        return selected

    def metadata(self):
        return {
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


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def execute_fixture():
    request = fixture_request()
    manifest, trajectory = execute_capture(
        request,
        implementation_revision=REVISION,
        backend=FakeBackend(request),
    )
    return request, manifest, trajectory


def rebind_all_hashes(manifest: dict, trajectory: dict) -> None:
    identity_keys = (
        "schema_version",
        "protocol_id",
        "run_id",
        "repository_commit",
        "request_sha256",
        "model",
        "backend_request",
        "backend_observed",
        "capture",
        "determinism",
        "generation_parameters",
    )
    run_manifest_id = sha256_json({key: manifest[key] for key in identity_keys})
    manifest["run_manifest_id"] = run_manifest_id
    trajectory["run_manifest_id"] = run_manifest_id
    trajectory_payload = {
        key: value for key, value in trajectory.items() if key != "trajectory_sha256"
    }
    trajectory_sha = sha256_json(trajectory_payload)
    trajectory["trajectory_sha256"] = trajectory_sha
    manifest["artifacts"]["captured_trajectory_sha256"] = trajectory_sha
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_json(manifest_payload)


class CaptureRound4RegressionTests(unittest.TestCase):
    def test_shared_schema_version_is_bound_to_capture_contract(self):
        request, manifest, trajectory = execute_fixture()
        manifest["schema_version"] = trajectory["schema_version"] = "bogus"
        rebind_all_hashes(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "schema_version"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_shared_protocol_id_is_bound_to_capture_contract(self):
        request, manifest, trajectory = execute_fixture()
        manifest["protocol_id"] = trajectory["protocol_id"] = "bogus"
        rebind_all_hashes(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "protocol_id"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_shared_repository_commit_must_be_canonical_git_sha(self):
        request, manifest, trajectory = execute_fixture()
        manifest["repository_commit"] = trajectory["repository_commit"] = "bogus"
        rebind_all_hashes(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "repository_commit"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_cuda_sdpa_policy_is_forced_to_math_only(self):
        init_source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        policy_source = inspect.getsource(HuggingFacePyTorchBackend._force_sdpa_math_policy)
        self.assertIn('self._attention_implementation == "sdpa"', init_source)
        self.assertIn("self._force_sdpa_math_policy()", init_source)
        self.assertIn('(\"enable_flash_sdp\", False)', policy_source)
        self.assertIn('(\"enable_mem_efficient_sdp\", False)', policy_source)
        self.assertIn('(\"enable_math_sdp\", True)', policy_source)
        self.assertIn('getattr(cuda_backend, \"enable_cudnn_sdp\", None)', policy_source)
        self.assertIn("cudnn_toggle(False)", policy_source)


if __name__ == "__main__":
    unittest.main()
