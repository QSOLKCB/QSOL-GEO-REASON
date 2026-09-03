from __future__ import annotations

import copy
import inspect
import json
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend,
    execute_capture,
    verify_capture_bundle,
)
from qsol_geo_reason.capture_validation import validate_capture_request

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
REV = "d" * 40


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class FakeBackend:
    def __init__(self, request: dict):
        self.request = request

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def hidden_states(self, input_ids, layer_indices, *, pool_span):
        start, end = pool_span
        out = {}
        for layer in layer_indices:
            rows = [
                [float(token + layer), float(token * (layer + 1))]
                for token in input_ids[start:end]
            ]
            out[layer] = {
                "vector": [
                    math.fsum(row[i] for row in rows) / len(rows)
                    for i in range(2)
                ],
                "vector_dimension": 2,
                "observed_dtype": "float64",
            }
        return out

    def metadata(self):
        return {
            "name": "ignored-by-simulation-canonicalization",
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


def rehash_trajectory_and_manifest(manifest: dict, trajectory: dict) -> None:
    trajectory["trajectory_sha256"] = sha256_json(
        {k: v for k, v in trajectory.items() if k != "trajectory_sha256"}
    )
    manifest["artifacts"]["captured_trajectory_sha256"] = trajectory["trajectory_sha256"]
    manifest["manifest_sha256"] = sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )


class CaptureRound8RegressionTests(unittest.TestCase):
    def test_snapshot_receipts_bracket_checkpoint_loading(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        before_model = source.index("model_hashes_before =")
        before_tokenizer = source.index("tokenizer_hashes_before =")
        load_tokenizer = source.index("AutoTokenizer.from_pretrained")
        load_model = source.index("AutoModelForCausalLM.from_pretrained")
        after_model = source.index("model_hashes_after =")
        after_tokenizer = source.index("tokenizer_hashes_after =")
        self.assertLess(before_model, load_tokenizer)
        self.assertLess(before_tokenizer, load_tokenizer)
        self.assertLess(load_tokenizer, after_model)
        self.assertLess(load_model, after_model)
        self.assertLess(load_model, after_tokenizer)
        self.assertIn("model_hashes_before != model_hashes_after", source)
        self.assertIn("tokenizer_hashes_before != tokenizer_hashes_after", source)

    def test_sdpa_policy_is_reasserted_and_verified_per_forward(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        force = source.index("self._force_sdpa_math_policy()")
        forward = source.index("self._base_model(")
        verify = source.index("self._assert_sdpa_math_policy()")
        self.assertLess(force, forward)
        self.assertLess(forward, verify)

    def test_ambient_autocast_is_rejected_before_forward(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        self.assertLess(
            source.index("self._assert_autocast_disabled()"),
            source.index("self._base_model("),
        )

    def test_mps_fallback_and_fast_math_are_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._device_type = "mps"
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, clear=False):
            with self.assertRaisesRegex(CaptureContractError, "MPS_FALLBACK"):
                backend._assert_mps_execution_policy()
        with patch.dict(os.environ, {"PYTORCH_MPS_FAST_MATH": "1"}, clear=False):
            with self.assertRaisesRegex(CaptureContractError, "MPS_FAST_MATH"):
                backend._assert_mps_execution_policy()

    def test_seed_is_bounded_to_torch_manual_seed_range(self):
        request = fixture_request()
        request["determinism"]["seed"] = (1 << 64) - 1
        validate_capture_request(request)
        request["determinism"]["seed"] = 1 << 64
        with self.assertRaisesRegex(CaptureContractError, "torch.manual_seed"):
            validate_capture_request(request)

        request_schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
        )
        manifest_schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(request_schema["$defs"]["determinism"]["properties"]["seed"]["maximum"], (1 << 64) - 1)
        self.assertEqual(manifest_schema["$defs"]["determinism"]["properties"]["seed"]["maximum"], (1 << 64) - 1)

    def test_cuda_requests_require_explicit_device_index(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda"
        with self.assertRaisesRegex(CaptureContractError, "cuda:N"):
            validate_capture_request(request)
        request["backend"]["device"] = "cuda:0"
        self.assertEqual(validate_capture_request(request)["backend"]["device"], "cuda:0")

        schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
        )
        branches = schema["$defs"]["backend"]["properties"]["device"]["oneOf"]
        cuda_pattern = next(branch["pattern"] for branch in branches if "pattern" in branch)
        self.assertEqual(cuda_pattern, "^cuda:[0-9]+$")

    def test_boolean_step_and_layer_indices_are_rejected(self):
        request = fixture_request()
        manifest, trajectory = execute_capture(
            request,
            implementation_revision=REV,
            backend=FakeBackend(request),
        )
        trajectory["steps"][0]["step_index"] = False
        rehash_trajectory_and_manifest(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "step_index"):
            verify_capture_bundle(request, manifest, trajectory)

        manifest, trajectory = execute_capture(
            request,
            implementation_revision=REV,
            backend=FakeBackend(request),
        )
        trajectory["steps"][0]["layers"][0]["layer_index"] = False
        rehash_trajectory_and_manifest(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "layer_index"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_boolean_token_count_cannot_masquerade_as_one(self):
        request = fixture_request()
        request["capture"]["prefix_text"] = ""
        request["capture"]["step_joiner"] = ""
        request["capture"]["layers"] = [0]
        request["steps"] = [{"step_id": "s0", "text": "A"}]
        manifest, trajectory = execute_capture(
            request,
            implementation_revision=REV,
            backend=FakeBackend(request),
        )
        self.assertEqual(trajectory["steps"][0]["token_count"], 1)
        trajectory["steps"][0]["token_count"] = True
        rehash_trajectory_and_manifest(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "token_count"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_manifest_schema_records_new_runtime_policy_fields(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        production = schema["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        for field in (
            "cuda_resolved_device_index", "cuda_device_uuid", "cuda_visible_devices",
            "sdpa_flash_enabled", "sdpa_mem_efficient_enabled", "sdpa_math_enabled",
            "sdpa_cudnn_enabled", "mps_fallback_env", "mps_fast_math_env",
            "autocast_disabled",
        ):
            self.assertIn(field, required)
        self.assertEqual(
            production["properties"]["snapshot_authentication"]["const"],
            "sha256_all_snapshot_files_pre_and_post_load",
        )


if __name__ == "__main__":
    unittest.main()
