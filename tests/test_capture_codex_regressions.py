from __future__ import annotations

import copy
import inspect
import json
import math
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend,
    _resolve_hidden_state_layout,
    execute_capture,
    verify_capture_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
IMPLEMENTATION_REVISION = "d" * 40


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
            "device": "cpu",
            "dtype": "float32",
            "quantization": "none",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_cache": False,
            "capture_phase": "replayed_prefix",
            "kv_cache_reuse": False,
            "determinism_mode": "required",
        }


def fixture_request() -> dict:
    return json.loads(FIXTURE_REQUEST.read_text(encoding="utf-8"))


def execute_fixture():
    request = fixture_request()
    manifest, trajectory = execute_capture(
        request,
        implementation_revision=IMPLEMENTATION_REVISION,
        backend=FakeBackend(request),
    )
    return request, manifest, trajectory


def rehash_trajectory_and_manifest(manifest: dict, trajectory: dict) -> None:
    trajectory_payload = {
        key: value for key, value in trajectory.items() if key != "trajectory_sha256"
    }
    trajectory_sha = sha256_json(trajectory_payload)
    trajectory["trajectory_sha256"] = trajectory_sha
    manifest["artifacts"]["captured_trajectory_sha256"] = trajectory_sha
    rehash_manifest(manifest)


def rehash_manifest(manifest: dict) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256_json(payload)


class SelectiveCaptureRegressionTests(unittest.TestCase):
    def test_production_backend_does_not_request_full_hidden_state_tuple(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        self.assertIn("output_hidden_states=False", source)
        self.assertNotIn("output_hidden_states=True", source)
        self.assertIn("register_forward_pre_hook", source)
        self.assertIn("register_forward_hook", source)

    def test_pooling_moves_selected_span_to_cpu_before_float64(self):
        source = inspect.getsource(HuggingFacePyTorchBackend._pool_tensor_record)
        self.assertIn('to(device="cpu")', source)
        self.assertIn("to(dtype=torch.float64)", source)
        self.assertLess(
            source.index('to(device="cpu")'),
            source.index("to(dtype=torch.float64)"),
        )

    def test_hidden_state_layout_requires_one_config_bound_block_sequence(self):
        blocks = [object(), object(), object()]
        base = SimpleNamespace(layers=blocks)
        model = SimpleNamespace(
            base_model=base,
            config=SimpleNamespace(num_hidden_layers=3),
        )
        resolved_base, path, resolved_blocks = _resolve_hidden_state_layout(model)
        self.assertIs(resolved_base, base)
        self.assertEqual(path, "layers")
        self.assertIs(resolved_blocks, blocks)

        ambiguous = SimpleNamespace(layers=blocks, h=list(blocks))
        model = SimpleNamespace(
            base_model=ambiguous,
            config=SimpleNamespace(num_hidden_layers=3),
        )
        with self.assertRaisesRegex(CaptureContractError, "exactly one recognized"):
            _resolve_hidden_state_layout(model)


class BundleShapeRegressionTests(unittest.TestCase):
    def test_unknown_trajectory_root_field_is_rejected_even_after_rehash(self):
        request, manifest, trajectory = execute_fixture()
        trajectory["unknown"] = "self-authenticated"
        rehash_trajectory_and_manifest(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "unknown fields"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_unknown_manifest_root_field_is_rejected_even_after_rehash(self):
        request, manifest, trajectory = execute_fixture()
        manifest["unknown"] = "self-authenticated"
        rehash_manifest(manifest)
        with self.assertRaisesRegex(CaptureContractError, "unknown fields"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_unknown_artifact_field_is_rejected_even_after_rehash(self):
        request, manifest, trajectory = execute_fixture()
        manifest["artifacts"]["unknown"] = "self-authenticated"
        rehash_manifest(manifest)
        with self.assertRaisesRegex(CaptureContractError, "unknown fields"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_unknown_representation_field_is_rejected_even_after_rehash(self):
        request, manifest, trajectory = execute_fixture()
        trajectory["representation_definition"]["unknown"] = "self-authenticated"
        rehash_trajectory_and_manifest(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "unknown fields"):
            verify_capture_bundle(request, manifest, trajectory)


class SchemaParityRegressionTests(unittest.TestCase):
    def test_hub_id_schema_matches_runtime_terminal_character_rule(self):
        for path in (
            ROOT / "schemas" / "capture-request.schema.json",
            ROOT / "schemas" / "capture-run-manifest.schema.json",
        ):
            schema = json.loads(path.read_text(encoding="utf-8"))
            pattern = schema["$defs"]["hubRepoId"]["pattern"]
            with self.subTest(path=path.name):
                self.assertIsNotNone(re.fullmatch(pattern, "foo/bar"))
                self.assertIsNotNone(re.fullmatch(pattern, "foo/bar_"))
                self.assertIsNone(re.fullmatch(pattern, "foo/bar-"))
                self.assertIsNone(re.fullmatch(pattern, "foo/bar."))

    def test_run_manifest_schema_has_truthful_simulation_branch(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        branch = schema["$defs"]["backendObservedSimulation"]
        self.assertFalse(branch["additionalProperties"])
        self.assertEqual(
            set(branch["required"]),
            {
                "name",
                "observed_model_commit",
                "observed_tokenizer_commit",
                "device",
                "dtype",
                "quantization",
                "local_files_only",
                "trust_remote_code",
                "use_cache",
                "capture_phase",
                "kv_cache_reuse",
                "determinism_mode",
            },
        )
        _, manifest, _ = execute_fixture()
        self.assertEqual(set(manifest["backend_observed"]), set(branch["required"]))


if __name__ == "__main__":
    unittest.main()
