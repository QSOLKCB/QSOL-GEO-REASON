from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.capture import (
    CaptureContractError,
    _quantization_reasons,
    _validate_loading_info,
    execute_capture,
    validate_capture_request,
    verify_capture_bundle,
    write_capture_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
FIXTURE_EXPECTED = ROOT / "fixtures" / "capture-contract-expected.json"
IMPLEMENTATION_REVISION = "d" * 40


class FakeBackend:
    """Software-only test double. It is not an empirical model backend."""

    def __init__(self, request: dict):
        self.request = request
        self.requested_layer_calls: list[tuple[int, ...]] = []

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def hidden_states(self, input_ids, layer_indices):
        self.requested_layer_calls.append(tuple(layer_indices))
        if any(index < 0 or index >= 3 for index in layer_indices):
            bad = next(index for index in layer_indices if index < 0 or index >= 3)
            raise CaptureContractError(
                f"requested layer {bad} outside backend hidden_states range [0, 2]"
            )
        return {
            layer_index: [
                [float(token_id + layer_index), float(token_id * (layer_index + 1))]
                for token_id in input_ids
            ]
            for layer_index in layer_indices
        }

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


class BadBackend(FakeBackend):
    def __init__(self, request: dict, *, metadata_updates=None, state_mutator=None):
        super().__init__(request)
        self.metadata_updates = metadata_updates or {}
        self.state_mutator = state_mutator

    def metadata(self):
        value = dict(super().metadata())
        value.update(self.metadata_updates)
        return value

    def hidden_states(self, input_ids, layer_indices):
        states = super().hidden_states(input_ids, layer_indices)
        if self.state_mutator is not None:
            self.state_mutator(states)
        return states


def fixture_request() -> dict:
    return json.loads(FIXTURE_REQUEST.read_text(encoding="utf-8"))


def execute(request: dict, backend=None):
    selected_backend = backend or FakeBackend(request)
    return execute_capture(
        request,
        implementation_revision=IMPLEMENTATION_REVISION,
        backend=selected_backend,
    )


class CaptureValidationTests(unittest.TestCase):
    def test_fixture_request_validates_without_mutating_caller(self):
        request = fixture_request()
        request["model"]["revision"] = "A" * 40
        validated = validate_capture_request(request)
        self.assertEqual(validated["protocol_id"], "GEO-CAP-001")
        self.assertEqual(validated["model"]["revision"], "a" * 40)
        self.assertEqual(request["model"]["revision"], "A" * 40)

    def test_non_object_json_roots_are_rejected(self):
        for root in (None, [], [["protocol_id", "GEO-CAP-001"]], "text", 1):
            with self.subTest(root=root):
                with self.assertRaisesRegex(CaptureContractError, "must be an object"):
                    validate_capture_request(root)  # type: ignore[arg-type]

    def test_unknown_root_field_is_rejected(self):
        request = fixture_request()
        request["mystery"] = 1
        with self.assertRaisesRegex(CaptureContractError, "unknown fields"):
            validate_capture_request(request)

    def test_floating_model_revision_is_rejected(self):
        request = fixture_request()
        request["model"]["revision"] = "main"
        with self.assertRaisesRegex(CaptureContractError, "40-hex"):
            validate_capture_request(request)

    def test_model_revision_kind_must_be_hf_commit(self):
        request = fixture_request()
        request["model"]["revision_kind"] = "tag"
        with self.assertRaisesRegex(CaptureContractError, "revision_kind"):
            validate_capture_request(request)

    def test_tokenizer_revision_kind_must_be_hf_commit(self):
        request = fixture_request()
        request["model"]["tokenizer_revision_kind"] = "branch"
        with self.assertRaisesRegex(CaptureContractError, "tokenizer_revision_kind"):
            validate_capture_request(request)

    def test_canonical_backend_requires_local_only(self):
        request = fixture_request()
        request["backend"]["local_files_only"] = False
        with self.assertRaisesRegex(CaptureContractError, "local_files_only"):
            validate_capture_request(request)

    def test_canonical_backend_forbids_remote_code(self):
        request = fixture_request()
        request["backend"]["trust_remote_code"] = True
        with self.assertRaisesRegex(CaptureContractError, "trust_remote_code"):
            validate_capture_request(request)

    def test_canonical_backend_forbids_declared_quantization(self):
        request = fixture_request()
        request["backend"]["quantization"] = "int8"
        with self.assertRaisesRegex(CaptureContractError, "quantization"):
            validate_capture_request(request)

    def test_duplicate_layers_are_rejected(self):
        request = fixture_request()
        request["capture"]["layers"] = [1, 1]
        with self.assertRaisesRegex(CaptureContractError, "unique"):
            validate_capture_request(request)

    def test_bounded_pooling_requires_positive_window(self):
        request = fixture_request()
        request["capture"]["pooling"] = {
            "mode": "bounded_context_mean",
            "window_tokens": 0,
        }
        with self.assertRaisesRegex(CaptureContractError, ">= 1"):
            validate_capture_request(request)


class CheckpointLoadingTests(unittest.TestCase):
    def test_clean_loading_info_is_accepted(self):
        _validate_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )

    def test_missing_or_unexpected_weights_are_rejected(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            info = {name: [] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
            info[key] = ["bad-entry"]
            with self.subTest(key=key):
                with self.assertRaisesRegex(CaptureContractError, "not exact"):
                    _validate_loading_info(info)

    def test_embedded_quantization_signals_are_rejected(self):
        clean = SimpleNamespace(
            config=SimpleNamespace(quantization_config=None),
            is_quantized=False,
            hf_quantizer=None,
        )
        self.assertEqual(_quantization_reasons(clean), [])

        cases = [
            SimpleNamespace(
                config=SimpleNamespace(quantization_config={"bits": 8}),
                is_quantized=False,
                hf_quantizer=None,
            ),
            SimpleNamespace(
                config=SimpleNamespace(quantization_config=None),
                is_quantized=True,
                hf_quantizer=None,
            ),
            SimpleNamespace(
                config=SimpleNamespace(quantization_config=None),
                is_quantized=False,
                hf_quantizer=object(),
            ),
        ]
        for model in cases:
            with self.subTest(model=model):
                self.assertTrue(_quantization_reasons(model))


class CaptureExecutionTests(unittest.TestCase):
    def test_test_double_defaults_to_simulation(self):
        request = fixture_request()
        _, trajectory = execute(request)
        self.assertEqual(trajectory["evidence_class"], "SIMULATION")

    def test_test_double_cannot_emit_observation(self):
        request = fixture_request()
        with self.assertRaisesRegex(CaptureContractError, "concrete HuggingFace"):
            execute_capture(
                request,
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=FakeBackend(request),
                evidence_class="OBSERVATION",
            )

    def test_backend_receives_only_requested_layers(self):
        request = fixture_request()
        request["capture"]["layers"] = [2]
        backend = FakeBackend(request)
        execute(request, backend=backend)
        self.assertEqual(backend.requested_layer_calls, [(2,), (2,)])

    def test_cumulative_token_spans_are_explicit(self):
        request = fixture_request()
        _, trajectory = execute(request)
        spans = [step["changed_token_span"] for step in trajectory["steps"]]
        self.assertEqual(spans, [[1, 4], [4, 7]])
        self.assertEqual(trajectory["steps"][0]["input_ids"], [80, 124, 65, 66])
        self.assertEqual(
            trajectory["steps"][1]["input_ids"],
            [80, 124, 65, 66, 124, 67, 68],
        )

    def test_isolated_mode_uses_prefix_baseline(self):
        request = fixture_request()
        request["capture"]["context_mode"] = "isolated"
        _, trajectory = execute(request)
        self.assertEqual(
            [step["changed_token_span"] for step in trajectory["steps"]],
            [[1, 4], [1, 4]],
        )

    def test_step_mean_uses_changed_span(self):
        request = fixture_request()
        _, trajectory = execute(request)
        layer0 = trajectory["steps"][0]["layers"][0]
        self.assertEqual(layer0["pool_span"], [1, 4])
        self.assertEqual(layer0["vector"], [85.0, 85.0])

    def test_last_token_pooling_is_exact(self):
        request = fixture_request()
        request["capture"]["pooling"] = {"mode": "last_token"}
        _, trajectory = execute(request)
        layer0 = trajectory["steps"][0]["layers"][0]
        self.assertEqual(layer0["pool_span"], [3, 4])
        self.assertEqual(layer0["vector"], [66.0, 66.0])

    def test_bounded_context_pooling_records_window(self):
        request = fixture_request()
        request["capture"]["pooling"] = {
            "mode": "bounded_context_mean",
            "window_tokens": 2,
        }
        _, trajectory = execute(request)
        layer0 = trajectory["steps"][1]["layers"][0]
        self.assertEqual(layer0["pool_span"], [5, 7])
        self.assertEqual(layer0["vector"], [67.5, 67.5])

    def test_frozen_software_fixture_hashes(self):
        request = fixture_request()
        expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))
        manifest, trajectory = execute(request)
        self.assertEqual(expected["evidence_class"], "SIMULATION")
        self.assertEqual(trajectory["evidence_class"], "SIMULATION")
        self.assertEqual(manifest["run_manifest_id"], expected["expected_run_manifest_id"])
        self.assertEqual(manifest["manifest_sha256"], expected["expected_manifest_sha256"])
        self.assertEqual(trajectory["trajectory_sha256"], expected["expected_trajectory_sha256"])
        self.assertEqual(
            [step["changed_token_span"] for step in trajectory["steps"]],
            expected["expected_changed_token_spans"],
        )

    def test_hashes_are_stable_and_content_bound(self):
        request = fixture_request()
        manifest_a, trajectory_a = execute(copy.deepcopy(request))
        manifest_b, trajectory_b = execute(copy.deepcopy(request))
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(trajectory_a, trajectory_b)
        changed = copy.deepcopy(request)
        changed["steps"][1]["text"] = "CE"
        changed_manifest, changed_trajectory = execute(changed)
        self.assertNotEqual(
            trajectory_a["trajectory_sha256"], changed_trajectory["trajectory_sha256"]
        )
        self.assertNotEqual(
            manifest_a["manifest_sha256"], changed_manifest["manifest_sha256"]
        )

    def test_missing_requested_layer_is_rejected(self):
        request = fixture_request()
        request["capture"]["layers"] = [3]
        with self.assertRaisesRegex(CaptureContractError, "outside"):
            execute(request)

    def test_nonfinite_hidden_state_is_rejected(self):
        request = fixture_request()

        def poison(states):
            states[0][0][0] = math.inf

        with self.assertRaisesRegex(CaptureContractError, "non-finite"):
            execute_capture(
                request,
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=BadBackend(request, state_mutator=poison),
            )

    def test_observed_model_revision_mismatch_is_rejected(self):
        request = fixture_request()
        with self.assertRaisesRegex(CaptureContractError, "observed model commit"):
            execute_capture(
                request,
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=BadBackend(
                    request,
                    metadata_updates={"observed_model_commit": "3" * 40},
                ),
            )

    def test_observed_tokenizer_revision_mismatch_is_rejected(self):
        request = fixture_request()
        with self.assertRaisesRegex(CaptureContractError, "observed tokenizer commit"):
            execute_capture(
                request,
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=BadBackend(
                    request,
                    metadata_updates={"observed_tokenizer_commit": "3" * 40},
                ),
            )

    def test_backend_identity_mismatch_is_rejected(self):
        request = fixture_request()
        with self.assertRaisesRegex(CaptureContractError, "backend metadata name"):
            execute_capture(
                request,
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=BadBackend(
                    request,
                    metadata_updates={"name": "not-the-canonical-backend"},
                ),
            )


class CaptureBundleTests(unittest.TestCase):
    def test_valid_bundle_verifies_and_writes(self):
        request = fixture_request()
        manifest, trajectory = execute(request)
        validated = verify_capture_bundle(request, manifest, trajectory)
        self.assertEqual(validated, request)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_capture_bundle(output, request, manifest, trajectory)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["capture-request.json", "captured-trajectory.json", "run-manifest.json"],
            )

    def test_cross_run_trajectory_is_rejected(self):
        request = fixture_request()
        manifest, trajectory = execute(request)
        changed = copy.deepcopy(request)
        changed["steps"][1]["text"] = "CE"
        _, changed_trajectory = execute(changed)
        with self.assertRaisesRegex(CaptureContractError, "trajectory hash"):
            verify_capture_bundle(request, manifest, changed_trajectory)

    def test_tampered_trajectory_hash_is_rejected(self):
        request = fixture_request()
        manifest, trajectory = execute(request)
        trajectory["steps"][0]["step_id"] = "tampered"
        with self.assertRaisesRegex(CaptureContractError, "trajectory SHA-256"):
            verify_capture_bundle(request, manifest, trajectory)

    def test_tampered_manifest_hash_is_rejected(self):
        request = fixture_request()
        manifest, trajectory = execute(request)
        manifest["run_id"] = "other-run"
        with self.assertRaisesRegex(CaptureContractError, "manifest SHA-256"):
            verify_capture_bundle(request, manifest, trajectory)


if __name__ == "__main__":
    unittest.main()
