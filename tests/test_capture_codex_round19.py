from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_execute import execute_capture
from qsol_geo_reason.capture_publish import write_capture_bundle

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class FakeConfig:
    def __init__(self):
        self.flag = "canonical"

    def to_dict(self):
        return {"flag": self.flag}


class FakeChild:
    def forward(self, value=None):
        return value


class FakeGraphModel:
    def __init__(self):
        self.child = FakeChild()
        self.config = FakeConfig()

    def forward(self, value=None):
        return self.child.forward(value)

    def named_modules(self):
        return [("", self), ("child", self.child)]


class FakeSlowTokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "b": 1}
        self.backend_tokenizer = None
        self.added_tokens_encoder = {}
        self.special_tokens_map = {"eos_token": "b"}
        self.all_special_tokens = ["b"]
        self.all_special_ids = [1]
        self.init_kwargs = {}
        self.add_prefix_space = False
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 1024

    def get_vocab(self):
        return dict(self.vocab)


class FakeProcessTorch:
    def __init__(self):
        self.cpu_rng = b"ambient-cpu-rng"
        self.deterministic = False
        self.warn_only = True
        self.default_generator = SimpleNamespace(manual_seed=self.manual_seed)
        self.cuda = SimpleNamespace(
            is_available=lambda: False,
        )
        self.mps = SimpleNamespace(
            is_available=lambda: False,
        )
        self.xpu = SimpleNamespace(
            is_available=lambda: False,
        )

    def get_rng_state(self):
        return self.cpu_rng

    def set_rng_state(self, state):
        self.cpu_rng = state

    def manual_seed(self, seed):
        self.cpu_rng = f"seed:{seed}".encode("ascii")

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, enabled, *, warn_only=False):
        self.deterministic = bool(enabled)
        self.warn_only = bool(warn_only)


class CaptureRound19RegressionTests(unittest.TestCase):
    def test_executable_model_graph_seal_detects_forward_monkeypatch(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeGraphModel()
        original = backend._model_executable_state_seal()
        backend._model.child.forward = lambda value=None: ("patched", value)
        self.assertNotEqual(backend._model_executable_state_seal(), original)

        backend._model = FakeGraphModel()
        original = backend._model_executable_state_seal()
        backend._model.config.flag = "mutated"
        self.assertNotEqual(backend._model_executable_state_seal(), original)

    def test_slow_tokenizer_live_behavior_is_sealed(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._tokenizer = FakeSlowTokenizer()
        original = backend._tokenizer_live_state_seal()
        self.assertEqual(backend._tokenizer.init_kwargs, {})
        backend._tokenizer.add_prefix_space = True
        self.assertEqual(backend._tokenizer.init_kwargs, {})
        self.assertNotEqual(backend._tokenizer_live_state_seal(), original)

    def test_observation_backend_is_single_use(self):
        request = fixture_request()
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._applied_seed = request["determinism"]["seed"]
        backend._determinism_mode = request["determinism"]["mode"]
        backend._model_identifier = request["model"]["identifier"]
        backend._model_revision = request["model"]["revision"]
        backend._tokenizer_identifier = request["model"]["tokenizer_identifier"]
        backend._tokenizer_revision = request["model"]["tokenizer_revision"]
        backend._device = request["backend"]["device"]
        backend._dtype_name = request["backend"]["dtype"]
        backend._live_state_seal_initialized = False
        backend._observation_consumed = True
        with self.assertRaisesRegex(CaptureContractError, "single-use"):
            backend.assert_execution_request(request)

    def test_observation_session_restores_host_rng_and_determinism(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeProcessTorch()
        backend._device = "cpu"
        backend._applied_seed = 17
        backend._determinism_mode = "required"
        backend._canonical_deterministic_algorithms_enabled = True
        backend._canonical_deterministic_warn_only_enabled = False
        backend._last_deterministic_algorithms_enabled = None
        backend._live_state_seal_initialized = False
        backend._observation_consumed = False
        backend._observation_active = False
        backend._observation_ambient_process_state = None

        ambient_rng = backend._torch.cpu_rng
        ambient_deterministic = backend._torch.deterministic
        ambient_warn = backend._torch.warn_only
        backend.begin_observation()
        self.assertEqual(backend._torch.cpu_rng, b"seed:17")
        self.assertIs(backend._torch.deterministic, True)
        self.assertIs(backend._torch.warn_only, False)
        backend.end_observation()
        self.assertEqual(backend._torch.cpu_rng, ambient_rng)
        self.assertIs(backend._torch.deterministic, ambient_deterministic)
        self.assertIs(backend._torch.warn_only, ambient_warn)

        source = inspect.getsource(execute_capture)
        self.assertIn("backend.begin_observation()", source)
        self.assertIn("finally:", source)
        self.assertIn("backend.end_observation()", source)
        init_source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertIn("finally:", init_source)
        self.assertIn("self._restore_torch_process_state", init_source)

    def test_publisher_writes_private_snapshots_not_later_caller_mutations(self):
        request = {"request": "before"}
        manifest = {"manifest": "before"}
        trajectory = {"trajectory": "before"}

        def verify_side_effect(request_snapshot, manifest_snapshot, trajectory_snapshot):
            self.assertIsNot(request_snapshot, request)
            self.assertIsNot(manifest_snapshot, manifest)
            self.assertIsNot(trajectory_snapshot, trajectory)
            manifest["manifest"] = "after"
            trajectory["trajectory"] = "after"
            return request_snapshot

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            with patch(
                "qsol_geo_reason.capture_publish.verify_capture_bundle",
                side_effect=verify_side_effect,
            ):
                write_capture_bundle(output, request, manifest, trajectory)

            written_manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
            written_trajectory = json.loads(
                (output / "captured-trajectory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_manifest, {"manifest": "before"})
            self.assertEqual(written_trajectory, {"trajectory": "before"})
            self.assertEqual(manifest, {"manifest": "after"})
            self.assertEqual(trajectory, {"trajectory": "after"})


if __name__ == "__main__":
    unittest.main()
