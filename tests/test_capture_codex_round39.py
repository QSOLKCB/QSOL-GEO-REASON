"""Round 39 regressions for Hub warm-up, CUDA placement, signals, and CUDA libraries."""
from __future__ import annotations

import hashlib
import inspect
import json
import signal
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as ProductionBackend
from qsol_geo_reason.capture_backend_round39 import HuggingFacePyTorchBackend as Round39Backend
from qsol_geo_reason.capture_cuda_runtime import _cuda_runtime_library_receipt
from qsol_geo_reason.capture_hub_tree import prepare_tree_receipts
from qsol_geo_reason.capture_runtime import _validate_torch_build_metadata
from qsol_geo_reason.capture_signals import _assert_no_async_signal_instrumentation
from qsol_geo_reason.capture_common import CaptureContractError
from test_capture_codex_round6 import fixture_request

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound39RegressionTests(unittest.TestCase):
    def test_online_warmup_generates_qsol_tree_artifact_and_receipt(self):
        request = fixture_request()
        request["model"].pop("revision_tree_sha256", None)
        request["model"].pop("tokenizer_revision_tree_sha256", None)
        model_commit = request["model"]["revision"]
        tokenizer_commit = request["model"]["tokenizer_revision"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "models--fixture" / "snapshots"
            for commit in {model_commit, tokenizer_commit}:
                (snapshots / commit).mkdir(parents=True)

            class Info:
                def __init__(self, sha):
                    self.sha = sha

            class Api:
                def model_info(self, *, repo_id, revision):
                    return Info(revision)

                def list_repo_tree(self, *, repo_id, revision, recursive):
                    return [
                        {"path": "config.json", "size": 2, "blob_id": "1" * 40},
                        {"path": "model.safetensors", "size": 3, "blob_id": "2" * 40},
                    ]

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = Api
            fake_hub.snapshot_download = lambda **kwargs: str(snapshots / kwargs["revision"])
            with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                receipts = prepare_tree_receipts(request)

            for field, commit in (
                ("revision_tree_sha256", model_commit),
                ("tokenizer_revision_tree_sha256", tokenizer_commit),
            ):
                tree_path = snapshots.parent / "trees" / f"{commit}.json"
                self.assertTrue(tree_path.is_file())
                self.assertEqual(hashlib.sha256(tree_path.read_bytes()).hexdigest(), receipts[field])
                payload = json.loads(tree_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["format_version"], 1)
                self.assertIn("model.safetensors", payload["files"])

    def test_cli_and_example_document_generated_tree_receipt_workflow(self):
        from qsol_geo_reason.capture_cli import main as cli_main

        source = inspect.getsource(cli_main)
        self.assertIn('"--prepare-tree-receipts"', source)
        self.assertIn("prepare_tree_receipts(validated)", source)
        template = json.loads((ROOT / "examples" / "GEO-CAP-001.example.json").read_text(encoding="utf-8"))
        notes = template["notes"]
        self.assertIn("--prepare-tree-receipts", notes)
        self.assertIn("does NOT create", notes)

    def test_schema_requires_nonnegative_cuda_index_and_null_elsewhere(self):
        schema = json.loads((ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8"))
        rules = schema["$defs"]["backendObservedProduction"]["allOf"]
        matching = []
        for rule in rules:
            device = rule.get("if", {}).get("properties", {}).get("device", {})
            then_index = rule.get("then", {}).get("properties", {}).get("cuda_resolved_device_index")
            else_index = rule.get("else", {}).get("properties", {}).get("cuda_resolved_device_index")
            if device.get("pattern") == "^cuda:[0-9]+$" and then_index is not None:
                matching.append((then_index, else_index))
        self.assertEqual(len(matching), 1)
        then_index, else_index = matching[0]
        self.assertEqual(then_index, {"type": "integer", "minimum": 0})
        self.assertEqual(else_index, {"type": "null"})

    def test_custom_signal_handler_is_rejected(self):
        custom = lambda *_args: None
        sigint = signal.SIGINT
        sigterm = signal.SIGTERM

        def getsignal(signum):
            return signal.default_int_handler if signum == sigint else custom

        with mock.patch("signal.valid_signals", return_value={sigint, sigterm}), mock.patch(
            "signal.getsignal", side_effect=getsignal
        ), mock.patch("signal.getitimer", return_value=(0.0, 0.0)):
            with self.assertRaisesRegex(CaptureContractError, "custom asynchronous signal handler"):
                _assert_no_async_signal_instrumentation()

    def test_armed_interval_timer_is_rejected(self):
        sigint = signal.SIGINT
        timer_values = [value for value in (getattr(signal, "ITIMER_REAL", None), getattr(signal, "ITIMER_VIRTUAL", None), getattr(signal, "ITIMER_PROF", None)) if value is not None]
        if not timer_values:
            self.skipTest("interval timers are unavailable on this platform")

        def getitimer(timer):
            return (0.25, 0.0) if timer == timer_values[0] else (0.0, 0.0)

        with mock.patch("signal.valid_signals", return_value={sigint}), mock.patch(
            "signal.getsignal", return_value=signal.default_int_handler
        ), mock.patch("signal.getitimer", side_effect=getitimer):
            with self.assertRaisesRegex(CaptureContractError, "armed asynchronous interval timer"):
                _assert_no_async_signal_instrumentation()

    def test_signal_guard_dispatches_inside_existing_exclusive_boundary(self):
        production = inspect.getsource(ProductionBackend.begin_observation)
        self.assertLess(
            production.index("self._enter_exclusive_python_thread_boundary()"),
            production.index("self._assert_no_active_torch_override_modes()"),
        )
        round39 = inspect.getsource(Round39Backend._assert_no_active_torch_override_modes)
        self.assertIn("_assert_no_async_signal_instrumentation()", round39)

    def test_loaded_cuda_library_receipt_is_content_bound_and_relocatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / "libcuda.so.1"
            cublas = root / "libcublas.so.12"
            driver.write_bytes(b"driver-A")
            cublas.write_bytes(b"cublas-A")
            count_a, receipt_a = _cuda_runtime_library_receipt([driver, cublas])
            self.assertEqual(count_a, 2)
            self.assertRegex(receipt_a, r"^[0-9a-f]{64}$")

            cublas.write_bytes(b"cublas-B")
            count_b, receipt_b = _cuda_runtime_library_receipt([driver, cublas])
            self.assertEqual(count_b, 2)
            self.assertNotEqual(receipt_a, receipt_b)

    def test_cuda_library_receipt_is_embedded_in_authenticated_runtime_config(self):
        source = inspect.getsource(Round39Backend.metadata)
        self.assertIn("loaded_cuda_runtime_library_provenance()", source)
        self.assertIn("QSOL_GEO_CUDA_RUNTIME=", source)
        config = 'SYNTHETIC\nQSOL_GEO_CUDA_RUNTIME={"loaded_cuda_runtime_libraries":{"cuda_runtime_library_file_count":2,"cuda_runtime_library_receipt_sha256":"' + "a" * 64 + '"}}\n'
        observed = {
            "torch_build_config": config,
            "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        }
        _validate_torch_build_metadata(observed)


if __name__ == "__main__":
    unittest.main()
