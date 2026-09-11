from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason.capture_common import (
    CaptureBackendUnavailable,
    CaptureContractError,
)
from qsol_geo_reason.provenance import SourceIdentityError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_first_production_observation.py"
SPEC = importlib.util.spec_from_file_location("qsol_first_observation_tool", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class FirstProductionObservationTests(unittest.TestCase):
    def _materialized_request(self) -> dict:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return TOOL._assert_exact_experiment_request(request, require_receipts=True)

    def test_template_freezes_small_cpu_reference_model_and_capture_definition(self) -> None:
        request = TOOL._load_template()
        self.assertEqual(request["model"]["identifier"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(
            request["model"]["revision"],
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        )
        self.assertEqual(request["model"]["tokenizer_identifier"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(
            request["model"]["tokenizer_revision"],
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        )
        self.assertEqual(request["backend"]["device"], "cpu")
        self.assertEqual(request["backend"]["dtype"], "float32")
        self.assertEqual(request["capture"]["layers"], [0, 6, 12, 18, 24])
        self.assertEqual(request["capture"]["pooling"], {"mode": "step_mean"})
        self.assertEqual(request["determinism"], {"mode": "required", "seed": 20260912})
        self.assertNotIn("revision_tree_sha256", request["model"])
        self.assertNotIn("tokenizer_revision_tree_sha256", request["model"])

    def test_only_authenticated_tree_receipts_may_materialize_after_preregistration(self) -> None:
        final = self._materialized_request()
        self.assertEqual(final["model"]["revision_tree_sha256"], "1" * 64)
        self.assertEqual(final["model"]["tokenizer_revision_tree_sha256"], "2" * 64)

        final["capture"]["layers"] = [0, 12, 24]
        with self.assertRaises(CaptureContractError):
            TOOL._assert_exact_experiment_request(final, require_receipts=True)

    def test_prepare_writes_one_final_request_and_refuses_overwrite(self) -> None:
        receipts = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "final-request.json"
            with mock.patch.object(TOOL, "prepare_tree_receipts", return_value=receipts):
                request_sha256 = TOOL.prepare(output)
                written = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(written["model"]["revision_tree_sha256"], "a" * 64)
                self.assertEqual(written["model"]["tokenizer_revision_tree_sha256"], "b" * 64)
                self.assertEqual(request_sha256, TOOL.sha256_json(written))
                with self.assertRaises(CaptureContractError):
                    TOOL.prepare(output)

    def test_failed_payload_sync_does_not_publish_partial_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "request.json"
            with mock.patch.object(TOOL.os, "fsync", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(CaptureContractError, "unable to persist"):
                    TOOL._exclusive_write_json(output, {"value": 1})
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob(".request.json.tmp.*")), [])

    def test_output_root_creation_durably_publishes_every_new_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output_root = base / "level-a" / "level-b" / "observation"
            synced: list[Path] = []

            with mock.patch.object(
                TOOL,
                "_fsync_directory",
                side_effect=lambda path: synced.append(Path(path)),
            ):
                TOOL._create_output_root_durable(output_root)

            self.assertTrue(output_root.is_dir())
            self.assertEqual(
                synced,
                [
                    base,
                    base / "level-a",
                    base / "level-a" / "level-b",
                ],
            )

    def test_observe_snapshots_request_before_launching_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(original), encoding="utf-8")
            output_root = directory / "observation"
            seen_paths: list[Path] = []
            seen_revisions: list[str | None] = []
            repository_commit = "f" * 40

            def fake_run(path: Path, output: Path, revision: str | None) -> str:
                seen_paths.append(path)
                seen_revisions.append(revision)
                if len(seen_paths) == 1:
                    changed = json.loads(json.dumps(original))
                    changed["model"]["revision_tree_sha256"] = "9" * 64
                    request_path.write_text(json.dumps(changed), encoding="utf-8")
                return "a" * 64

            verdict = {"replay_outcome": "byte_identical"}
            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ) as resolver,
                mock.patch.object(TOOL, "_run_capture", side_effect=fake_run),
                mock.patch.object(TOOL, "build_replay_verdict", return_value=verdict),
                mock.patch.object(TOOL, "verify_replay_verdict", return_value=verdict),
            ):
                observed, status = TOOL.observe(request_path, output_root, None)

            resolver.assert_called_once_with(None, require_checkout=True)
            self.assertEqual(status, 0)
            self.assertEqual(observed, verdict)
            self.assertEqual(len(seen_paths), 2)
            self.assertEqual(seen_paths[0], seen_paths[1])
            self.assertEqual(seen_revisions, [repository_commit, repository_commit])
            self.assertNotEqual(seen_paths[0], request_path)
            snapshot = json.loads(
                (output_root / "validated-request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot, original)
            self.assertNotEqual(json.loads(request_path.read_text(encoding="utf-8")), original)

    def test_failed_observation_is_bound_to_revision_and_retry_path_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request_path = directory / "request.json"
            request_path.write_text(
                json.dumps(self._materialized_request()), encoding="utf-8"
            )
            output_root = directory / "missing-parent" / "observation"
            repository_commit = "e" * 40

            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ) as resolver,
                mock.patch.object(
                    TOOL,
                    "_run_capture",
                    side_effect=CaptureContractError("transient backend failure"),
                ) as run_capture,
            ):
                with self.assertRaisesRegex(
                    CaptureContractError, "incomplete evidence preserved"
                ):
                    TOOL.observe(request_path, output_root, None)

            resolver.assert_called_once_with(None, require_checkout=True)
            self.assertEqual(run_capture.call_count, 1)
            self.assertEqual(run_capture.call_args.args[2], repository_commit)
            self.assertFalse(output_root.exists())
            failed = list((directory / "missing-parent").glob("observation.failed-*"))
            self.assertEqual(len(failed), 1)
            marker = json.loads(
                (failed[0] / "execution-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["repository_commit"], repository_commit)
            self.assertEqual(marker["attempt_status"], "failed_before_replay_verdict")
            self.assertEqual(marker["completed_run_directories"], [])
            self.assertTrue((failed[0] / "validated-request.json").is_file())

    def test_revision_resolution_failure_creates_no_attempt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request_path = directory / "request.json"
            request_path.write_text(
                json.dumps(self._materialized_request()), encoding="utf-8"
            )
            output_root = directory / "observation"

            with mock.patch.object(
                TOOL,
                "resolve_implementation_revision",
                side_effect=SourceIdentityError("checkout is dirty"),
            ):
                with self.assertRaisesRegex(
                    CaptureContractError, "unable to bind production observation"
                ):
                    TOOL.observe(request_path, output_root, None)

            self.assertFalse(output_root.exists())

    def test_prepare_missing_capture_dependency_uses_argparse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "request.json"
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    TOOL,
                    "prepare_tree_receipts",
                    side_effect=CaptureBackendUnavailable(
                        "install qsol-geo-reason[capture]"
                    ),
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(SCRIPT),
                        "prepare",
                        "--output",
                        str(output),
                    ],
                ),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    TOOL.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("install qsol-geo-reason[capture]", stderr.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
