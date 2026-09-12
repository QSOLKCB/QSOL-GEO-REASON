from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import (
    CaptureBackendUnavailable,
    CaptureContractError,
)
from qsol_geo_reason.provenance import SourceIdentityError
from reference_environment_fixture import reference_environment_receipt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_first_production_observation.py"


class FirstProductionObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_environment = reference_environment_receipt()

        patcher = mock.patch.object(TOOL, "authenticate_tracked_file_against_revision")
        self.addCleanup(patcher.stop)
        self.template_auth = patcher.start()

        launcher_patcher = mock.patch.object(
            TOOL,
            "authenticate_tracked_tool_against_revision",
            return_value="1" * 40,
        )
        self.addCleanup(launcher_patcher.stop)
        self.launcher_auth = launcher_patcher.start()

        environment_patcher = mock.patch.object(
            TOOL,
            "verify_current_reference_environment",
            side_effect=lambda expected=None: dict(expected or self.reference_environment),
        )
        self.addCleanup(environment_patcher.stop)
        self.environment_auth = environment_patcher.start()

    def _materialized_request(self) -> dict:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return TOOL._assert_exact_experiment_request(request, require_receipts=True)

    def _write_preparation_receipt(
        self,
        request_path: Path,
        request: dict,
        repository_commit: str = "d" * 40,
    ) -> Path:
        receipt = TOOL.build_preparation_receipt(
            request=request,
            repository_commit=repository_commit,
            experiment_id=TOOL.EXPERIMENT_ID,
            reference_environment_receipt=self.reference_environment,
        )
        path = TOOL._default_preparation_receipt_path(request_path)
        path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path

    def test_tools_entrypoint_contains_no_evidence_orchestration(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"qsol_geo_reason.first_production_observation"', source)
        self.assertIn("os.execv", source)
        self.assertNotIn("prepare_tree_receipts", source)
        self.assertNotIn("build_replay_verdict", source)
        self.assertNotIn("subprocess.run", source)

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

    def test_prepare_binds_clean_revision_launcher_lock_environment_and_provenance(self) -> None:
        receipts = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
        }
        repository_commit = "c" * 40
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "final-request.json"
            with (
                mock.patch.object(TOOL, "prepare_tree_receipts", return_value=receipts),
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ) as resolver,
            ):
                request_sha256 = TOOL.prepare(output)
                written = json.loads(output.read_text(encoding="utf-8"))
                preparation_path = TOOL._default_preparation_receipt_path(output)
                preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
                verified = TOOL.verify_preparation_receipt(
                    preparation,
                    request=written,
                    experiment_id=TOOL.EXPERIMENT_ID,
                )
                self.assertEqual(verified, preparation)
                self.assertEqual(
                    preparation["preparation_repository_commit"], repository_commit
                )
                self.assertEqual(
                    preparation["reference_environment"], self.reference_environment
                )
                self.assertEqual(written["model"]["revision_tree_sha256"], "a" * 64)
                self.assertEqual(
                    written["model"]["tokenizer_revision_tree_sha256"], "b" * 64
                )
                self.assertEqual(request_sha256, TOOL.sha256_json(written))
                self.assertEqual(resolver.call_count, 2)
                resolver.assert_has_calls(
                    [
                        mock.call(None, require_checkout=True),
                        mock.call(repository_commit, require_checkout=True),
                    ]
                )
                template_calls = [
                    call
                    for call in self.template_auth.call_args_list
                    if call.args and call.args[0] == TOOL.TEMPLATE
                ]
                lock_calls = [
                    call
                    for call in self.template_auth.call_args_list
                    if call.args and call.args[0] == TOOL.REFERENCE_LOCK
                ]
                self.assertEqual(len(template_calls), 3)
                self.assertGreaterEqual(len(lock_calls), 2)
                self.assertGreaterEqual(self.launcher_auth.call_count, 2)
                self.assertEqual(self.environment_auth.call_count, 2)
                with self.assertRaises(CaptureContractError):
                    TOOL.prepare(output)

    def test_concurrent_request_publication_failure_never_deletes_shared_preparation_receipt(self) -> None:
        receipts = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
        }
        repository_commit = "c" * 40
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final-request.json"
            preparation_path = TOOL._default_preparation_receipt_path(output)
            real_write = TOOL._exclusive_write_json

            def concurrent_publish(path, value):
                if path == preparation_path:
                    return real_write(path, value)
                if path == output:
                    # Model the recovery process publishing the request first. The
                    # original publisher then observes the no-replace conflict.
                    real_write(path, value)
                    raise CaptureContractError(
                        f"refusing to overwrite existing artifact {path}"
                    )
                return real_write(path, value)

            with (
                mock.patch.object(TOOL, "prepare_tree_receipts", return_value=receipts),
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ),
                mock.patch.object(
                    TOOL,
                    "_exclusive_write_json",
                    side_effect=concurrent_publish,
                ),
            ):
                with self.assertRaisesRegex(CaptureContractError, "refusing to overwrite"):
                    TOOL.prepare(output)

            self.assertTrue(output.is_file())
            self.assertTrue(preparation_path.is_file())
            request = json.loads(output.read_text(encoding="utf-8"))
            receipt = json.loads(preparation_path.read_text(encoding="utf-8"))
            self.assertEqual(
                TOOL.verify_preparation_receipt(
                    receipt,
                    request=request,
                    experiment_id=TOOL.EXPERIMENT_ID,
                ),
                receipt,
            )

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

            def record(path: Path) -> None:
                synced.append(Path(path))

            with (
                mock.patch.object(TOOL, "_fsync_directory", side_effect=record),
                mock.patch(
                    "qsol_geo_reason.capture_publish._fsync_directory",
                    side_effect=record,
                ),
            ):
                TOOL._create_output_root_durable(output_root)
            self.assertTrue(output_root.is_dir())
            self.assertEqual(
                synced,
                [base, base / "level-a", base / "level-a" / "level-b"],
            )

    def test_intermediate_capture_cli_is_isolated_no_site_and_python_env_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text("{}", encoding="utf-8")
            output = root / "run-a"
            execution_receipt = root / "run-a-execution-receipt.json"
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = list(command)
                seen["env"] = dict(kwargs["env"])
                execution_receipt.write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="a" * 64 + "\n",
                    stderr="",
                )

            with (
                mock.patch.object(TOOL.subprocess, "run", side_effect=fake_run),
                mock.patch.dict(
                    TOOL.os.environ,
                    {
                        "PYTHONPATH": "/tmp/adversarial",
                        "PYTHONHOME": "/tmp/fake-home",
                    },
                    clear=False,
                ),
            ):
                receipt = TOOL._run_capture(
                    request,
                    output,
                    "f" * 40,
                    "EXEC-A",
                    execution_receipt,
                )
            self.assertEqual(receipt, "a" * 64)
            command = seen["command"]
            self.assertEqual(command[:5], [sys.executable, "-I", "-S", "-B", "-c"])
            self.assertIn("qsol_geo_reason.capture_cli", command[5])
            self.assertIn("--", command)
            environment = seen["env"]
            self.assertFalse(
                any(str(key).upper().startswith("PYTHON") for key in environment)
            )

    def test_observe_authenticates_template_launcher_lock_environment_and_archives_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(original), encoding="utf-8")
            preparation_path = self._write_preparation_receipt(request_path, original)
            output_root = directory / "observation"
            seen_paths: list[Path] = []
            seen_revisions: list[str] = []
            seen_execution_ids: list[str] = []
            seen_receipts: list[Path] = []
            repository_commit = "f" * 40

            def fake_run(path, output, revision, execution_id, execution_receipt_path):
                seen_paths.append(path)
                seen_revisions.append(revision)
                seen_execution_ids.append(execution_id)
                seen_receipts.append(execution_receipt_path)
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
                mock.patch.object(TOOL, "build_replay_verdict", return_value=verdict) as build,
                mock.patch.object(TOOL, "verify_replay_verdict", return_value=verdict) as verify,
            ):
                observed, status = TOOL.observe(
                    request_path,
                    output_root,
                    None,
                    preparation_path,
                )

            self.assertEqual(resolver.call_count, 3)
            resolver.assert_has_calls(
                [
                    mock.call(None, require_checkout=True),
                    mock.call(repository_commit, require_checkout=True),
                    mock.call(repository_commit, require_checkout=True),
                ]
            )
            template_calls = [
                call
                for call in self.template_auth.call_args_list
                if call.args and call.args[0] == TOOL.TEMPLATE
            ]
            lock_calls = [
                call
                for call in self.template_auth.call_args_list
                if call.args and call.args[0] == TOOL.REFERENCE_LOCK
            ]
            self.assertGreaterEqual(len(template_calls), 3)
            self.assertGreaterEqual(len(lock_calls), 3)
            self.assertGreaterEqual(self.launcher_auth.call_count, 3)
            self.assertEqual(self.environment_auth.call_count, 3)
            self.assertEqual(status, 0)
            self.assertEqual(observed, verdict)
            self.assertEqual(seen_paths[0], seen_paths[1])
            self.assertNotEqual(seen_paths[0], request_path)
            self.assertEqual(seen_revisions, [repository_commit, repository_commit])
            self.assertNotEqual(seen_execution_ids[0], seen_execution_ids[1])
            self.assertTrue(seen_execution_ids[0].endswith(":run-a"))
            self.assertTrue(seen_execution_ids[1].endswith(":run-b"))
            self.assertEqual(
                seen_receipts,
                [
                    output_root / "run-a-execution-receipt.json",
                    output_root / "run-b-execution-receipt.json",
                ],
            )
            archived_preparation_path = output_root / "preparation-receipt.json"
            self.assertEqual(
                build.call_args.kwargs["preparation_receipt_path"],
                archived_preparation_path,
            )
            self.assertEqual(
                verify.call_args.kwargs["preparation_receipt_path"],
                archived_preparation_path,
            )
            snapshot = json.loads(
                (output_root / "validated-request.json").read_text(encoding="utf-8")
            )
            archived_preparation = json.loads(
                archived_preparation_path.read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot, original)
            self.assertEqual(
                archived_preparation,
                json.loads(preparation_path.read_text(encoding="utf-8")),
            )
            self.assertEqual(
                archived_preparation["reference_environment"],
                self.reference_environment,
            )
            self.assertNotEqual(json.loads(request_path.read_text(encoding="utf-8")), original)

    def test_observe_rejects_tampered_preparation_before_creating_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            preparation_path = self._write_preparation_receipt(request_path, request)
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            preparation["request_sha256"] = "0" * 64
            preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
            output_root = directory / "observation"
            with self.assertRaisesRegex(
                CaptureContractError, "request_sha256 does not match"
            ):
                TOOL.observe(request_path, output_root, None, preparation_path)
            self.assertFalse(output_root.exists())

    def test_observe_rejects_environment_drift_before_creating_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            preparation_path = self._write_preparation_receipt(request_path, request)
            output_root = directory / "observation"
            with mock.patch.object(
                TOOL,
                "verify_current_reference_environment",
                side_effect=CaptureContractError(
                    "current reference environment does not match the preparation reference environment receipt"
                ),
            ):
                with self.assertRaisesRegex(CaptureContractError, "does not match the preparation"):
                    TOOL.observe(request_path, output_root, None, preparation_path)
            self.assertFalse(output_root.exists())

    def test_failed_observation_preserves_both_phase_revisions_and_planned_execution_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            preparation_path = self._write_preparation_receipt(
                request_path,
                request,
                repository_commit="c" * 40,
            )
            output_root = directory / "missing-parent" / "observation"
            repository_commit = "e" * 40
            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ),
                mock.patch.object(
                    TOOL,
                    "_run_capture",
                    side_effect=CaptureContractError("transient backend failure"),
                ) as run_capture,
            ):
                with self.assertRaisesRegex(CaptureContractError, "incomplete evidence preserved"):
                    TOOL.observe(
                        request_path,
                        output_root,
                        None,
                        preparation_path,
                    )
            self.assertEqual(run_capture.call_count, 1)
            self.assertEqual(run_capture.call_args.args[2], repository_commit)
            self.assertFalse(output_root.exists())
            failed = list((directory / "missing-parent").glob("observation.failed-*"))
            self.assertEqual(len(failed), 1)
            marker = json.loads(
                (failed[0] / "execution-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["repository_commit"], repository_commit)
            self.assertEqual(marker["preparation_repository_commit"], "c" * 40)
            self.assertRegex(marker["preparation_receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(marker["attempt_status"], "failed_before_replay_verdict")
            self.assertEqual(marker["completed_run_directories"], [])
            self.assertNotEqual(
                marker["planned_execution_ids"]["run-a"],
                marker["planned_execution_ids"]["run-b"],
            )
            self.assertTrue((failed[0] / "validated-request.json").is_file())
            self.assertTrue((failed[0] / "preparation-receipt.json").is_file())

    def test_failure_rename_reports_new_path_even_if_parent_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output_root = directory / "observation"
            output_root.mkdir()
            preparation = {
                "preparation_repository_commit": "c" * 40,
                "preparation_receipt_sha256": "d" * 64,
            }
            with (
                mock.patch.object(TOOL, "_exclusive_write_json"),
                mock.patch.object(
                    TOOL,
                    "_fsync_directory",
                    side_effect=OSError("metadata sync failed"),
                ),
            ):
                preserved = TOOL._preserve_failed_attempt(
                    output_root,
                    CaptureContractError("worker failed"),
                    "e" * 40,
                    {"run-a": "EXEC-A", "run-b": "EXEC-B"},
                    preparation,
                )
            self.assertNotEqual(preserved, output_root)
            self.assertFalse(output_root.exists())
            self.assertTrue(preserved.exists())
            self.assertTrue(preserved.name.startswith("observation.failed-"))

    def test_revision_resolution_failure_creates_no_attempt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            preparation_path = self._write_preparation_receipt(request_path, request)
            output_root = directory / "observation"
            with mock.patch.object(
                TOOL,
                "resolve_implementation_revision",
                side_effect=SourceIdentityError("checkout is dirty"),
            ):
                with self.assertRaisesRegex(CaptureContractError, "unable to bind production observation"):
                    TOOL.observe(request_path, output_root, None, preparation_path)
            self.assertFalse(output_root.exists())

    def test_prepare_missing_capture_dependency_uses_argparse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "request.json"
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value="f" * 40,
                ),
                mock.patch.object(
                    TOOL,
                    "prepare_tree_receipts",
                    side_effect=CaptureBackendUnavailable("install qsol-geo-reason[capture]"),
                ),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    TOOL.main(["prepare", "--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("install qsol-geo-reason[capture]", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(TOOL._default_preparation_receipt_path(output).exists())


if __name__ == "__main__":
    unittest.main()
