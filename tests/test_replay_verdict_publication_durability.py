from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError
from reference_environment_fixture import reference_environment_receipt


class ReplayVerdictPublicationDurabilityTests(unittest.TestCase):
    def _materialized_request(self) -> dict:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return TOOL._assert_exact_experiment_request(request, require_receipts=True)

    def test_final_verdict_fsync_failure_is_recorded_as_post_publication(self) -> None:
        reference_environment = reference_environment_receipt()
        repository_commit = "e" * 40

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = self._materialized_request()
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            preparation_path = TOOL._default_preparation_receipt_path(request_path)
            preparation_receipt = TOOL.build_preparation_receipt(
                request=request,
                repository_commit="c" * 40,
                experiment_id=TOOL.EXPERIMENT_ID,
                reference_environment_receipt=reference_environment,
            )
            preparation_path.write_text(
                json.dumps(preparation_receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            output_root = directory / "observation"
            verdict = {"replay_outcome": "byte_identical"}

            real_write = TOOL._exclusive_write_json
            real_fsync_directory = TOOL._fsync_directory

            def write_with_final_verdict_sync_failure(path, value):
                path = Path(path)
                if path.name != "replay-verdict.json":
                    return real_write(path, value)

                def fail_only_verdict_parent_sync(sync_path):
                    sync_path = Path(sync_path)
                    if sync_path == output_root:
                        raise OSError("simulated replay-verdict parent fsync failure")
                    return real_fsync_directory(sync_path)

                with mock.patch.object(
                    TOOL,
                    "_fsync_directory",
                    side_effect=fail_only_verdict_parent_sync,
                ):
                    return real_write(path, value)

            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ),
                mock.patch.object(TOOL, "authenticate_tracked_file_against_revision"),
                mock.patch.object(
                    TOOL,
                    "authenticate_tracked_tool_against_revision",
                    return_value="a" * 64,
                ),
                mock.patch.object(
                    TOOL,
                    "verify_current_reference_environment",
                    side_effect=lambda expected=None: dict(
                        expected or reference_environment
                    ),
                ),
                mock.patch.object(TOOL, "_run_capture", return_value="a" * 64),
                mock.patch.object(TOOL, "build_replay_verdict", return_value=verdict),
                mock.patch.object(
                    TOOL,
                    "_exclusive_write_json",
                    side_effect=write_with_final_verdict_sync_failure,
                ),
            ):
                with self.assertRaisesRegex(
                    CaptureContractError,
                    "incomplete evidence preserved",
                ):
                    TOOL.observe(
                        request_path,
                        output_root,
                        None,
                        preparation_path,
                    )

            self.assertFalse(output_root.exists())
            failed = list(directory.glob("observation.failed-*"))
            self.assertEqual(len(failed), 1)
            self.assertTrue((failed[0] / "replay-verdict.json").is_file())
            marker = json.loads(
                (failed[0] / "execution-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["attempt_status"],
                "failed_after_replay_verdict_publication",
            )
            self.assertIs(marker["replay_verdict_published"], True)
            self.assertIn("parent fsync failure", marker["error_message"])


if __name__ == "__main__":
    unittest.main()
