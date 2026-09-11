from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason import capture_replay


ROOT = Path(__file__).resolve().parents[1]


class CaptureReplayVerdictTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[dict, Path, Path, Path]:
        request = {"frozen": "request", "receipts": ["a", "b"]}
        validated_request = root / "validated-request.json"
        validated_request.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        run_a = root / "run-a"
        run_b = root / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        for directory in (run_a, run_b):
            for name in capture_replay.REPLAY_BUNDLE_FILES:
                (directory / name).write_text(
                    json.dumps({"artifact": name}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return request, validated_request, run_a, run_b

    @staticmethod
    def _fake_bundle_loader(directory: Path, expected_request: dict):
        manifest = {
            "manifest_sha256": "a" * 64,
            "repository_commit": "b" * 40,
        }
        trajectory = {
            "evidence_class": "OBSERVATION",
            "trajectory_sha256": "c" * 64,
        }
        return dict(expected_request), manifest, trajectory

    def _build_verdict(self, root: Path):
        request, snapshot, run_a, run_b = self._workspace(root)
        verdict = capture_replay.build_replay_verdict(
            request=request,
            validated_request_path=snapshot,
            run_a_dir=run_a,
            run_b_dir=run_b,
            run_a_manifest_receipt="a" * 64,
            run_b_manifest_receipt="a" * 64,
            experiment_id="EXP-TEST-001",
        )
        return request, snapshot, run_a, run_b, verdict

    def test_schema_is_closed_and_declares_semantic_verifier_fields(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "replay-verdict.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("validated_request_artifact_sha256", schema["required"])
        self.assertIn("bundle_file_byte_equality", schema["required"])
        self.assertEqual(
            schema["properties"]["replay_outcome"]["enum"],
            ["byte_identical", "diverged"],
        )

    def test_build_and_verify_byte_identical_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                capture_replay,
                "_load_verified_observation_bundle",
                side_effect=self._fake_bundle_loader,
            ):
                request, snapshot, run_a, run_b, verdict = self._build_verdict(
                    Path(tmp)
                )
                verified = capture_replay.verify_replay_verdict(
                    verdict,
                    request=request,
                    validated_request_path=snapshot,
                    run_a_dir=run_a,
                    run_b_dir=run_b,
                    run_a_manifest_receipt="a" * 64,
                    run_b_manifest_receipt="a" * 64,
                    experiment_id="EXP-TEST-001",
                )
            self.assertEqual(verified, verdict)
            self.assertEqual(verdict["replay_outcome"], "byte_identical")
            self.assertTrue(all(verdict["bundle_file_byte_equality"].values()))

    def test_verifier_rejects_rehashed_or_edited_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                capture_replay,
                "_load_verified_observation_bundle",
                side_effect=self._fake_bundle_loader,
            ):
                request, snapshot, run_a, run_b, verdict = self._build_verdict(
                    Path(tmp)
                )
                tampered = json.loads(json.dumps(verdict))
                tampered["request_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    CaptureContractError, "does not match the verified request"
                ):
                    capture_replay.verify_replay_verdict(
                        tampered,
                        request=request,
                        validated_request_path=snapshot,
                        run_a_dir=run_a,
                        run_b_dir=run_b,
                        run_a_manifest_receipt="a" * 64,
                        run_b_manifest_receipt="a" * 64,
                        experiment_id="EXP-TEST-001",
                    )

    def test_verifier_rejects_malformed_scalar_types_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                capture_replay,
                "_load_verified_observation_bundle",
                side_effect=self._fake_bundle_loader,
            ):
                request, snapshot, run_a, run_b, verdict = self._build_verdict(
                    Path(tmp)
                )

                malformed_experiment = json.loads(json.dumps(verdict))
                malformed_experiment["experiment_id"] = 7
                with self.assertRaisesRegex(
                    CaptureContractError, "experiment_id.*non-empty string"
                ):
                    capture_replay.verify_replay_verdict(
                        malformed_experiment,
                        request=request,
                        validated_request_path=snapshot,
                        run_a_dir=run_a,
                        run_b_dir=run_b,
                        run_a_manifest_receipt="a" * 64,
                        run_b_manifest_receipt="a" * 64,
                        experiment_id=7,  # type: ignore[arg-type]
                    )

                malformed_outcome = json.loads(json.dumps(verdict))
                malformed_outcome["replay_outcome"] = ["byte_identical"]
                with self.assertRaisesRegex(
                    CaptureContractError, "replay_outcome"
                ):
                    capture_replay.verify_replay_verdict(
                        malformed_outcome,
                        request=request,
                        validated_request_path=snapshot,
                        run_a_dir=run_a,
                        run_b_dir=run_b,
                        run_a_manifest_receipt="a" * 64,
                        run_b_manifest_receipt="a" * 64,
                        experiment_id="EXP-TEST-001",
                    )

    def test_bundle_byte_divergence_is_recorded_not_tuned_away(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, snapshot, run_a, run_b = self._workspace(root)
            (run_b / "captured-trajectory.json").write_text(
                '{"artifact":"captured-trajectory.json","changed":true}\n',
                encoding="utf-8",
            )
            with mock.patch.object(
                capture_replay,
                "_load_verified_observation_bundle",
                side_effect=self._fake_bundle_loader,
            ):
                verdict = capture_replay.build_replay_verdict(
                    request=request,
                    validated_request_path=snapshot,
                    run_a_dir=run_a,
                    run_b_dir=run_b,
                    run_a_manifest_receipt="a" * 64,
                    run_b_manifest_receipt="a" * 64,
                    experiment_id="EXP-TEST-001",
                )
            self.assertEqual(verdict["replay_outcome"], "diverged")
            self.assertFalse(
                verdict["bundle_file_byte_equality"]["captured-trajectory.json"]
            )


if __name__ == "__main__":
    unittest.main()
