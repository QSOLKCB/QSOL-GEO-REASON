from __future__ import annotations

import inspect
import json
import re
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_provenance import (
    _validate_required_determinism,
    _validate_snapshot_receipt,
)
from qsol_geo_reason import capture_publish
from qsol_geo_reason.capture_verify import (
    _validate_observed_dtype_map,
    verify_capture_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound5RegressionTests(unittest.TestCase):
    def test_observed_dtype_map_must_match_trajectory_records_exactly(self):
        recorded = {0: {"float32"}, 2: {"bfloat16", "float32"}}
        observed = {
            "observed_hidden_state_dtypes": {
                "0": ["float32"],
                "2": ["bfloat16", "float32"],
            }
        }
        _validate_observed_dtype_map(observed, recorded)
        observed["observed_hidden_state_dtypes"]["2"] = ["float16"]
        with self.assertRaisesRegex(CaptureContractError, "observed_hidden_state_dtypes"):
            _validate_observed_dtype_map(observed, recorded)
        self.assertIn(
            "_validate_observed_dtype_map(observed, recorded_dtypes)",
            inspect.getsource(verify_capture_bundle),
        )

    def test_snapshot_digest_must_be_lowercase_hex_sha256(self):
        hashes = {"weights.bin": "g" * 64}
        observed = {
            "model_snapshot_file_sha256": hashes,
            "model_snapshot_file_count": 1,
            "model_snapshot_receipt_sha256": sha256_json(hashes),
        }
        with self.assertRaisesRegex(CaptureContractError, "artifact hashes are malformed"):
            _validate_snapshot_receipt(observed, "model")

        uppercase = {"weights.bin": "A" * 64}
        observed = {
            "model_snapshot_file_sha256": uppercase,
            "model_snapshot_file_count": 1,
            "model_snapshot_receipt_sha256": sha256_json(uppercase),
        }
        with self.assertRaisesRegex(CaptureContractError, "artifact hashes are malformed"):
            _validate_snapshot_receipt(observed, "model")

    def test_required_determinism_requires_enabled_algorithms(self):
        required = {"determinism": {"mode": "required"}}
        with self.assertRaisesRegex(CaptureContractError, "required determinism"):
            _validate_required_determinism(
                {"deterministic_algorithms_enabled": False}, required
            )
        _validate_required_determinism(
            {"deterministic_algorithms_enabled": True}, required
        )
        _validate_required_determinism(
            {"deterministic_algorithms_enabled": False},
            {"determinism": {"mode": "best_effort"}},
        )

    def test_windows_directory_sync_path_does_not_open_directory(self):
        fake_path = Path("windows-directory-placeholder")
        with (
            mock.patch.object(capture_publish.os, "name", "nt"),
            mock.patch.object(
                capture_publish.os,
                "open",
                side_effect=AssertionError("Windows directory path called os.open"),
            ),
        ):
            capture_publish._fsync_directory(fake_path)

    def test_request_schema_rejects_whitespace_only_nonempty_fields(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        patterns = (
            schema["properties"]["run_id"]["pattern"],
            schema["$defs"]["backend"]["properties"]["device"]["pattern"],
            schema["$defs"]["step"]["properties"]["step_id"]["pattern"],
            schema["$defs"]["step"]["properties"]["text"]["pattern"],
        )
        for pattern in patterns:
            self.assertIsNone(re.search(pattern, "   \t"))
            self.assertIsNotNone(re.search(pattern, "value"))


if __name__ == "__main__":
    unittest.main()
