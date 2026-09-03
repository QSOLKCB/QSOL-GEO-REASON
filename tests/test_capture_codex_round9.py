from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_execute import execute_capture
from qsol_geo_reason.capture_publish import _rename_directory_noreplace
from qsol_geo_reason.capture_validation import validate_capture_request

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class CaptureRound9RegressionTests(unittest.TestCase):
    def test_only_instrumented_device_forms_are_accepted(self):
        for device in ("cpu", "mps", "cuda:0", "cuda:17"):
            request = fixture_request()
            request["backend"]["device"] = device
            with self.subTest(device=device):
                self.assertEqual(validate_capture_request(request)["backend"]["device"], device)

        for device in ("meta", "xpu:0", "cpu ", " cuda:0", "cuda", "cuda:-1"):
            request = fixture_request()
            request["backend"]["device"] = device
            with self.subTest(device=device):
                with self.assertRaisesRegex(CaptureContractError, "backend.device"):
                    validate_capture_request(request)

    def test_device_schema_whitelist_matches_runtime(self):
        request_schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
        )
        manifest_schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        request_device = request_schema["$defs"]["backend"]["properties"]["device"]
        manifest_device = manifest_schema["$defs"]["device"]
        for definition in (request_device, manifest_device):
            branches = definition["oneOf"]
            self.assertIn({"enum": ["cpu", "mps"]}, branches)
            self.assertIn({"type": "string", "pattern": "^cuda:[0-9]+$"}, branches)

    def test_observed_dtype_state_is_cleared_before_capture_steps(self):
        source = inspect.getsource(execute_capture)
        clear = source.index("backend._observed_hidden_state_dtypes.clear()")
        capture = source.index("steps, prefix_ids = _capture_steps")
        self.assertLess(clear, capture)

    def test_atomic_noreplace_publication_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "staging"
            destination = root / "bundle"
            source.mkdir()
            destination.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            (destination / "existing.txt").write_text("existing", encoding="utf-8")

            with self.assertRaises(CaptureContractError):
                _rename_directory_noreplace(source, destination)

            self.assertTrue(source.is_dir())
            self.assertEqual(
                (destination / "existing.txt").read_text(encoding="utf-8"),
                "existing",
            )
            self.assertFalse((destination / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
