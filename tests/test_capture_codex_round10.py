from __future__ import annotations

import inspect
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound10RegressionTests(unittest.TestCase):
    def test_mps_prefer_metal_override_is_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._device_type = "mps"
        with patch.dict(os.environ, {"PYTORCH_MPS_PREFER_METAL": "1"}, clear=False):
            with self.assertRaisesRegex(CaptureContractError, "MPS_PREFER_METAL"):
                backend._assert_mps_execution_policy()

    def test_mps_policy_guard_runs_at_construction_and_each_forward(self):
        init_source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        forward_source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        self.assertIn("self._assert_mps_execution_policy()", init_source)
        self.assertLess(
            forward_source.index("self._assert_mps_execution_policy()"),
            forward_source.index("self._base_model("),
        )

    def test_trajectory_schema_rejects_whitespace_only_run_and_step_ids(self):
        schema = json.loads(
            (ROOT / "schemas" / "captured-trajectory.schema.json").read_text(
                encoding="utf-8"
            )
        )
        run_pattern = schema["properties"]["run_id"]["pattern"]
        step_pattern = schema["$defs"]["step"]["properties"]["step_id"]["pattern"]
        for pattern in (run_pattern, step_pattern):
            self.assertIsNone(re.search(pattern, "   \t"))
            self.assertIsNotNone(re.search(pattern, "identity-001"))


if __name__ == "__main__":
    unittest.main()
