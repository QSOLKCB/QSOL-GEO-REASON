from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_reference_environment as REFERENCE
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]


class Phase2ARoundFinalReviewTests(unittest.TestCase):
    def test_direct_orchestration_facade_execution_fails_closed(self) -> None:
        bootstrap = (
            "import runpy,sys;"
            f"sys.path.insert(0,{str((ROOT / 'src').resolve())!r});"
            "runpy.run_module('qsol_geo_reason.first_production_observation',run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", bootstrap],
            cwd=ROOT,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("PYTHON")
            },
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "direct module execution is not an authenticated production boundary",
            completed.stderr,
        )
        self.assertIn(
            "python -I -S -B tools/run_first_production_observation.py",
            completed.stderr,
        )
        self.assertNotIn("usage:", completed.stderr)

    def test_identical_duplicate_distribution_metadata_is_rejected(self) -> None:
        first = types.SimpleNamespace(
            metadata={"Name": "requests"},
            version="2.34.2",
        )
        second = types.SimpleNamespace(
            metadata={"Name": "requests"},
            version="2.34.2",
        )
        with mock.patch.object(
            REFERENCE.importlib.metadata,
            "distributions",
            return_value=[first, second],
        ):
            with self.assertRaisesRegex(
                CaptureContractError,
                "multiple installed distributions normalize to the same name",
            ):
                REFERENCE._installed_runtime_versions()


if __name__ == "__main__":
    unittest.main()
