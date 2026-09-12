from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_worker
from qsol_geo_reason.no_site_subprocess import isolated_package_command


ROOT = Path(__file__).resolve().parents[1]


class NoSiteObservationBoundaryTests(unittest.TestCase):
    def test_both_observation_subprocess_hops_use_isolated_no_site_bootstrap(self) -> None:
        for module in (
            "qsol_geo_reason.capture_cli",
            "qsol_geo_reason.capture_worker",
        ):
            command = isolated_package_command(module, ["request.json"])
            self.assertEqual(command[:5], [sys.executable, "-I", "-S", "-B", "-c"])
            self.assertIn(module, command[5])
            self.assertIn(str((ROOT / "src").resolve()), command)
            self.assertIn("--", command)

    def test_worker_rejects_isolated_process_when_site_is_still_enabled(self) -> None:
        flags = types.SimpleNamespace(
            isolated=1,
            no_site=0,
            no_user_site=1,
            ignore_environment=1,
        )
        with (
            mock.patch.dict(
                capture_worker.os.environ,
                {"QSOL_GEO_CAPTURE_FRESH_WORKER": "1"},
                clear=True,
            ),
            mock.patch.object(capture_worker.sys, "flags", flags),
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated no-site mode"):
                capture_worker._assert_fresh_worker_boundary()

    def test_direct_core_module_execution_fails_closed(self) -> None:
        bootstrap = (
            "import runpy,sys;"
            f"sys.path.insert(0,{str((ROOT / 'src').resolve())!r});"
            "runpy.run_module('qsol_geo_reason.first_production_observation_core',run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", bootstrap],
            cwd=ROOT,
            env={key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not an executable evidence boundary", completed.stderr)
        self.assertNotIn("usage:", completed.stderr)


if __name__ == "__main__":
    unittest.main()
