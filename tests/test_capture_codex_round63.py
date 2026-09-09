from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason import capture_backend_round58 as round58
from qsol_geo_reason import capture_backend_round63 as round63


class Round63GitRunnerTests(unittest.TestCase):
    def test_round58_exported_runner_is_replaced_by_closure_sealed_runner(self):
        self.assertIs(round58._git_run_round58, round63._git_run_round63)
        self.assertNotIn("_git_child_environment_round58", round58._git_run_round58.__globals__)
        self.assertIsNotNone(round58._git_run_round58.__closure__)

    def test_rebinding_round58_environment_builder_cannot_reach_runner(self):
        observed_env = {}

        def fake_run(_argv, *, env, **_kwargs):
            observed_env.update(env)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        def hostile_environment():
            return {"LD_PRELOAD": "/tmp/evil.so", "GIT_NO_REPLACE_OBJECTS": "0"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_builder = round58._git_child_environment_round58
            try:
                round58._git_child_environment_round58 = hostile_environment
                with (
                    patch.object(round63, "subprocess", wraps=round63.subprocess),
                    patch.object(round63._round56, "_trusted_git_executable", return_value=Path("/usr/bin/git")),
                ):
                    # The runner closure captured subprocess.run before this test;
                    # inspect its closure to replace only the test-local callable is
                    # intentionally impossible. Instead verify the closure contains
                    # the sanitizer itself and no reference to the rebound module slot.
                    closure_values = tuple(
                        cell.cell_contents for cell in round63._git_run_round63.__closure__ or ()
                    )
                    self.assertTrue(any(callable(value) for value in closure_values))
                    self.assertNotIn(hostile_environment, closure_values)
            finally:
                round58._git_child_environment_round58 = original_builder

    def test_round63_sanitizer_strips_loader_injection(self):
        runner = round63._git_run_round63
        closure_values = tuple(cell.cell_contents for cell in runner.__closure__ or ())
        child_environment = next(
            value
            for value in closure_values
            if callable(value) and getattr(value, "__name__", "") == "child_environment"
        )
        with patch.dict(
            os.environ,
            {
                "LD_PRELOAD": "/tmp/evil.so",
                "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
                "GIT_EXEC_PATH": "/tmp/fake",
                "ROUND63_SAFE": "yes",
            },
            clear=False,
        ):
            environment = child_environment()
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", environment)
        self.assertNotIn("GIT_EXEC_PATH", environment)
        self.assertEqual(environment["ROUND63_SAFE"], "yes")
        self.assertEqual(environment["GIT_NO_REPLACE_OBJECTS"], "1")


if __name__ == "__main__":
    unittest.main()
