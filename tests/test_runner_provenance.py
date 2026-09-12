from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import runner_provenance
from qsol_geo_reason.provenance import SourceIdentityError


class RunnerProvenanceTests(unittest.TestCase):
    def test_direct_blob_authentication_ignores_assume_unchanged_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"],
                check=True,
            )
            tools = root / "tools"
            tools.mkdir()
            runner = tools / "run.py"
            runner.write_text("print('trusted')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tools/run.py"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "freeze runner"],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            with mock.patch.object(
                runner_provenance,
                "source_repo_root",
                return_value=root,
            ):
                committed_blob = runner_provenance.authenticate_tracked_tool_against_revision(
                    runner, revision
                )
                self.assertRegex(committed_blob, r"^[0-9a-f]{40}$")

                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "update-index",
                        "--assume-unchanged",
                        "tools/run.py",
                    ],
                    check=True,
                )
                runner.write_text("print('tampered')\n", encoding="utf-8")
                status = subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain=v1"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertEqual(status, "")

                with self.assertRaisesRegex(
                    SourceIdentityError, "runner bytes do not match"
                ):
                    runner_provenance.authenticate_tracked_tool_against_revision(
                        runner, revision
                    )


if __name__ == "__main__":
    unittest.main()
