from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason.provenance import SourceIdentityError
from qsol_geo_reason.tracked_artifact import authenticate_tracked_file_against_revision


class TrackedArtifactAuthenticationTests(unittest.TestCase):
    def _git(self, root: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        self._git(root, "init", "-q")
        self._git(root, "config", "user.name", "QSOL Test")
        self._git(root, "config", "user.email", "qsol@example.invalid")
        template = root / "experiments" / "frozen.json"
        decoy = root / "fixtures" / "decoy.json"
        template.parent.mkdir(parents=True)
        decoy.parent.mkdir(parents=True)
        template.write_bytes(b'{"frozen":true}\n')
        decoy.write_bytes(b'{"fixture":true}\n')
        self._git(root, "add", "experiments/frozen.json", "fixtures/decoy.json")
        self._git(root, "commit", "-qm", "freeze template")
        return template, decoy, self._git(root, "rev-parse", "HEAD")

    def _assert_index_hint_cannot_hide_change(self, hint: str) -> None:
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template, _decoy, revision = self._fixture(root)
            authenticated = authenticate_tracked_file_against_revision(
                template,
                revision,
                repo_root=root,
            )
            self.assertEqual(authenticated, "experiments/frozen.json")

            self._git(root, "update-index", hint, "experiments/frozen.json")
            template.write_bytes(b'{"frozen":false}\n')
            self.assertEqual(self._git(root, "status", "--porcelain"), "")

            with self.assertRaisesRegex(
                SourceIdentityError,
                "independently of index flags",
            ):
                authenticate_tracked_file_against_revision(
                    template,
                    revision,
                    repo_root=root,
                )

    def test_assume_unchanged_cannot_hide_modified_template_bytes(self) -> None:
        self._assert_index_hint_cannot_hide_change("--assume-unchanged")

    def test_skip_worktree_cannot_hide_modified_template_bytes(self) -> None:
        self._assert_index_hint_cannot_hide_change("--skip-worktree")

    @unittest.skipIf(os.name == "nt", "symlink creation requires platform-specific privileges")
    def test_skip_worktree_cannot_redirect_tracked_artifact_through_symlink(self) -> None:
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template, _decoy, revision = self._fixture(root)
            self._git(root, "update-index", "--skip-worktree", "experiments/frozen.json")

            template.unlink()
            template.symlink_to(Path("..") / "fixtures" / "decoy.json", target_is_directory=False)
            self.assertTrue(template.is_symlink())
            self.assertEqual(self._git(root, "status", "--porcelain"), "")

            with self.assertRaisesRegex(
                SourceIdentityError,
                "pathname must not be a symlink: experiments/frozen.json",
            ):
                authenticate_tracked_file_against_revision(
                    template,
                    revision,
                    repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
