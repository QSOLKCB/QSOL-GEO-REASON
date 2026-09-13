from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "run_first_production_observation.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "qsol_test_preimport_source_launcher",
        LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load production launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(git: Path, root: Path, *args: str) -> str:
    completed = subprocess.run(
        [str(git), "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class LauncherPreimportSourceAuthenticationTests(unittest.TestCase):
    def test_skip_worktree_self_restoring_tracked_payload_never_executes(self) -> None:
        launcher = _load_launcher()
        git, git_digest = launcher._trusted_git_identity()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("\n", encoding="utf-8")
            clean_canonical = "from pathlib import Path\nVALUE = 1\nCLEAN = True\n"
            canonical = package / "canonical.py"
            canonical.write_text(clean_canonical, encoding="utf-8")
            (package / "first_production_observation.py").write_text(
                "from . import canonical\n\n"
                "def main():\n"
                "    return 0\n",
                encoding="utf-8",
            )

            _git(git, root, "init", "-q")
            _git(git, root, "config", "user.email", "qsol-test@example.invalid")
            _git(git, root, "config", "user.name", "QSOL Test")
            _git(git, root, "add", "src/qsol_geo_reason")
            _git(git, root, "commit", "-qm", "fixture")
            revision = _git(git, root, "rev-parse", "HEAD")

            with mock.patch.object(launcher, "ROOT", root):
                bound_revision, source_manifest = launcher._authenticate_tracked_package_source(
                    git_path=git,
                    git_digest=git_digest,
                )
            self.assertEqual(bound_revision, revision)
            self.assertIn("canonical.py", source_manifest)

            marker = root / "payload-executed"
            malicious = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
                f"Path(__file__).write_text({clean_canonical!r}, encoding='utf-8')\n"
            )
            self.assertEqual(len(malicious.splitlines()), len(clean_canonical.splitlines()))
            canonical.write_text(malicious, encoding="utf-8")
            _git(
                git,
                root,
                "update-index",
                "--skip-worktree",
                "src/qsol_geo_reason/canonical.py",
            )
            self.assertEqual(_git(git, root, "status", "--porcelain"), "")

            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "tracked package source that does not match bound revision",
                ):
                    launcher._authenticate_tracked_package_source(
                        git_path=git,
                        git_digest=git_digest,
                        revision=revision,
                    )
            self.assertFalse(marker.exists())

            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    launcher._ORCHESTRATOR_BOOTSTRAP,
                    str(root / "src"),
                    revision,
                    json.dumps(source_manifest, sort_keys=True, separators=(",", ":")),
                    "--",
                ],
                cwd=root,
                env=launcher._sanitized_orchestrator_environment(),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "tracked package source that does not match bound revision",
                completed.stderr,
            )
            self.assertFalse(marker.exists())
            self.assertEqual(canonical.read_text(encoding="utf-8"), malicious)


if __name__ == "__main__":
    unittest.main()
