from __future__ import annotations

import importlib.util
import json
import os
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

    def test_authenticated_git_blob_bootstrap_never_executes_modified_working_launcher(self) -> None:
        launcher = _load_launcher()
        git, _ = launcher._trusted_git_identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "tools"
            tools.mkdir()
            tracked_launcher = tools / "run_first_production_observation.py"
            clean_marker = root / "committed-launcher-executed"
            attacker_marker = root / "working-launcher-executed"
            clean = (
                "from pathlib import Path\n"
                f"Path({str(clean_marker)!r}).write_text('committed', encoding='utf-8')\n"
            )
            malicious = (
                "from pathlib import Path\n"
                f"Path({str(attacker_marker)!r}).write_text('attacker', encoding='utf-8')\n"
                f"Path(__file__).write_text({clean!r}, encoding='utf-8')\n"
            )
            tracked_launcher.write_text(clean, encoding="utf-8")
            _git(git, root, "init", "-q")
            _git(git, root, "config", "user.email", "qsol-test@example.invalid")
            _git(git, root, "config", "user.name", "QSOL Test")
            _git(git, root, "add", "tools/run_first_production_observation.py")
            _git(git, root, "commit", "-qm", "launcher fixture")
            _git(
                git,
                root,
                "update-index",
                "--skip-worktree",
                "tools/run_first_production_observation.py",
            )
            tracked_launcher.write_text(malicious, encoding="utf-8")
            self.assertEqual(_git(git, root, "status", "--porcelain"), "")

            bootstrap = (
                "import os,pathlib,subprocess,sys;"
                "root=pathlib.Path(sys.argv[1]).resolve();"
                "git=pathlib.Path(sys.argv[2]).resolve(strict=True);"
                "path=root/'tools'/'run_first_production_observation.py';"
                "env={'PATH':'/usr/bin:/bin','LC_ALL':'C','GIT_NO_REPLACE_OBJECTS':'1',"
                "'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull,'GIT_OPTIONAL_LOCKS':'0'};"
                "src=subprocess.run([str(git),'-C',str(root),'cat-file','blob',"
                "'HEAD:tools/run_first_production_observation.py'],env=env,check=True,"
                "capture_output=True).stdout;"
                "ns={'__name__':'__main__','__file__':str(path),'__package__':None};"
                "sys.argv=[str(path)];exec(compile(src,str(path),'exec'),ns,ns)"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", bootstrap, str(root), str(git)],
                cwd=root,
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(clean_marker.is_file())
            self.assertFalse(attacker_marker.exists())
            self.assertEqual(tracked_launcher.read_text(encoding="utf-8"), malicious)

    def test_committed_launcher_rejects_symlinked_working_path_before_root_resolution(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trusted_root = workspace / "trusted"
            trusted_tools = trusted_root / "tools"
            trusted_tools.mkdir(parents=True)
            attacker_root = workspace / "attacker"
            attacker_tools = attacker_root / "tools"
            attacker_tools.mkdir(parents=True)
            attacker_launcher = attacker_tools / "run_first_production_observation.py"
            attacker_launcher.write_text("# attacker checkout\n", encoding="utf-8")
            working_launcher = trusted_tools / "run_first_production_observation.py"
            working_launcher.symlink_to(attacker_launcher)

            bootstrap = (
                "import sys;"
                "src=sys.argv[1];path=sys.argv[2];"
                "ns={'__name__':'qsol_symlink_test','__file__':path,'__package__':None};"
                "exec(compile(src,path,'exec'),ns,ns)"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", bootstrap, source, str(working_launcher)],
                cwd=trusted_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("rejects symlinked launcher file", completed.stderr)

    def test_repository_root_boundary_rejects_symlinked_tools_parent(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trusted_root = workspace / "trusted"
            trusted_root.mkdir()
            attacker_tools = workspace / "attacker-tools"
            attacker_tools.mkdir()
            (attacker_tools / "run_first_production_observation.py").write_text(
                "# attacker launcher\n",
                encoding="utf-8",
            )
            (trusted_root / "tools").symlink_to(attacker_tools, target_is_directory=True)
            candidate = trusted_root / "tools" / "run_first_production_observation.py"
            with self.assertRaisesRegex(RuntimeError, "symlinked launcher tools directory"):
                launcher._repository_root_from_launcher_path(candidate)


if __name__ == "__main__":
    unittest.main()
