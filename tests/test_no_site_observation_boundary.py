from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_worker, no_site_subprocess


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


class NoSiteObservationBoundaryTests(unittest.TestCase):
    def _command_with_manifest(
        self,
        module: str,
        argv: list[str],
        *,
        root: Path = ROOT,
        manifest: dict[str, str] | None = None,
    ) -> list[str]:
        if manifest is None:
            manifest = {"__init__.py": "b" * 64}
        with (
            mock.patch.object(no_site_subprocess, "ROOT", root),
            mock.patch.object(
                no_site_subprocess,
                "_AUTHENTICATED_SOURCE_REVISION",
                REVISION,
            ),
            mock.patch.object(
                no_site_subprocess,
                "_AUTHENTICATED_SOURCE_MANIFEST",
                manifest,
            ),
        ):
            return no_site_subprocess.isolated_package_command(module, argv)

    def test_all_package_subprocesses_recheck_bound_source_before_import(self) -> None:
        for module in (
            "qsol_geo_reason.capture_cli",
            "qsol_geo_reason.capture_worker",
            "qsol_geo_reason.capture_hub_prepare_worker",
        ):
            with self.subTest(module=module):
                command = self._command_with_manifest(module, ["request.json"])
                self.assertEqual(
                    command[:5],
                    [sys.executable, "-I", "-S", "-B", "-c"],
                )
                bootstrap = command[5]
                self.assertIn(module, bootstrap)
                self.assertIn("source_manifest=json.loads(sys.argv[3])", bootstrap)
                self.assertIn("sourcebad=sorted", bootstrap)
                self.assertIn("install_authenticated_source_manifest", bootstrap)
                self.assertLess(
                    bootstrap.index("sourcebad=sorted"),
                    bootstrap.index("sys.path.insert(0,src)"),
                )
                self.assertLess(
                    bootstrap.index("sys.path.insert(0,src)"),
                    bootstrap.index(f"from {module} import main"),
                )
                self.assertIn(str((ROOT / "src").resolve()), command)
                self.assertIn(REVISION, command)
                self.assertIn("--", command)

    def test_self_restoring_child_payloads_are_rejected_before_import(self) -> None:
        modules = (
            "capture_cli",
            "capture_worker",
            "capture_hub_prepare_worker",
        )
        for target_name in modules:
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "src" / "qsol_geo_reason"
                package.mkdir(parents=True)
                init_path = package / "__init__.py"
                helper_path = package / "no_site_subprocess.py"
                target_path = package / f"{target_name}.py"
                init_bytes = b"\n"
                helper_bytes = (
                    b"def install_authenticated_source_manifest(revision, source_manifest):\n"
                    b"    return None\n"
                )
                clean_target = b"def main():\n    return 47\n"
                init_path.write_bytes(init_bytes)
                helper_path.write_bytes(helper_bytes)
                target_path.write_bytes(clean_target)

                manifest = {
                    "__init__.py": hashlib.sha256(init_bytes).hexdigest(),
                    "no_site_subprocess.py": hashlib.sha256(helper_bytes).hexdigest(),
                    f"{target_name}.py": hashlib.sha256(clean_target).hexdigest(),
                }
                marker = root / f"{target_name}-payload-executed"
                malicious = (
                    "from pathlib import Path\n"
                    f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
                    f"Path(__file__).write_bytes({clean_target!r})\n"
                    "def main():\n"
                    "    return 0\n"
                )
                target_path.write_text(malicious, encoding="utf-8")

                command = self._command_with_manifest(
                    f"qsol_geo_reason.{target_name}",
                    [],
                    root=root,
                    manifest=manifest,
                )
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if not key.upper().startswith("PYTHON")
                    },
                    input="{}\n" if target_name == "capture_hub_prepare_worker" else None,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("does not match bound revision", completed.stderr)
                self.assertFalse(
                    marker.exists(),
                    f"{target_name} top-level payload executed before authentication",
                )
                self.assertEqual(target_path.read_text(encoding="utf-8"), malicious)

    def test_launcher_installs_manifest_before_importing_orchestrator(self) -> None:
        launcher = (ROOT / "tools" / "run_first_production_observation.py").read_text(
            encoding="utf-8"
        )
        bootstrap = launcher.split("_ORCHESTRATOR_BOOTSTRAP = (", 1)[1].split(
            "\n)\n", 1
        )[0]
        self.assertIn("install_authenticated_source_manifest", bootstrap)
        self.assertLess(
            bootstrap.index("install_authenticated_source_manifest"),
            bootstrap.index("from qsol_geo_reason.first_production_observation import main"),
        )

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
        self.assertIn("not an executable evidence boundary", completed.stderr)
        self.assertNotIn("usage:", completed.stderr)


if __name__ == "__main__":
    unittest.main()
