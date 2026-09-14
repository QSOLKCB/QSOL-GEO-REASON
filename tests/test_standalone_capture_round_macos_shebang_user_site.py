from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason import no_site_subprocess


ROOT = Path(__file__).resolve().parents[1]
POSIX_WRAPPER = ROOT / "scripts" / "qsol-geo-capture"
WINDOWS_STAGE0 = ROOT / "scripts" / "qsol-geo-capture-windows.py"
PAYLOAD = ROOT / "scripts" / "qsol-geo-capture-python"
WORKFLOW = ROOT / ".github" / "workflows" / "phase1-reference.yml"


class StandaloneCaptureMacShebangUserSiteTests(unittest.TestCase):
    def test_posix_wrapper_is_macos_portable_without_gnu_readlink_f(self) -> None:
        source = POSIX_WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("readlink -f", source)
        self.assertNotIn("[[ -v", source)
        self.assertIn("/Library/Frameworks/Python.framework/Versions/*/bin/python*", source)
        self.assertIn("/opt/homebrew/Cellar/python*", source)
        self.assertIn("/usr/local/Cellar/python*", source)
        self.assertIn("/Users/runner/hostedtoolcache/Python/*/bin/python*", source)
        self.assertIn("qsol_link_hops", source)

    def test_platform_stage0s_carry_validated_package_paths_into_authenticated_payload(self) -> None:
        posix = POSIX_WRAPPER.read_text(encoding="utf-8")
        windows = WINDOWS_STAGE0.read_text(encoding="utf-8")
        payload = PAYLOAD.read_text(encoding="utf-8")
        for source in (posix, windows):
            self.assertIn('"_QSOL_STANDALONE_PACKAGE_PATHS"', source)
        self.assertIn('_STAGE0_PACKAGE_PATHS_NAME = "_QSOL_STANDALONE_PACKAGE_PATHS"', payload)
        self.assertIn("candidates: list[Path] = list(_stage0_package_paths())", payload)
        self.assertIn('"[sys.path.append(p) for p in paths if p not in sys.path];"', payload)

        nested = Path(no_site_subprocess.__file__).read_text(encoding="utf-8")
        self.assertIn("for raw in sys.path", nested)
        self.assertIn('candidate.name.lower() == "site-packages"', nested)
        self.assertLess(
            payload.index('"[sys.path.append(p) for p in paths if p not in sys.path];"'),
            payload.index('"from qsol_geo_reason.capture_cli import main;"'),
        )

    def test_user_site_only_dependency_is_importable_in_isolated_capture_child(self) -> None:
        namespace = runpy.run_path(str(PAYLOAD), run_name="qsol_standalone_capture_test")
        bootstrap = namespace["_authenticated_capture_bootstrap"]()
        with tempfile.TemporaryDirectory(prefix="qsol-user-site-child-") as tmp:
            root = Path(tmp)
            src = root / "src"
            package = src / "qsol_geo_reason"
            user_site = root / "user" / "site-packages"
            package.mkdir(parents=True)
            user_site.mkdir(parents=True)

            files = {
                "__init__.py": "",
                "no_site_subprocess.py": (
                    "def install_authenticated_source_manifest(revision, source_manifest):\n"
                    "    return None\n"
                ),
                "capture_cli.py": (
                    "def main():\n"
                    "    import qsol_user_only_dependency\n"
                    "    print(qsol_user_only_dependency.VALUE)\n"
                    "    return 0\n"
                ),
            }
            manifest: dict[str, str] = {}
            for name, content in files.items():
                path = package / name
                path.write_text(content, encoding="utf-8")
                manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (user_site / "qsol_user_only_dependency.py").write_text(
                "VALUE = 'USER-SITE-OK'\n", encoding="utf-8"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    bootstrap,
                    str(src),
                    "a" * 40,
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    str(user_site),
                    "--",
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("USER-SITE-OK", completed.stdout)

    def test_posix_wrapper_authentication_includes_complete_shebang_bytes(self) -> None:
        source = PAYLOAD.read_text(encoding="utf-8")
        auth = source.split("def _authenticate_installed_bootstrap", 1)[1].split(
            "def _authenticate_checkout", 1
        )[0]
        self.assertIn("if installed != committed:", auth)
        self.assertNotIn("committed_parts", auth)
        self.assertNotIn("installed_parts", auth)
        self.assertNotIn("startswith(b\"#!\")", auth)

        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("macos-standalone-capture:", workflow)
        self.assertIn('/bin/bash -p "$installed" --help', workflow)
        self.assertIn("#!/tmp/fake", workflow)
        self.assertIn(
            "installed qsol-geo-capture wrapper does not match the bound Git revision",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
