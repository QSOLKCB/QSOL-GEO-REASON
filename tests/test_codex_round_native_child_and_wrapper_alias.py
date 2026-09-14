from __future__ import annotations

import hashlib
import importlib.machinery
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import no_site_subprocess


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"
REVISION = "a" * 40
REFERENCE_PYTHON = "/tmp/qsol-geo-reason-capture-py311/bin/python"


def _documented_bootstrap_function() -> str:
    text = EXPERIMENT.read_text(encoding="utf-8")
    return text.split("qsol_first_observation() {", 1)[1].split("\n}\n```", 1)[0]


def _documented_bootstrap_literal() -> str:
    function = _documented_bootstrap_function()
    marker = f"' qsol_first_observation {REFERENCE_PYTHON} '"
    return function.split(marker, 1)[1].split("' \"$@\"", 1)[0]


def _documented_bootstrap_function_with(code: str, python_path: Path) -> str:
    function = _documented_bootstrap_function()
    literal = _documented_bootstrap_literal()
    function = function.replace(
        f" qsol_first_observation {REFERENCE_PYTHON} ",
        f" qsol_first_observation {shlex.quote(str(python_path))} ",
        1,
    )
    return function.replace(f"'{literal}'", shlex.quote(code), 1)


def _make_reference_python(tmp_path: Path) -> Path:
    python_path = tmp_path / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(Path(sys.executable).resolve())
    return python_path


class NativeChildAndWrapperAliasTests(unittest.TestCase):
    def test_authenticated_child_rejects_package_local_native_shadow_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)

            init_bytes = b"\n"
            helper_bytes = (
                b"def install_authenticated_source_manifest(revision, source_manifest):\n"
                b"    return None\n"
            )
            target_bytes = b"def main():\n    return 0\n"
            (package / "__init__.py").write_bytes(init_bytes)
            (package / "no_site_subprocess.py").write_bytes(helper_bytes)
            (package / "capture_cli.py").write_bytes(target_bytes)

            suffix = (
                importlib.machinery.EXTENSION_SUFFIXES[0]
                if importlib.machinery.EXTENSION_SUFFIXES
                else ".so"
            )
            native_shadow = package / f"capture_cli{suffix}"
            native_shadow.write_bytes(b"not-a-real-extension")

            manifest = {
                "__init__.py": hashlib.sha256(init_bytes).hexdigest(),
                "no_site_subprocess.py": hashlib.sha256(helper_bytes).hexdigest(),
                "capture_cli.py": hashlib.sha256(target_bytes).hexdigest(),
            }
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
                command = no_site_subprocess.isolated_package_command(
                    "qsol_geo_reason.capture_cli",
                    [],
                )

            bootstrap = command[5]
            self.assertIn("nativesfx=tuple", bootstrap)
            self.assertIn("native=sorted", bootstrap)
            self.assertIn("bytecode+native+pkgdirs", bootstrap)
            self.assertLess(
                bootstrap.index("native=sorted"),
                bootstrap.index("sys.path.insert(0,src)"),
            )

            completed = subprocess.run(
                command,
                cwd=root,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if not key.upper().startswith("PYTHON")
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "authenticated package subprocess rejects importable source shadows before src is trusted",
                completed.stderr,
            )
            self.assertIn(native_shadow.name, completed.stderr)
            self.assertNotIn("invalid ELF", completed.stderr)

    def test_documented_wrapper_name_bypasses_preexisting_alias_on_definition_and_call(self) -> None:
        document = EXPERIMENT.read_text(encoding="utf-8")
        self.assertIn("function qsol_first_observation() {", document)
        self.assertIn("\\qsol_first_observation prepare", document)
        self.assertIn("\\qsol_first_observation observe", document)
        self.assertIn("pre-existing alias named `qsol_first_observation`", document)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            python_path = _make_reference_python(tmp_path)
            function_body = _documented_bootstrap_function_with(
                "import sys; sys.exit(59)",
                python_path,
            )
            marker = tmp_path / "wrapper-alias-executed"
            alias_payload = (
                "printf intercepted > " + shlex.quote(str(marker)) + "; false #"
            )
            script = (
                "shopt -s expand_aliases\n"
                + "alias qsol_first_observation="
                + shlex.quote(alias_payload)
                + "\n"
                + "function qsol_first_observation() {"
                + function_body
                + "\n}\n"
                + "\\qsol_first_observation prepare\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc"],
                input=script,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 59, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "pre-existing wrapper alias executed before the authenticated boundary",
            )


if __name__ == "__main__":
    unittest.main()
