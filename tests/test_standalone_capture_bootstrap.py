from __future__ import annotations

import hashlib
import importlib.machinery
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "qsol-geo-capture"
WINDOWS_WRAPPER = ROOT / "scripts" / "qsol-geo-capture.cmd"
WINDOWS_STAGE0 = ROOT / "scripts" / "qsol-geo-capture-windows.py"
PYTHON_PAYLOAD = ROOT / "scripts" / "qsol-geo-capture-python"
PYPROJECT = ROOT / "pyproject.toml"
PROTOCOL = ROOT / "protocols" / "GEO-CAP-001.md"


class StandaloneCaptureBootstrapTests(unittest.TestCase):
    def test_packaging_installs_platform_native_capture_wrappers(self) -> None:
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        self.assertIn('"scripts/qsol-geo-capture"', pyproject)
        self.assertIn('"scripts/qsol-geo-capture.cmd"', pyproject)
        self.assertIn('"scripts/qsol-geo-capture-windows.py"', pyproject)
        self.assertNotIn(
            'qsol-geo-capture = "qsol_geo_reason.capture_cli:main"',
            pyproject,
        )

    def test_installed_wrapper_uses_installer_bound_interpreter_and_isolated_stage0(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertEqual(source.splitlines()[0], "#!/bin/bash -p")
        self.assertIn('qsol_interpreter_anchor="$qsol_bindir/qsol-geo-sim"', source)
        self.assertIn('IFS= read -r qsol_shebang < "$qsol_interpreter_anchor"', source)
        self.assertIn("qsol_python=${qsol_shebang#\\#!}", source)
        self.assertIn('exec "$qsol_python" -I -S -B - "$qsol_script" "$@"', source)
        self.assertNotIn('qsol_python="$qsol_bindir/python"', source)
        self.assertNotIn('qsol_python="$qsol_bindir/python3"', source)
        self.assertIn("not sys.flags.isolated", source)
        self.assertIn("not sys.flags.no_site", source)
        self.assertIn("not sys.dont_write_bytecode", source)
        self.assertIn('commit + ":scripts/qsol-geo-capture-python"', source)
        self.assertIn("editable_checkout_root", source)
        self.assertIn("direct_url.json", source)
        self.assertIn("_QSOL_STANDALONE_AUTHENTICATED_ROOT", source)
        self.assertNotIn("root = pathlib.Path.cwd()", source)
        self.assertNotIn("#!/usr/bin/env python", source)
        self.assertNotIn("import qsol_geo_reason", source)

        interpreter_anchor = source.index('qsol_interpreter_anchor="$qsol_bindir/qsol-geo-sim"')
        python_exec = source.index('exec "$qsol_python" -I -S -B')
        stage0_flag_check = source.index("if not sys.flags.isolated")
        discover_root = source.index("root = editable_checkout_root()")
        read_payload = source.index('commit + ":scripts/qsol-geo-capture-python"')
        invoke_payload = source.index('namespace["main"]()')
        self.assertLess(interpreter_anchor, python_exec)
        self.assertLess(python_exec, stage0_flag_check)
        self.assertLess(stage0_flag_check, discover_root)
        self.assertLess(discover_root, read_payload)
        self.assertLess(read_payload, invoke_payload)

    def test_stage0_preserves_lexical_virtualenv_and_derives_user_site_from_wrapper(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("executable = pathlib.Path(sys.executable)", source)
        self.assertNotIn("pathlib.Path(sys.executable).resolve", source)
        self.assertIn("venv_root = executable.parent.parent", source)
        self.assertIn("install_prefix = wrapper.parent.parent", source)
        self.assertIn('install_prefix / "lib" / version / "site-packages"', source)
        self.assertIn('install_prefix / "lib64" / version / "site-packages"', source)

    def test_windows_wrapper_starts_isolated_stage0_and_is_packaged(self) -> None:
        wrapper = WINDOWS_WRAPPER.read_text(encoding="utf-8")
        stage0 = WINDOWS_STAGE0.read_text(encoding="utf-8")
        self.assertIn("qsol-geo-capture-windows.py", wrapper)
        self.assertIn("-I -S -B", wrapper)
        self.assertIn("python.exe", wrapper)
        self.assertIn("..\\python.exe", wrapper)
        self.assertIn("not sys.flags.isolated", stage0)
        self.assertIn("not sys.flags.no_site", stage0)
        self.assertIn("not sys.dont_write_bytecode", stage0)
        self.assertIn("direct_url.json", stage0)
        self.assertIn('"scripts/qsol-geo-capture.cmd"', stage0)
        self.assertIn('"scripts/qsol-geo-capture-windows.py"', stage0)
        self.assertIn('commit + ":scripts/qsol-geo-capture-python"', stage0)
        self.assertIn("_QSOL_STANDALONE_AUTHENTICATED_ROOT", stage0)
        self.assertNotIn("Path.cwd()", stage0)
        self.assertNotIn("import qsol_geo_reason", stage0)

    def test_payload_authenticates_stage0_root_and_wrapper_before_any_package_import(self) -> None:
        source = PYTHON_PAYLOAD.read_text(encoding="utf-8")
        self.assertIn('_STAGE0_ROOT_NAME = "_QSOL_STANDALONE_AUTHENTICATED_ROOT"', source)
        self.assertIn(
            '_STAGE0_BOOTSTRAP_PATH_NAME = "_QSOL_STANDALONE_BOOTSTRAP_GIT_PATH"',
            source,
        )
        self.assertIn('"scripts/qsol-geo-capture.cmd"', source)
        self.assertIn('"HEAD^{commit}"', source)
        self.assertIn('"ls-tree"', source)
        self.assertIn('_PACKAGE_GIT_ROOT = "src/qsol_geo_reason"', source)
        self.assertIn("_authenticate_installed_bootstrap", source)
        self.assertIn("_git_blob_sha1(raw) != object_id", source)
        self.assertIn('"-I",', source)
        self.assertIn('"-S",', source)
        self.assertIn('"-B",', source)

        authenticate_source = source.split(
            "def _authenticate_checkout()", 1
        )[1].split("def _literal_site_package_paths", 1)[0]
        self.assertIn("root = _stage0_authenticated_root()", authenticate_source)
        self.assertNotIn("Path.cwd()", authenticate_source)
        self.assertIn('"--show-toplevel"', authenticate_source)

        native_scan = source.index('"native=sorted(')
        verifier = source.index('"sourcebad=sorted(')
        expose_source = source.index('"sys.path.insert(0,src);"')
        seed_child = source.index(
            '"_qsol_ns.install_authenticated_source_manifest(revision,source_manifest);"'
        )
        import_cli = source.index('"from qsol_geo_reason.capture_cli import main;"')
        self.assertLess(native_scan, verifier)
        self.assertLess(verifier, expose_source)
        self.assertLess(expose_source, seed_child)
        self.assertLess(seed_child, import_cli)

        before_child_bootstrap = source[: source.index("def _authenticated_capture_bootstrap")]
        self.assertNotIn("from qsol_geo_reason", before_child_bootstrap)
        self.assertNotIn("import qsol_geo_reason", before_child_bootstrap)

    def test_direct_python_payload_execution_fails_closed(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PYTHON_PAYLOAD), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("internal authenticated payload", completed.stderr)
        self.assertNotIn("usage:", completed.stdout.lower())

    def test_package_local_native_shadow_is_rejected_before_import(self) -> None:
        namespace = runpy.run_path(
            str(PYTHON_PAYLOAD), run_name="qsol_standalone_capture_test"
        )
        child_bootstrap = namespace["_authenticated_capture_bootstrap"]()

        with tempfile.TemporaryDirectory(prefix="qsol-native-shadow-") as tmp:
            src = Path(tmp) / "src"
            package = src / "qsol_geo_reason"
            package.mkdir(parents=True)
            tracked = package / "no_site_subprocess.py"
            tracked.write_text("# clean tracked source\n", encoding="utf-8")
            manifest = {
                "no_site_subprocess.py": hashlib.sha256(tracked.read_bytes()).hexdigest()
            }
            suffix = next(
                (value for value in importlib.machinery.EXTENSION_SUFFIXES if value),
                ".so",
            )
            shadow = package / f"no_site_subprocess{suffix}"
            shadow.write_bytes(
                b"not a real extension; it must be rejected before import\n"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    child_bootstrap,
                    str(src),
                    "a" * 40,
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    "--",
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "rejects importable source shadows before src is trusted",
                completed.stderr,
            )
            self.assertIn(shadow.name, completed.stderr)

    def test_bound_revision_argument_cannot_be_replaced_by_caller(self) -> None:
        namespace = runpy.run_path(
            str(PYTHON_PAYLOAD), run_name="qsol_standalone_capture_test"
        )
        bind = namespace["_bound_cli_arguments"]
        revision = "a" * 40

        observed = bind(
            ["request.json", "--output-dir", "run", "--implementation-revision", revision],
            revision,
        )
        self.assertEqual(observed.count("--implementation-revision"), 1)
        self.assertEqual(observed[-2:], ["--implementation-revision", revision])

        with self.assertRaisesRegex(
            namespace["StandaloneCaptureBootstrapError"],
            "does not match the authenticated",
        ):
            bind(
                [
                    "request.json",
                    "--output-dir",
                    "run",
                    "--implementation-revision",
                    "b" * 40,
                ],
                revision,
            )

    def test_protocol_keeps_installed_command_as_general_production_entry(self) -> None:
        protocol = PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("qsol-geo-capture /tmp/GEO-CAP-001-request.json", protocol)


if __name__ == "__main__":
    unittest.main()
