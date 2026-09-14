from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "qsol-geo-capture"
PYPROJECT = ROOT / "pyproject.toml"
PROTOCOL = ROOT / "protocols" / "GEO-CAP-001.md"


class StandaloneCaptureBootstrapTests(unittest.TestCase):
    def test_packaging_routes_qsol_geo_capture_through_standalone_script(self) -> None:
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        self.assertIn('script-files = ["scripts/qsol-geo-capture"]', pyproject)
        self.assertNotIn(
            'qsol-geo-capture = "qsol_geo_reason.capture_cli:main"',
            pyproject,
        )

    def test_first_interpreter_is_isolated_and_no_site_before_bootstrap(self) -> None:
        first_line = BOOTSTRAP.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first_line, "#!/usr/bin/python3 -ISB")
        self.assertNotIn("/usr/bin/env", first_line)
        self.assertIn("-I", "-ISB")
        self.assertIn("S", "-ISB")
        self.assertIn("B", "-ISB")

    def test_bootstrap_authenticates_before_any_package_import(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('_BOOTSTRAP_GIT_PATH = "scripts/qsol-geo-capture"', source)
        self.assertIn('"HEAD^{commit}"', source)
        self.assertIn('"ls-tree"', source)
        self.assertIn('_PACKAGE_GIT_ROOT = "src/qsol_geo_reason"', source)
        self.assertIn("_authenticate_installed_bootstrap", source)
        self.assertIn("_git_blob_sha1(raw) != object_id", source)
        self.assertIn('"-I",', source)
        self.assertIn('"-S",', source)
        self.assertIn('"-B",', source)

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

    def test_package_local_native_shadow_is_rejected_before_import(self) -> None:
        namespace = runpy.run_path(str(BOOTSTRAP), run_name="qsol_standalone_capture_test")
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
                (
                    value
                    for value in importlib.machinery.EXTENSION_SUFFIXES
                    if value
                ),
                ".so",
            )
            shadow = package / f"no_site_subprocess{suffix}"
            shadow.write_bytes(b"not a real extension; it must be rejected before import\n")

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
        namespace = runpy.run_path(str(BOOTSTRAP), run_name="qsol_standalone_capture_test")
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
