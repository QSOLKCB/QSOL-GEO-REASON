from __future__ import annotations

import runpy
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

        verifier = source.index('sourcebad=sorted(')
        expose_source = source.index('"sys.path.insert(0,src);"')
        seed_child = source.index('"_qsol_ns.install_authenticated_source_manifest(revision,source_manifest);"')
        import_cli = source.index('"from qsol_geo_reason.capture_cli import main;"')
        self.assertLess(verifier, expose_source)
        self.assertLess(expose_source, seed_child)
        self.assertLess(seed_child, import_cli)

        # The only package imports in this installed script must be inside the
        # authenticated child-bootstrap string, after the manifest check.
        before_child_bootstrap = source[: source.index("def _authenticated_capture_bootstrap")]
        self.assertNotIn("from qsol_geo_reason", before_child_bootstrap)
        self.assertNotIn("import qsol_geo_reason", before_child_bootstrap)

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
