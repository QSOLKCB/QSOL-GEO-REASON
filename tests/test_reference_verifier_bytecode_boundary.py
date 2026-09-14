from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_capture_reference_environment.py"
WORKFLOW = ROOT / ".github" / "workflows" / "capture-production-integration.yml"


class ReferenceVerifierBytecodeBoundaryTests(unittest.TestCase):
    def test_verifier_disables_bytecode_before_qsol_imports(self) -> None:
        source = VERIFIER.read_text(encoding="utf-8")
        self.assertIn("sys.dont_write_bytecode = True", source)
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("from qsol_geo_reason import capture_reference_environment"),
        )
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("from qsol_geo_reason.capture_common import CaptureContractError"),
        )

    def test_plain_python_verifier_import_does_not_create_qsol_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            tools = root / "tools"
            package.mkdir(parents=True)
            tools.mkdir()

            (package / "__init__.py").write_text("\n", encoding="utf-8")
            (package / "capture_reference_environment.py").write_text(
                "LOCK = None\n",
                encoding="utf-8",
            )
            (package / "capture_common.py").write_text(
                "class CaptureContractError(Exception):\n    pass\n",
                encoding="utf-8",
            )
            verifier = tools / VERIFIER.name
            shutil.copyfile(VERIFIER, verifier)

            bootstrap = (
                "import importlib.util,sys;"
                f"sys.path.insert(0,{str((root / 'src').resolve())!r});"
                f"spec=importlib.util.spec_from_file_location('qsol_test_verifier',{str(verifier.resolve())!r});"
                "module=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", bootstrap],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(list(package.rglob("*.pyc")), [])
            self.assertFalse((package / "__pycache__").exists())

    def test_production_integration_invokes_verifier_with_dash_b(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "run: python -B tools/verify_capture_reference_environment.py",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
