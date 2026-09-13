from __future__ import annotations

import importlib.util
import inspect
import json
import py_compile
import secrets
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason import first_production_observation_core as CORE
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "run_first_production_observation.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "qsol_test_shadow_hardened_launcher",
        LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load production launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase2AShadowReceiptSymlinkRegressions(unittest.TestCase):
    def test_launcher_rejects_python_package_shadow_before_orchestrator_import(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            tracked_shape = package / "capture_reference_environment.py"
            tracked_shape.write_text("VALUE = 'tracked'\n", encoding="utf-8")
            shadow = package / "capture_reference_environment" / "__init__.py"
            shadow.parent.mkdir()
            shadow.write_text("VALUE = 'ignored-shadow'\n", encoding="utf-8")

            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rejects pure-Python package shadows",
                ):
                    launcher._assert_no_importable_python_shadows()

        bootstrap = launcher._ORCHESTRATOR_BOOTSTRAP
        self.assertIn("pyshadows=", bootstrap)
        self.assertLess(
            bootstrap.index("pyshadows="),
            bootstrap.index("sys.path.insert(0,src)"),
        )
        self.assertLess(
            bootstrap.index("pyshadows="),
            bootstrap.index("from qsol_geo_reason.first_production_observation"),
        )
        main_source = inspect.getsource(launcher.main)
        self.assertLess(
            main_source.index("_assert_no_importable_python_shadows()"),
            main_source.index("os.execve("),
        )

    def test_launcher_rejects_ignored_top_level_module_before_src_prepend(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "src"
            package = source_root / "qsol_geo_reason"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("\n", encoding="utf-8")
            hostile = source_root / "json.py"
            hostile.write_text("raise RuntimeError('ignored top-level shadow executed')\n", encoding="utf-8")

            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "importable source shadows",
                ):
                    launcher._assert_no_importable_python_shadows()

        bootstrap = launcher._ORCHESTRATOR_BOOTSTRAP
        self.assertIn("topmods=", bootstrap)
        self.assertLess(
            bootstrap.index("topmods="),
            bootstrap.index("sys.path.insert(0,src)"),
        )

    def test_launcher_rejects_sourceless_package_bytecode_before_src_prepend(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("\n", encoding="utf-8")
            (package / "canonical.py").write_text("VALUE = 'tracked'\n", encoding="utf-8")
            shadow = package / "canonical" / "__init__.pyc"
            shadow.parent.mkdir()
            shadow.write_bytes(b"sourceless-shadow")

            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "importable source shadows",
                ):
                    launcher._assert_no_importable_python_shadows()

        bootstrap = launcher._ORCHESTRATOR_BOOTSTRAP
        self.assertIn("bytecode=", bootstrap)
        self.assertIn("pkgdirs=", bootstrap)
        self.assertLess(
            bootstrap.index("bytecode="),
            bootstrap.index("sys.path.insert(0,src)"),
        )
        self.assertLess(
            bootstrap.index("pkgdirs="),
            bootstrap.index("sys.path.insert(0,src)"),
        )

    def test_launcher_rejects_valid_cache_tagged_bytecode_before_src_prepend(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("\n", encoding="utf-8")
            source = package / "canonical.py"
            source.write_text("VALUE = 'tracked'\n", encoding="utf-8")
            cache = Path(importlib.util.cache_from_source(str(source)))
            py_compile.compile(str(source), cfile=str(cache), doraise=True)
            self.assertTrue(cache.is_file())
            self.assertEqual(cache.parent.name, "__pycache__")

            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "importable source shadows",
                ):
                    launcher._assert_no_importable_python_shadows()

        bootstrap = launcher._ORCHESTRATOR_BOOTSTRAP
        self.assertIn("bytecode=", bootstrap)
        self.assertNotIn("'__pycache__' not in p.parts", bootstrap)
        self.assertLess(
            bootstrap.index("bytecode="),
            bootstrap.index("sys.path.insert(0,src)"),
        )

    def test_linked_preparation_receipt_survives_parent_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            receipt = directory / "request.preparation-receipt.json"
            value = {"schema_version": "test", "receipt": "durable-bytes"}

            with mock.patch.object(
                CORE,
                "_fsync_directory",
                side_effect=OSError("parent directory fsync failed"),
            ):
                with self.assertRaisesRegex(
                    CaptureContractError,
                    "after no-replace publication",
                ):
                    CORE._exclusive_write_json(receipt, value)

            self.assertTrue(receipt.is_file())
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")),
                value,
            )
            self.assertEqual(
                list(directory.glob(".request.preparation-receipt.json.tmp.*")),
                [],
            )

    def test_prepare_rejects_dangling_checkout_symlink_before_hub_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external_target = Path(tmp) / "outside" / "request.json"
            output = ROOT / f".qsol-dangling-prepare-{secrets.token_hex(8)}.json"
            sidecar = TOOL._default_preparation_receipt_path(output)
            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())
            try:
                try:
                    output.symlink_to(external_target)
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f"symlinks unavailable: {exc}")
                self.assertTrue(output.is_symlink())
                self.assertFalse(external_target.exists())

                with (
                    mock.patch.object(TOOL, "resolve_implementation_revision") as resolver,
                    mock.patch.object(TOOL, "prepare_tree_receipts") as warmup,
                ):
                    with self.assertRaisesRegex(
                        CaptureContractError,
                        "preparation output must be outside the source checkout",
                    ):
                        TOOL.prepare(output)

                resolver.assert_not_called()
                warmup.assert_not_called()
                self.assertTrue(output.is_symlink())
                self.assertFalse(sidecar.exists())
            finally:
                try:
                    output.unlink(missing_ok=True)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
