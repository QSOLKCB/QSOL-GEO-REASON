from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

from qsol_geo_reason import capture_package as PACKAGE


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "run_first_production_observation.py"
EXPERIMENT = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "qsol_test_bootstrap_revision_launcher",
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


class _FakeDistribution:
    def __init__(self, root: Path, files: list[PurePosixPath]) -> None:
        self._root = root
        self.files = files
        self.metadata = {"Name": "numpy"}
        self.version = "1.26.4"

    def locate_file(self, path):
        return self._root / os.fspath(path)


class BootstrapRevisionAndDistributionRuntimeTests(unittest.TestCase):
    def test_bootstrap_commit_a_cannot_silently_bind_checkout_b(self) -> None:
        launcher = _load_launcher()
        git, git_digest = launcher._trusted_git_identity()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "tools"
            package = root / "src" / "qsol_geo_reason"
            tools.mkdir(parents=True)
            package.mkdir(parents=True)
            launcher_path = tools / "run_first_production_observation.py"
            launcher_a = b"# committed launcher A\n"
            launcher_path.write_bytes(launcher_a)
            (package / "__init__.py").write_text("VALUE = 'A'\n", encoding="utf-8")

            _git(git, root, "init", "-q")
            _git(git, root, "config", "user.email", "qsol-test@example.invalid")
            _git(git, root, "config", "user.name", "QSOL Test")
            _git(git, root, "add", "tools", "src/qsol_geo_reason")
            _git(git, root, "commit", "-qm", "commit A")
            revision_a = _git(git, root, "rev-parse", "HEAD")
            launcher_digest_a = hashlib.sha256(launcher_a).hexdigest()

            launcher_path.write_text("# committed launcher B\n", encoding="utf-8")
            (package / "__init__.py").write_text("VALUE = 'B'\n", encoding="utf-8")
            _git(git, root, "add", "tools", "src/qsol_geo_reason")
            _git(git, root, "commit", "-qm", "commit B")
            revision_b = _git(git, root, "rev-parse", "HEAD")
            self.assertNotEqual(revision_a, revision_b)

            with mock.patch.object(launcher, "ROOT", root):
                launcher._authenticate_bootstrap_launcher_blob(
                    revision=revision_a,
                    launcher_digest=launcher_digest_a,
                    git_path=git,
                    git_digest=git_digest,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "tracked package source that does not match bound revision",
                ):
                    launcher._authenticate_tracked_package_source(
                        git_path=git,
                        git_digest=git_digest,
                        revision=revision_a,
                    )

    def test_documented_bootstrap_resolves_commit_before_launcher_blob(self) -> None:
        text = EXPERIMENT.read_text(encoding="utf-8")
        bootstrap = text.split("QSOL_FIRST_OBSERVATION_BOOTSTRAP=", 1)[1].split(
            "\nqsol_first_observation()", 1
        )[0]
        self.assertIn('"rev-parse","--verify","HEAD^{commit}"', bootstrap)
        self.assertIn('commit+":tools/run_first_production_observation.py"', bootstrap)
        self.assertIn('"_QSOL_AUTHENTICATED_BOOTSTRAP_REVISION":commit', bootstrap)
        self.assertIn('"_QSOL_AUTHENTICATED_LAUNCHER_SHA256":launcher_sha', bootstrap)
        self.assertLess(bootstrap.index('"rev-parse"'), bootstrap.index('"cat-file"'))

        launcher = _load_launcher()
        main_source = importlib.util.find_spec("inspect")
        self.assertIsNotNone(main_source)
        self.assertIn("revision=bootstrap_revision", LAUNCHER.read_text(encoding="utf-8"))
        self.assertIn("'--implementation-revision',revision", launcher._ORCHESTRATOR_BOOTSTRAP)

    def test_distribution_receipt_binds_sibling_numpy_native_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            numpy = site / "numpy"
            numpy_libs = site / "numpy.libs"
            metadata = site / "numpy-1.26.4.dist-info"
            numpy.mkdir()
            numpy_libs.mkdir()
            metadata.mkdir()
            init = numpy / "__init__.py"
            init.write_text("VALUE = 1\n", encoding="utf-8")
            native = numpy_libs / "libgfortran-fixture.so"
            native.write_bytes(b"native-A")
            (metadata / "METADATA").write_text("Name: numpy\nVersion: 1.26.4\n", encoding="utf-8")

            distribution = _FakeDistribution(
                site,
                [
                    PurePosixPath("numpy/__init__.py"),
                    PurePosixPath("numpy.libs/libgfortran-fixture.so"),
                    PurePosixPath("numpy-1.26.4.dist-info/METADATA"),
                ],
            )
            spec = types.SimpleNamespace(
                origin=str(init),
                submodule_search_locations=[str(numpy)],
            )
            with (
                mock.patch.object(
                    PACKAGE.importlib.metadata,
                    "distributions",
                    return_value=[distribution],
                ),
                mock.patch.object(
                    PACKAGE.importlib.util,
                    "find_spec",
                    return_value=spec,
                ),
            ):
                before = PACKAGE._distribution_package_provenance(
                    "numpy",
                    "numpy",
                    "NumPy test distribution",
                )
                native.write_bytes(b"native-B")
                after = PACKAGE._distribution_package_provenance(
                    "numpy",
                    "numpy",
                    "NumPy test distribution",
                )

            self.assertEqual(before["file_count"], 2)
            self.assertEqual(after["file_count"], 2)
            self.assertNotEqual(before["receipt_sha256"], after["receipt_sha256"])

    def test_distribution_receipt_rejects_record_omission_of_sibling_native_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            numpy = site / "numpy"
            numpy_libs = site / "numpy.libs"
            metadata = site / "numpy-1.26.4.dist-info"
            numpy.mkdir()
            numpy_libs.mkdir()
            metadata.mkdir()
            init = numpy / "__init__.py"
            init.write_text("VALUE = 1\n", encoding="utf-8")
            native = numpy_libs / "libgfortran-fixture.so"
            native.write_bytes(b"native-clean")
            (metadata / "METADATA").write_text("Name: numpy\nVersion: 1.26.4\n", encoding="utf-8")

            distribution = _FakeDistribution(
                site,
                [
                    PurePosixPath("numpy/__init__.py"),
                    PurePosixPath("numpy.libs/libgfortran-fixture.so"),
                    PurePosixPath("numpy-1.26.4.dist-info/METADATA"),
                ],
            )
            spec = types.SimpleNamespace(
                origin=str(init),
                submodule_search_locations=[str(numpy)],
            )
            with (
                mock.patch.object(
                    PACKAGE.importlib.metadata,
                    "distributions",
                    return_value=[distribution],
                ),
                mock.patch.object(
                    PACKAGE.importlib.util,
                    "find_spec",
                    return_value=spec,
                ),
            ):
                clean = PACKAGE._distribution_package_provenance(
                    "numpy",
                    "numpy",
                    "NumPy RECORD test distribution",
                )
                self.assertEqual(clean["file_count"], 2)

                distribution.files = [
                    PurePosixPath("numpy/__init__.py"),
                    PurePosixPath("numpy-1.26.4.dist-info/METADATA"),
                ]
                native.write_bytes(b"native-patched-after-record-omission")
                with self.assertRaisesRegex(
                    PACKAGE.CaptureContractError,
                    "file inventory omits independently discovered runtime files.*numpy.libs/libgfortran-fixture.so",
                ):
                    PACKAGE._distribution_package_provenance(
                        "numpy",
                        "numpy",
                        "NumPy RECORD test distribution",
                    )


if __name__ == "__main__":
    unittest.main()
