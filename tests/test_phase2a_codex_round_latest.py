from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_hub_tree as HUB_TREE
from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "run_first_production_observation.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "qsol_test_latest_hardened_launcher",
        LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load production launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase2ALatestCodexReviewTests(unittest.TestCase):
    def test_launcher_rejects_native_extension_shadowing_before_package_import(self) -> None:
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            hostile = package / "capture_reference_environment.cpython-311-x86_64-linux-gnu.so"
            hostile.write_bytes(b"not-a-real-extension")
            with mock.patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rejects native extension artifacts",
                ):
                    launcher._assert_no_importable_native_extensions()

        bootstrap = launcher._ORCHESTRATOR_BOOTSTRAP
        self.assertIn("importlib.machinery.EXTENSION_SUFFIXES", bootstrap)
        self.assertIn("native=", bootstrap)
        self.assertLess(
            bootstrap.index("native="),
            bootstrap.index("from qsol_geo_reason.first_production_observation"),
        )
        main_source = inspect.getsource(launcher.main)
        self.assertLess(
            main_source.index("_assert_no_importable_native_extensions()"),
            main_source.index("os.execve("),
        )

    def test_hub_tree_publication_fsyncs_tree_directory_after_replace(self) -> None:
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--qsol--fixture"
            snapshot = root / "snapshots" / commit
            snapshot.mkdir(parents=True)
            tree_dir = root / "trees"
            synced: list[Path] = []
            real_ensure = HUB_TREE._ensure_parent_directory_durable
            real_fsync = HUB_TREE._fsync_directory

            def record_sync(path: Path) -> None:
                synced.append(Path(path))
                real_fsync(path)

            with (
                mock.patch.object(
                    HUB_TREE,
                    "_ensure_parent_directory_durable",
                    wraps=real_ensure,
                ) as ensure,
                mock.patch.object(
                    HUB_TREE,
                    "_fsync_directory",
                    side_effect=record_sync,
                ),
                mock.patch.object(HUB_TREE, "_cached_hub_commit_tree"),
            ):
                receipt = HUB_TREE._write_tree_artifact(
                    snapshot,
                    commit,
                    {"config.json": {"size": 2, "blob_id": "b" * 40}},
                    "model",
                )

            ensure.assert_called_once_with(tree_dir)
            self.assertIn(tree_dir, synced)
            self.assertRegex(receipt, r"^[0-9a-f]{64}$")
            self.assertTrue((tree_dir / f"{commit}.json").is_file())

    def test_post_verdict_failure_marker_is_distinct_from_pre_verdict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "observation"
            output_root.mkdir()
            (output_root / "replay-verdict.json").write_text("{}\n", encoding="utf-8")
            preparation = {
                "preparation_repository_commit": "c" * 40,
                "preparation_receipt_sha256": "d" * 64,
            }
            preserved = TOOL._preserve_failed_attempt(
                output_root,
                CaptureContractError("post-publication verification read failed"),
                "e" * 40,
                {"run-a": "EXEC-A", "run-b": "EXEC-B"},
                preparation,
                replay_verdict_published=True,
            )
            marker = json.loads(
                (preserved / "execution-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["attempt_status"],
                "failed_after_replay_verdict_publication",
            )
            self.assertTrue(marker["replay_verdict_published"])
            self.assertTrue((preserved / "replay-verdict.json").is_file())

        observe_source = inspect.getsource(TOOL.observe)
        self.assertIn("replay_verdict_published = False", observe_source)
        self.assertIn("replay_verdict_published = True", observe_source)
        self.assertIn(
            "replay_verdict_published=replay_verdict_published",
            observe_source,
        )


if __name__ == "__main__":
    unittest.main()
