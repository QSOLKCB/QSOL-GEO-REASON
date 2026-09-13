from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_hub_tree as HUB
from qsol_geo_reason.capture_common import CaptureContractError


class HubTreeReceiptImmutabilityTests(unittest.TestCase):
    COMMIT = "a" * 40

    def _snapshot(self, root: Path) -> Path:
        snapshot = root / "models--qsol--fixture" / "snapshots" / self.COMMIT
        snapshot.mkdir(parents=True)
        return snapshot

    def _files(self, *, size: int = 1, blob: str = "b") -> dict[str, dict[str, object]]:
        return {
            "config.json": {
                "size": size,
                "blob_id": blob * 40,
            }
        }

    def _tree_bytes(self, files: dict[str, dict[str, object]]) -> bytes:
        return (
            json.dumps(
                {
                    "format_version": HUB._TREE_CACHE_FORMAT_VERSION,
                    "files": dict(sorted(files.items())),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )

    def test_existing_identical_commit_tree_is_retained_and_directory_synced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._snapshot(root)
            files = self._files()
            destination = snapshot.parent.parent / "trees" / f"{self.COMMIT}.json"
            destination.parent.mkdir()
            expected_bytes = self._tree_bytes(files)
            destination.write_bytes(expected_bytes)
            before = destination.stat()

            with (
                mock.patch.object(HUB, "_cached_hub_commit_tree", return_value=files),
                mock.patch.object(HUB, "_fsync_directory") as sync_directory,
            ):
                receipt = HUB._write_tree_artifact(snapshot, self.COMMIT, files, "model")

            after = destination.stat()
            self.assertEqual(destination.read_bytes(), expected_bytes)
            self.assertEqual(receipt, hashlib.sha256(expected_bytes).hexdigest())
            # A matching pre-existing artifact must be reused, not replaced, but
            # its containing directory must still be synced in case a concurrent
            # publisher linked the name and had not reached its own directory fsync.
            self.assertEqual(after.st_ino, before.st_ino)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            sync_directory.assert_called_once_with(destination.parent)
            self.assertEqual(list(destination.parent.glob(f".{self.COMMIT}.*.tmp")), [])

    def test_existing_different_commit_tree_fails_closed_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._snapshot(root)
            newly_observed = self._files()
            destination = snapshot.parent.parent / "trees" / f"{self.COMMIT}.json"
            destination.parent.mkdir()
            preserved_bytes = self._tree_bytes(self._files(size=2, blob="c"))
            destination.write_bytes(preserved_bytes)
            before = destination.stat()

            with (
                mock.patch.object(HUB, "_cached_hub_commit_tree") as verifier,
                self.assertRaisesRegex(
                    CaptureContractError,
                    "differs from newly observed metadata",
                ),
            ):
                HUB._write_tree_artifact(
                    snapshot,
                    self.COMMIT,
                    newly_observed,
                    "model",
                )

            after = destination.stat()
            self.assertEqual(destination.read_bytes(), preserved_bytes)
            self.assertEqual(after.st_ino, before.st_ino)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(list(destination.parent.glob(f".{self.COMMIT}.*.tmp")), [])
            verifier.assert_not_called()

    def test_staging_name_collision_does_not_unlink_unowned_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._snapshot(root)
            files = self._files()
            tree_dir = snapshot.parent.parent / "trees"
            tree_dir.mkdir()
            nonce = "c" * 32
            collision = tree_dir / (
                f".{self.COMMIT}.{os.getpid()}.{nonce}.tmp"
            )
            preserved = b"other-preparation-live-staging\n"
            collision.write_bytes(preserved)

            with (
                mock.patch.object(HUB.secrets, "token_hex", return_value=nonce),
                mock.patch.object(HUB, "_cached_hub_commit_tree") as verifier,
                self.assertRaisesRegex(
                    CaptureContractError,
                    "unable to persist trusted model Hub tree artifact",
                ),
            ):
                HUB._write_tree_artifact(snapshot, self.COMMIT, files, "model")

            # open("xb") never established ownership, so cleanup must leave the
            # pre-existing staging file untouched rather than deleting another
            # process's live file (or a stale crash artifact).
            self.assertTrue(collision.is_file())
            self.assertEqual(collision.read_bytes(), preserved)
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
