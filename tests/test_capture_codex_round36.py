"""Round 36 source, snapshot, live-state, and CPU-pooling provenance regressions."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture_backend import (
    HuggingFacePyTorchBackend as PolicyBackend,
    _remember_live_state_baseline,
)
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_dispatch import _validate_dispatch_metadata
from qsol_geo_reason.capture_snapshot import _snapshot_file_hashes
from qsol_geo_reason.provenance import SourceIdentityError, git_source_revision

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()


def make_snapshot(storage: Path, revision: str, payload: bytes = b"canonical bytes") -> Path:
    snapshot = storage / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (storage / "blobs").mkdir(exist_ok=True)
    (storage / "trees").mkdir(exist_ok=True)
    (snapshot / "config.json").write_bytes(payload)
    tree = {
        "format_version": 1,
        "files": {
            "config.json": {
                "size": len(payload),
                "blob_id": git_blob_sha1(payload),
            }
        },
    }
    (storage / "trees" / f"{revision}.json").write_text(
        json.dumps(tree), encoding="utf-8"
    )
    return snapshot


class CaptureRound36RegressionTests(unittest.TestCase):
    def test_external_live_state_vault_cannot_be_disabled_by_instance_flag(self):
        backend = object.__new__(PolicyBackend)
        float64, long = object(), object()
        backend._torch = types.SimpleNamespace(float64=float64, long=long)
        _remember_live_state_baseline(
            backend,
            ("original-live", "original-content", "original-tokenizer", float64, long),
        )
        # This is the exact caller mutation from the review finding. The legacy
        # marker must have no authority over the closure-owned baseline vault.
        backend._live_state_seal_initialized = False
        backend._model_live_state_seal = lambda: "changed-live"
        backend._model_content_state_seal = lambda: "original-content"
        backend._tokenizer_live_state_seal = lambda: "original-tokenizer"
        with self.assertRaisesRegex(CaptureContractError, "live model state changed"):
            backend._assert_live_state_authentication()

        for method in (
            PolicyBackend._assert_live_state_authentication,
            PolicyBackend.hidden_states,
            PolicyBackend.metadata,
        ):
            self.assertNotIn("_live_state_seal_initialized", inspect.getsource(method))

    def test_git_replace_cannot_rebind_checkout_to_unreplaced_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init")
            git(repo, "config", "user.email", "round36@example.invalid")
            git(repo, "config", "user.name", "Round 36")
            source = repo / "src" / "qsol_geo_reason"
            source.mkdir(parents=True)
            probe = source / "probe.py"
            probe.write_text("VALUE = 1\n", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "original")
            original = git(repo, "rev-parse", "HEAD")

            probe.write_text("VALUE = 2\n", encoding="utf-8")
            git(repo, "commit", "-am", "replacement")
            replacement = git(repo, "rev-parse", "HEAD")
            git(repo, "checkout", "--detach", original)
            git(repo, "replace", original, replacement)
            # Materialize the replacement tree while leaving HEAD named by the
            # original commit. Ordinary Git now reports a clean replacement view.
            git(repo, "reset", "--hard", "HEAD")
            self.assertEqual(git(repo, "rev-parse", "HEAD"), original)
            self.assertEqual(git(repo, "status", "--porcelain=v1"), "")
            self.assertEqual(probe.read_text(encoding="utf-8"), "VALUE = 2\n")

            with patch("qsol_geo_reason.provenance.source_repo_root", return_value=repo):
                with self.assertRaises(SourceIdentityError):
                    git_source_revision(
                        require_clean=True,
                        reject_importable_bytecode=True,
                    )

    def test_cached_hub_tree_binds_snapshot_bytes_not_just_directory_name(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            snapshot = make_snapshot(storage, revision)
            expected_sha = hashlib.sha256(b"canonical bytes").hexdigest()
            self.assertEqual(
                _snapshot_file_hashes(snapshot, revision, "model"),
                {"config.json": expected_sha},
            )

            # Keep the requested-commit directory name but replace its bytes.
            (snapshot / "config.json").write_bytes(b"replacement bytes")
            with self.assertRaisesRegex(CaptureContractError, "cached Hub"):
                _snapshot_file_hashes(snapshot, revision, "model")

    def test_snapshot_without_cached_hub_commit_tree_fails_closed(self):
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            snapshot = make_snapshot(storage, revision)
            (storage / "trees" / f"{revision}.json").unlink()
            with self.assertRaisesRegex(CaptureContractError, "lacks cached Hub commit-tree metadata"):
                _snapshot_file_hashes(snapshot, revision, "tokenizer")

    def test_cpu_pooling_dispatch_is_mandatory_for_every_production_device(self):
        for device in ("cpu", "cuda:0", "mps"):
            observed = {
                "cuda_matmul_allow_fp16_accumulation": None,
                "cpu_aten_capability": "DEFAULT",
                "aten_cpu_capability_env": None,
                "aten_cpu_capability_env_known": False,
                "mps_mac_model": "SYNTHETIC-MPS" if device == "mps" else None,
                "mps_cpu_brand": None,
            }
            with self.subTest(device=device):
                _validate_dispatch_metadata(observed, device)
                erased = dict(observed, cpu_aten_capability=None)
                with self.assertRaisesRegex(CaptureContractError, "CPU capability"):
                    _validate_dispatch_metadata(erased, device)

    def test_production_constructor_and_schema_bind_cpu_pooling_dispatch_on_accelerators(self):
        source = inspect.getsource(ProductionBackend.__init__)
        assignment = source.index("self._canonical_cpu_dispatch = {")
        processor_only = source.index('if device == "cpu":', assignment)
        self.assertLess(assignment, processor_only)

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        self.assertEqual(
            production["properties"]["cpu_aten_capability"]["enum"],
            ["DEFAULT", "VSX", "Z VECTOR", "NO AVX", "AVX2", "AVX512", "SVE256"],
        )
        self.assertEqual(
            production["properties"]["aten_cpu_capability_env_known"]["type"],
            "boolean",
        )
        cpu_rule = next(
            rule
            for rule in production["allOf"]
            if rule["if"].get("properties", {}).get("device", {}).get("const") == "cpu"
        )
        outside_cpu = cpu_rule["else"]["properties"]
        for field in (
            "cpu_aten_capability",
            "aten_cpu_capability_env",
            "aten_cpu_capability_env_known",
        ):
            self.assertNotIn(field, outside_cpu)


if __name__ == "__main__":
    unittest.main()
