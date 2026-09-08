"""Round 37 regressions for the Astra capture/provenance review."""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture_backend_final import (
    HuggingFacePyTorchBackend as FinalBackend,
    _assert_safetensors_only_checkpoint,
    _remember_final_construction_baseline,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_dispatch import _validate_dispatch_metadata
from qsol_geo_reason.capture_snapshot import _snapshot_file_hashes
from qsol_geo_reason.capture_validation import validate_capture_request
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


def write_tree(storage: Path, revision: str, payload: bytes) -> Path:
    tree = {
        "format_version": 1,
        "files": {
            "config.json": {
                "size": len(payload),
                "blob_id": git_blob_sha1(payload),
            }
        },
    }
    path = storage / "trees" / f"{revision}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, sort_keys=True), encoding="utf-8")
    return path


def make_snapshot(storage: Path, revision: str, payload: bytes) -> tuple[Path, Path]:
    snapshot = storage / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (storage / "blobs").mkdir(exist_ok=True)
    (snapshot / "config.json").write_bytes(payload)
    return snapshot, write_tree(storage, revision, payload)


def make_final_fixture() -> FinalBackend:
    backend = object.__new__(FinalBackend)
    backend._torch_build_provenance = {"build": "original"}
    backend._attention_implementation = "sdpa"
    backend._model = types.SimpleNamespace(
        config=types.SimpleNamespace(_attn_implementation="sdpa")
    )
    backend._model_snapshot_hashes = {"model.safetensors": "a" * 64}
    backend._tokenizer_snapshot_hashes = {"tokenizer.json": "b" * 64}
    _remember_final_construction_baseline(
        backend,
        (
            "1" * 64,
            "2" * 64,
            sha256_json(backend._torch_build_provenance),
            "sdpa",
            sha256_json(dict(sorted(backend._model_snapshot_hashes.items()))),
            sha256_json(dict(sorted(backend._tokenizer_snapshot_hashes.items()))),
        ),
    )
    return backend


class CaptureRound37RegressionTests(unittest.TestCase):
    def test_git_clean_filter_cannot_hide_modified_importable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git(repo, "init")
            git(repo, "config", "user.email", "round37@example.invalid")
            git(repo, "config", "user.name", "Round 37")
            source = repo / "src" / "qsol_geo_reason"
            source.mkdir(parents=True)
            probe = source / "probe.py"
            probe.write_text("VALUE = 1\n", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "original")

            cleaner = Path(tmp) / "clean.py"
            cleaner.write_text(
                "import sys\nsys.stdin.buffer.read()\nsys.stdout.write('VALUE = 1\\n')\n",
                encoding="utf-8",
            )
            git(repo, "config", "filter.mask.clean", f"python {cleaner}")
            git(repo, "config", "filter.mask.smudge", "cat")
            attributes = repo / ".git" / "info" / "attributes"
            attributes.write_text(
                "src/qsol_geo_reason/probe.py filter=mask\n", encoding="utf-8"
            )
            probe.write_text("VALUE = 2\n", encoding="utf-8")

            # Git's normal filtered comparison sees the clean-filter output as the
            # committed bytes, while canonical raw-byte provenance must not.
            self.assertEqual(git(repo, "status", "--porcelain=v1"), "")
            with patch("qsol_geo_reason.provenance.source_repo_root", return_value=repo):
                with self.assertRaises(SourceIdentityError):
                    git_source_revision(
                        require_clean=True,
                        reject_importable_bytecode=True,
                    )

    def test_frozen_tree_receipt_rejects_rewritten_snapshot_and_tree(self):
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            snapshot, tree_path = make_snapshot(storage, revision, b"canonical bytes")
            frozen = hashlib.sha256(tree_path.read_bytes()).hexdigest()
            self.assertEqual(
                _snapshot_file_hashes(
                    snapshot,
                    revision,
                    "model",
                    expected_tree_receipt_sha256=frozen,
                ),
                {"config.json": hashlib.sha256(b"canonical bytes").hexdigest()},
            )

            # Rewrite both the snapshot and its adjacent local tree consistently.
            # Local consistency alone must no longer establish the requested commit.
            replacement = b"replacement bytes"
            (snapshot / "config.json").write_bytes(replacement)
            write_tree(storage, revision, replacement)
            with self.assertRaisesRegex(CaptureContractError, "frozen request receipt"):
                _snapshot_file_hashes(
                    snapshot,
                    revision,
                    "model",
                    expected_tree_receipt_sha256=frozen,
                )

    def test_safetensors_only_gate_runs_before_inherited_model_loader(self):
        _assert_safetensors_only_checkpoint({"model.safetensors": "a" * 64})
        with self.assertRaisesRegex(CaptureContractError, "requires Safetensors"):
            _assert_safetensors_only_checkpoint({"pytorch_model.bin": "a" * 64})
        with self.assertRaisesRegex(CaptureContractError, "forbids pickle-capable"):
            _assert_safetensors_only_checkpoint(
                {"model.safetensors": "a" * 64, "pytorch_model.bin": "b" * 64}
            )

        # Python invokes __new__ before the inherited production __init__. Keep the
        # pre-load gate there so the established production constructor remains intact.
        source = inspect.getsource(FinalBackend.__new__)
        self.assertLess(
            source.index("_assert_safetensors_only_checkpoint"),
            source.index("_remember_final_construction_preflight"),
        )
        self.assertNotIn("__init__", FinalBackend.__dict__)

    def test_external_torch_build_baseline_rejects_instance_map_rewrite(self):
        backend = make_final_fixture()
        backend._assert_final_construction_baseline()
        backend._torch_build_provenance = {"build": "forged"}
        with self.assertRaisesRegex(CaptureContractError, "PyTorch build provenance changed"):
            backend._assert_final_construction_baseline()

    def test_external_attention_baseline_rejects_cuda_style_metadata_rewrite(self):
        backend = make_final_fixture()
        backend._attention_implementation = "eager"
        with self.assertRaisesRegex(CaptureContractError, "attention implementation changed"):
            backend._assert_final_construction_baseline()

        backend = make_final_fixture()
        backend._model.config._attn_implementation = "eager"
        with self.assertRaisesRegex(CaptureContractError, "attention implementation changed"):
            backend._assert_final_construction_baseline()

    def test_inactive_cuda_identity_is_rejected_by_semantics_and_schema(self):
        observed = {
            "cuda_device_name": "forged-cuda-device",
            "cuda_device_capability": None,
            "cuda_device_uuid": None,
            "cuda_matmul_allow_fp16_accumulation": None,
            "cpu_aten_capability": "DEFAULT",
            "aten_cpu_capability_env": None,
            "aten_cpu_capability_env_known": False,
        }
        with self.assertRaisesRegex(CaptureContractError, "CUDA device identity"):
            _validate_dispatch_metadata(observed, "cpu")

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        cuda_identity_rule = next(
            rule
            for rule in production["allOf"]
            if "else" in rule
            and "cuda_device_name" in rule.get("then", {}).get("properties", {})
        )
        outside_cuda = cuda_identity_rule["else"]["properties"]
        for field in ("cuda_device_name", "cuda_device_capability", "cuda_device_uuid"):
            self.assertEqual(outside_cuda[field], {"type": "null"})

    def test_tree_receipts_are_normalized_and_manifest_schema_compatible(self):
        request = json.loads(
            (ROOT / "fixtures" / "capture-contract-request.json").read_text(
                encoding="utf-8"
            )
        )
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        validated = validate_capture_request(request)
        self.assertEqual(validated["model"]["revision_tree_sha256"], "1" * 64)
        self.assertEqual(validated["model"]["tokenizer_revision_tree_sha256"], "2" * 64)

        request_schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
        )
        manifest_schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        for schema in (request_schema, manifest_schema):
            model = schema["$defs"]["model"]["properties"]
            self.assertIn("revision_tree_sha256", model)
            self.assertIn("tokenizer_revision_tree_sha256", model)


if __name__ == "__main__":
    unittest.main()
