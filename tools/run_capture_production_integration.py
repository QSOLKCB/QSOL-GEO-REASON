"""Real offline production-backend integration for the Phase 2A reference lane.

The script builds a tiny GPT-2-style model and fast tokenizer locally, arranges them
as Hugging Face cache snapshots with QSOL commit-tree receipts, then invokes the real
qsol-geo-capture CLI twice.  No model weights are downloaded.  Both bundles must pass
Draft 2020-12 schema validation, the canonical semantic verifier, and byte-for-byte
JSON equality across the replay.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "constraints" / "capture-reference-py311.txt"
MODEL_COMMIT = "1" * 40
TOKENIZER_COMMIT = "2" * 40

REFERENCE_VERSIONS = {
    "torch": "2.2.2",
    "transformers": "4.40.2",
    "huggingface-hub": "0.23.5",
    "tokenizers": "0.19.1",
    "safetensors": "0.4.3",
    "numpy": "1.26.4",
    "jsonschema": "4.23.0",
}


def _assert_reference_environment() -> None:
    mismatches: list[str] = []
    for distribution, expected in REFERENCE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{distribution}=MISSING (expected {expected})")
            continue
        # PyTorch's CPU wheel may add a local '+cpu' build label.
        normalized = actual.split("+", 1)[0]
        if normalized != expected:
            mismatches.append(f"{distribution}={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("reference dependency mismatch: " + "; ".join(mismatches))


def _git_blob_id(path: Path) -> str:
    raw = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(b"blob ")
    digest.update(str(len(raw)).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _write_tree_receipt(snapshot: Path, commit: str) -> str:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_dir():
            continue
        relative = path.relative_to(snapshot).as_posix()
        files[relative] = {
            "size": path.stat().st_size,
            "blob_id": _git_blob_id(path),
        }
    if not files:
        raise RuntimeError("integration snapshot contains no files")
    payload = {"format_version": 1, "files": files}
    tree_bytes = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    tree_dir = snapshot.parent.parent / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / f"{commit}.json").write_bytes(tree_bytes)
    return hashlib.sha256(tree_bytes).hexdigest()


def _snapshot_root(hf_home: Path, repo_id: str, commit: str) -> Path:
    storage = hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshot = storage / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    return snapshot


def _build_local_fixture(hf_home: Path) -> tuple[str, str]:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(123456)

    model_snapshot = _snapshot_root(hf_home, "fixture/model", MODEL_COMMIT)
    config = GPT2Config(
        vocab_size=16,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(model_snapshot, safe_serialization=True)

    tokenizer_snapshot = _snapshot_root(
        hf_home, "fixture/tokenizer", TOKENIZER_COMMIT
    )
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "epsilon": 8,
        "zeta": 9,
        "eta": 10,
        "theta": 11,
        "iota": 12,
        "kappa": 13,
        "lambda": 14,
        "mu": 15,
    }
    tokenizer_impl = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tokenizer_snapshot)

    return (
        _write_tree_receipt(model_snapshot, MODEL_COMMIT),
        _write_tree_receipt(tokenizer_snapshot, TOKENIZER_COMMIT),
    )


def _request(model_tree: str, tokenizer_tree: str, constraints_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": "GEO-CAP-001",
        "run_id": "ci-real-backend-reference-001",
        "model": {
            "identifier": "fixture/model",
            "revision": MODEL_COMMIT,
            "revision_kind": "hf_commit",
            "revision_tree_sha256": model_tree,
            "tokenizer_identifier": "fixture/tokenizer",
            "tokenizer_revision": TOKENIZER_COMMIT,
            "tokenizer_revision_kind": "hf_commit",
            "tokenizer_revision_tree_sha256": tokenizer_tree,
        },
        "backend": {
            "name": "huggingface-pytorch",
            "local_files_only": True,
            "trust_remote_code": False,
            "device": "cpu",
            "dtype": "float32",
            "quantization": "none",
        },
        "capture": {
            "context_mode": "cumulative",
            "phase": "replayed_prefix",
            "layers": [0, 1, 2],
            "pooling": {"mode": "step_mean"},
            "prefix_text": "alpha",
            "step_joiner": " ",
        },
        "determinism": {"mode": "required", "seed": 7},
        "generation_parameters": {
            "generation_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
        "steps": [
            {"step_id": "s0", "text": "beta"},
            {"step_id": "s1", "text": "gamma"},
        ],
        "notes": (
            "Real offline CI production-backend reference. "
            f"capture_reference_constraints_sha256={constraints_sha256}"
        ),
    }


def _run_cli(request_path: Path, output_dir: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "qsol_geo_reason.capture_cli",
            str(request_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "real production capture failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"capture CLI emitted unexpected stdout: {completed.stdout!r}")
    return lines[0]


def _read_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(
        json.loads((directory / name).read_text(encoding="utf-8"))
        for name in (
            "capture-request.json",
            "run-manifest.json",
            "captured-trajectory.json",
        )
    )  # type: ignore[return-value]


def _validate_schemas(request: dict[str, Any], manifest: dict[str, Any], trajectory: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    for data, schema_name in (
        (request, "capture-request.schema.json"),
        (manifest, "capture-run-manifest.schema.json"),
        (trajectory, "captured-trajectory.schema.json"),
    ):
        schema = json.loads(
            (ROOT / "schemas" / schema_name).read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(data)


def main() -> int:
    _assert_reference_environment()
    constraints_sha256 = hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory(prefix="qsol-capture-integration-") as tmp:
        workspace = Path(tmp)
        hf_home = workspace / "hf-home"
        model_tree, tokenizer_tree = _build_local_fixture(hf_home)
        request = _request(model_tree, tokenizer_tree, constraints_sha256)
        request_path = workspace / "request.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(hf_home),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )

        first_dir = workspace / "run-a"
        second_dir = workspace / "run-b"
        first_receipt = _run_cli(request_path, first_dir, environment)
        second_receipt = _run_cli(request_path, second_dir, environment)
        if first_receipt != second_receipt:
            raise RuntimeError("deterministic replay produced different manifest receipts")

        first = _read_bundle(first_dir)
        second = _read_bundle(second_dir)
        if first != second:
            raise RuntimeError("deterministic replay produced different canonical JSON bundles")

        captured_request, manifest, trajectory = first
        _validate_schemas(captured_request, manifest, trajectory)

        from qsol_geo_reason.capture import verify_capture_bundle

        verify_capture_bundle(captured_request, manifest, trajectory)

        if manifest["backend_observed"].get("name") != "huggingface-pytorch":
            raise RuntimeError("integration capture did not execute the real production backend")
        if trajectory.get("evidence_class") != "OBSERVATION":
            raise RuntimeError("integration capture was not emitted as OBSERVATION")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "manifest_sha256": first_receipt,
                    "constraints_sha256": constraints_sha256,
                    "torch": importlib.metadata.version("torch"),
                    "transformers": importlib.metadata.version("transformers"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
