"""Round 35 regressions for checkout, tokenizer, and manifest-schema provenance."""
from __future__ import annotations

import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture_backend_audit import (
    HuggingFacePyTorchBackend as AuditBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.provenance import SourceIdentityError, git_source_revision

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound35RegressionTests(unittest.TestCase):
    @staticmethod
    def _initialize_source_repo(root: Path) -> Path:
        package = root / "src" / "qsol_geo_reason"
        package.mkdir(parents=True)
        source = package / "capture.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
        return source

    def test_checkout_bytes_ignore_assume_unchanged_and_skip_worktree_hints(self):
        for flag in ("--assume-unchanged", "--skip-worktree"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = self._initialize_source_repo(root)
                relative = "src/qsol_geo_reason/capture.py"
                subprocess.run(
                    ["git", "-C", str(root), "update-index", flag, relative],
                    check=True,
                )
                source.write_text("VALUE = 2\n", encoding="utf-8")

                # Establish the review's premise: ordinary status is blind to the edit.
                status = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertEqual(status, "")

                with patch(
                    "qsol_geo_reason.provenance.source_repo_root", return_value=root
                ):
                    with self.assertRaisesRegex(
                        SourceIdentityError, "independently of index flags"
                    ):
                        git_source_revision(
                            require_clean=True,
                            reject_importable_bytecode=True,
                        )

    def test_every_slow_tokenizer_lane_fails_closed_without_a_receipt(self):
        tagger_type = type("Tagger", (), {})
        tagger_type.__module__ = "fugashi.fugashi"
        tokenizer = types.SimpleNamespace(mecab=tagger_type())
        backend = object.__new__(AuditBackend)
        backend._tokenizer = tokenizer
        backend._tokenizers_package_provenance_initialized = False
        backend._tokenizers_native_backend_active = False
        backend._tokenizers_package_provenance = None

        self.assertEqual(
            backend._native_slow_tokenizer_backend(tokenizer),
            "unbound-slow-tokenizer",
        )
        with self.assertRaisesRegex(CaptureContractError, "slow-tokenizer execution"):
            backend._initialize_tokenizers_package_provenance()

        fast = types.SimpleNamespace(backend_tokenizer=object())
        self.assertIsNone(backend._native_slow_tokenizer_backend(fast))

    def test_manifest_schema_ties_requested_and_observed_instrument_fields(self):
        schema = json.loads(
            (ROOT / "schemas/capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        rules = schema["allOf"]

        def request_rule(field: str, expected: str):
            return next(
                rule
                for rule in rules
                if rule.get("if", {})
                .get("properties", {})
                .get("backend_request", {})
                .get("properties", {})
                .get(field, {})
                .get("const")
                == expected
            )

        for device in ("cpu", "mps"):
            rule = request_rule("device", device)
            self.assertEqual(
                rule["then"]["properties"]["backend_observed"]["properties"][
                    "device"
                ],
                {"const": device},
            )

        cuda_rule = next(
            rule
            for rule in rules
            if rule.get("if", {})
            .get("properties", {})
            .get("backend_request", {})
            .get("properties", {})
            .get("device", {})
            .get("pattern")
            == "^cuda:[0-9]+$"
        )
        self.assertEqual(
            cuda_rule["then"]["properties"]["backend_observed"]["properties"][
                "device"
            ]["pattern"],
            "^cuda:[0-9]+$",
        )

        for dtype in ("float32", "float16", "bfloat16"):
            rule = request_rule("dtype", dtype)
            self.assertEqual(
                rule["then"]["properties"]["backend_observed"]["properties"][
                    "dtype"
                ],
                {"const": dtype},
            )

        for mode in ("required", "best_effort"):
            rule = next(
                item
                for item in rules
                if item.get("if", {})
                .get("properties", {})
                .get("determinism", {})
                .get("properties", {})
                .get("mode", {})
                .get("const")
                == mode
            )
            self.assertEqual(
                rule["then"]["properties"]["backend_observed"]["properties"][
                    "determinism_mode"
                ],
                {"const": mode},
            )

        semantic = schema["x-qsol-semantic-validation"]
        self.assertIs(semantic["required"], True)
        self.assertTrue(
            any("exact cuda:N" in constraint for constraint in semantic["constraints"])
        )


if __name__ == "__main__":
    unittest.main()
