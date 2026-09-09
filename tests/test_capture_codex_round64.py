from __future__ import annotations

import json
import types
import unittest
from pathlib import Path

from qsol_geo_reason import capture_backend_core as core
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import provenance
from qsol_geo_reason.capture import CaptureContractError
from qsol_geo_reason.capture_execute import resolve_implementation_revision


ROOT = Path(__file__).resolve().parents[1]


def _find_private_provenance_globals(value, seen: set[int] | None = None):
    """Locate Round-60's private provenance dictionary through closure cells only."""
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    if isinstance(value, dict) and "_GENERATED_TOP_LEVEL" in value and "git_source_revision" in value:
        return value
    if isinstance(value, types.FunctionType) and value.__closure__:
        for cell in value.__closure__:
            try:
                found = _find_private_provenance_globals(cell.cell_contents, seen)
            except ValueError:
                continue
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_private_provenance_globals(item, seen)
            if found is not None:
                return found
    return None


class CaptureRound64RegressionTests(unittest.TestCase):
    def test_round60_canonical_git_does_not_consult_round56_helpers(self):
        original_candidates = round56._system_git_candidates
        original_hash = round56._hash_executable
        try:
            def forbidden_candidates():
                raise AssertionError("Round60 consulted writable Round56 candidate resolver")

            def forbidden_hash(_path):
                raise AssertionError("Round60 consulted writable Round56 executable hasher")

            round56._system_git_candidates = forbidden_candidates
            round56._hash_executable = forbidden_hash
            observed = resolve_implementation_revision(require_checkout=True)
            self.assertRegex(observed, r"^[0-9a-f]{40}$")
        finally:
            round56._system_git_candidates = original_candidates
            round56._hash_executable = original_hash

    def test_round60_detaches_mutable_public_provenance_values(self):
        private = _find_private_provenance_globals(resolve_implementation_revision)
        self.assertIsNotNone(private)
        private_generated = private["_GENERATED_TOP_LEVEL"]
        self.assertIsInstance(private_generated, frozenset)
        self.assertIsNot(private_generated, provenance._GENERATED_TOP_LEVEL)

        already_present = "src" in provenance._GENERATED_TOP_LEVEL
        try:
            provenance._GENERATED_TOP_LEVEL.add("src")
            self.assertNotIn("src", private_generated)
        finally:
            if not already_present:
                provenance._GENERATED_TOP_LEVEL.discard("src")

    def test_position_limit_accepts_gpt2_aliases_and_rejects_ambiguity(self):
        backend = object.__new__(core.HuggingFacePyTorchBackend)
        backend._block_path = "transformer.h"
        backend._model = types.SimpleNamespace(
            config=types.SimpleNamespace(n_positions=1024, n_ctx=1024)
        )
        self.assertEqual(backend._model_position_limit(), 1024)

        backend._model = types.SimpleNamespace(
            config=types.SimpleNamespace(n_positions=1024, n_ctx=512)
        )
        with self.assertRaisesRegex(CaptureContractError, "ambiguous position limits"):
            backend._model_position_limit()

        backend._model = types.SimpleNamespace(
            config=types.SimpleNamespace(n_positions=True)
        )
        with self.assertRaisesRegex(CaptureContractError, "positive integer"):
            backend._model_position_limit()

    def test_supported_decoder_without_position_bound_fails_closed(self):
        backend = object.__new__(core.HuggingFacePyTorchBackend)
        backend._block_path = "transformer.h"
        backend._model = types.SimpleNamespace(config=types.SimpleNamespace())
        with self.assertRaisesRegex(CaptureContractError, "no recognized model position limit"):
            backend._model_position_limit()

    def test_schema_requires_hub_receipt_and_device_conditional_mps_receipt(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        production = schema["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        self.assertIn("huggingface_hub_package_file_count", required)
        self.assertIn("huggingface_hub_package_receipt_sha256", required)

        semantic = "\n".join(schema["x-qsol-semantic-validation"]["constraints"])
        self.assertIn("QSOL_GEO_MPS_RUNTIME", semantic)
        self.assertIn("Hugging Face Hub package", semantic)

        mps_rule = production["allOf"][0]
        mps_pattern = mps_rule["then"]["properties"]["torch_build_config"]["pattern"]
        self.assertIn("QSOL_GEO_MPS_RUNTIME", mps_pattern)
        non_mps_pattern = mps_rule["else"]["properties"]["torch_build_config"]["not"]["pattern"]
        self.assertIn("QSOL_GEO_MPS_RUNTIME", non_mps_pattern)

    def test_core_content_binds_hub_before_snapshot_resolution(self):
        source = Path(core.__file__).read_text(encoding="utf-8")
        self.assertIn('_preimport_package_provenance(\n            "huggingface_hub"', source)
        self.assertIn("loader_bindings_before_hub = _critical_loader_bindings", source)
        self.assertIn("loader_bindings_after_hub = _critical_loader_bindings", source)
        self.assertIn("Hugging Face Hub execution changed a Transformers loader/deserializer binding", source)
        self.assertIn('"huggingface_hub_package_receipt_sha256"', source)


if __name__ == "__main__":
    unittest.main()
