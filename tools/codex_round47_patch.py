"""Temporary guarded patcher for Codex review round 47; deleted by its workflow."""
from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one guarded match, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Persist omitted execution-policy state in the canonical producer metadata.
replace_once(
    "src/qsol_geo_reason/capture_backend_core.py",
    """        return enabled

    def _assert_required_determinism_policy(self) -> bool:
""",
    """        return enabled

    def _deterministic_warn_only_state(self) -> bool:
        checker = getattr(self._torch, "is_deterministic_algorithms_warn_only_enabled", None)
        if not callable(checker):
            raise CaptureContractError(
                "canonical capture requires torch.is_deterministic_algorithms_warn_only_enabled"
            )
        enabled = checker()
        if not isinstance(enabled, bool):
            raise CaptureContractError("deterministic warn-only state must be boolean")
        return enabled

    def _assert_required_determinism_policy(self) -> bool:
""",
)
replace_once(
    "src/qsol_geo_reason/capture_backend_core.py",
    """        if deterministic_enabled is None:
            deterministic_enabled = self._deterministic_algorithms_state()
        return {
""",
    """        if deterministic_enabled is None:
            deterministic_enabled = self._deterministic_algorithms_state()
        deterministic_warn_only = self._deterministic_warn_only_state()
        return {
""",
)
replace_once(
    "src/qsol_geo_reason/capture_backend_core.py",
    """            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_fast_math_env": os.environ.get("PYTORCH_MPS_FAST_MATH"),
""",
    """            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_fast_math_env": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "mps_prefer_metal_env": os.environ.get("PYTORCH_MPS_PREFER_METAL"),
""",
)
replace_once(
    "src/qsol_geo_reason/capture_backend_core.py",
    """            "kv_cache_reuse": False, "deterministic_algorithms_enabled": deterministic_enabled,
            "determinism_mode": self._determinism_mode,
""",
    """            "kv_cache_reuse": False, "deterministic_algorithms_enabled": deterministic_enabled,
            "deterministic_warn_only_enabled": deterministic_warn_only,
            "determinism_mode": self._determinism_mode,
""",
)

# Extend the exact production metadata key contract.
replace_once(
    "src/qsol_geo_reason/capture_common.py",
    '    "mps_macos_version", "mps_fallback_env", "mps_fast_math_env", "autocast_disabled",\n',
    '    "mps_macos_version", "mps_fallback_env", "mps_fast_math_env", "mps_prefer_metal_env", "autocast_disabled",\n',
)
replace_once(
    "src/qsol_geo_reason/capture_common.py",
    '    "deterministic_algorithms_enabled", "determinism_mode",\n',
    '    "deterministic_algorithms_enabled", "deterministic_warn_only_enabled", "determinism_mode",\n',
)

# Reject impossible or policy-weakened production observations at verification.
replace_once(
    "src/qsol_geo_reason/capture_provenance.py",
    """def _validate_required_determinism(observed: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    enabled = observed.get("deterministic_algorithms_enabled")
    if not isinstance(enabled, bool):
        raise CaptureContractError("deterministic_algorithms_enabled must be boolean")
    if request["determinism"]["mode"] == "required" and enabled is not True:
        raise CaptureContractError(
            "required determinism requires deterministic_algorithms_enabled=true"
        )
""",
    """def _validate_required_determinism(observed: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    enabled = observed.get("deterministic_algorithms_enabled")
    if not isinstance(enabled, bool):
        raise CaptureContractError("deterministic_algorithms_enabled must be boolean")
    warn_only = observed.get("deterministic_warn_only_enabled")
    if not isinstance(warn_only, bool):
        raise CaptureContractError("deterministic_warn_only_enabled must be boolean")
    if warn_only is not False:
        raise CaptureContractError(
            "canonical observation requires deterministic_warn_only_enabled=false"
        )
    if request["determinism"]["mode"] == "required" and enabled is not True:
        raise CaptureContractError(
            "required determinism requires deterministic_algorithms_enabled=true"
        )
""",
)
replace_once(
    "src/qsol_geo_reason/capture_provenance.py",
    """    native_tokenizers = observed.get("tokenizers_native_backend_active")
    if not isinstance(native_tokenizers, bool):
        raise CaptureContractError("tokenizers_native_backend_active must be boolean")
    if native_tokenizers:
        if not isinstance(observed.get("tokenizers_version"), str) or not observed["tokenizers_version"].strip():
            raise CaptureContractError("active native tokenizer requires tokenizers_version")
        _validate_python_package_provenance(
            observed,
            count_field="tokenizers_package_file_count",
            receipt_field="tokenizers_package_receipt_sha256",
            where="Tokenizers",
        )
    elif observed.get("tokenizers_package_file_count") is not None or observed.get("tokenizers_package_receipt_sha256") is not None:
        raise CaptureContractError("inactive native tokenizer must not carry package provenance")
""",
    """    native_tokenizers = observed.get("tokenizers_native_backend_active")
    if not isinstance(native_tokenizers, bool):
        raise CaptureContractError("tokenizers_native_backend_active must be boolean")
    if native_tokenizers is not True:
        raise CaptureContractError(
            "canonical observation requires tokenizers_native_backend_active=true"
        )
    if not isinstance(observed.get("tokenizers_version"), str) or not observed["tokenizers_version"].strip():
        raise CaptureContractError("active native tokenizer requires tokenizers_version")
    _validate_python_package_provenance(
        observed,
        count_field="tokenizers_package_file_count",
        receipt_field="tokenizers_package_receipt_sha256",
        where="Tokenizers",
    )
""",
)
replace_once(
    "src/qsol_geo_reason/capture_provenance.py",
    '        "mps_fallback_env", "mps_fast_math_env",\n',
    '        "mps_fallback_env", "mps_fast_math_env", "mps_prefer_metal_env",\n',
)
replace_once(
    "src/qsol_geo_reason/capture_provenance.py",
    """        if _env_flag_enabled(observed.get("mps_fast_math_env")):
            raise CaptureContractError("canonical MPS provenance forbids fast-math enablement")
""",
    """        if _env_flag_enabled(observed.get("mps_fast_math_env")):
            raise CaptureContractError("canonical MPS provenance forbids fast-math enablement")
        if _env_flag_enabled(observed.get("mps_prefer_metal_env")):
            raise CaptureContractError("canonical MPS provenance forbids Metal matmul preference")
""",
)

# Keep the published schema exactly aligned with the runtime verifier.
schema = "schemas/capture-run-manifest.schema.json"
replace_once(
    schema,
    '        "mps_macos_version", "mps_fallback_env", "mps_fast_math_env", "autocast_disabled",\n',
    '        "mps_macos_version", "mps_fallback_env", "mps_fast_math_env", "mps_prefer_metal_env", "autocast_disabled",\n',
)
replace_once(
    schema,
    '        "deterministic_algorithms_enabled", "determinism_mode"\n',
    '        "deterministic_algorithms_enabled", "deterministic_warn_only_enabled", "determinism_mode"\n',
)
replace_once(
    schema,
    '        "tokenizers_native_backend_active": {"type": "boolean"},\n',
    '        "tokenizers_native_backend_active": {"const": true},\n',
)
replace_once(
    schema,
    '          "type": ["integer", "null"], "minimum": 1,\n          "description": "Count of receipt-bound files in the imported native Tokenizers package when a fast tokenizer backend is active."\n',
    '          "type": "integer", "minimum": 1,\n          "description": "Count of receipt-bound files in the required native Tokenizers package used by every canonical production observation."\n',
)
replace_once(
    schema,
    '          "anyOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}],\n          "description": "Content receipt for the active native Tokenizers implementation, or null when the native fast-tokenizer backend is inactive. Canonical production currently rejects loaded slow tokenizers."\n',
    '          "$ref": "#/$defs/sha256",\n          "description": "Content receipt for the required native Tokenizers implementation used by every canonical production observation."\n',
)
replace_once(
    schema,
    '        "tokenizers_version": {"$ref": "#/$defs/nullableString"},\n',
    '        "tokenizers_version": {"type": "string", "minLength": 1, "pattern": "\\\\S"},\n',
)
replace_once(
    schema,
    '        "mps_fast_math_env": {"$ref": "#/$defs/nullableString"},\n        "autocast_disabled": {"const": true},\n',
    '        "mps_fast_math_env": {"$ref": "#/$defs/nullableString"},\n        "mps_prefer_metal_env": {"$ref": "#/$defs/nullableString"},\n        "autocast_disabled": {"const": true},\n',
)
replace_once(
    schema,
    '        "deterministic_algorithms_enabled": {"type": "boolean"},\n        "determinism_mode": {"enum": ["required", "best_effort"]}\n',
    '        "deterministic_algorithms_enabled": {"type": "boolean"},\n        "deterministic_warn_only_enabled": {"const": false},\n        "determinism_mode": {"enum": ["required", "best_effort"]}\n',
)
replace_once(
    schema,
    r'''              "mps_fast_math_env": {
                "anyOf": [
                  {"type": "null"},
                  {"type": "string", "pattern": "^\\s*(?:0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff])?\\s*$"}
                ]
              }
''',
    r'''              "mps_fast_math_env": {
                "anyOf": [
                  {"type": "null"},
                  {"type": "string", "pattern": "^\\s*(?:0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff])?\\s*$"}
                ]
              },
              "mps_prefer_metal_env": {
                "anyOf": [
                  {"type": "null"},
                  {"type": "string", "pattern": "^\\s*(?:0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff])?\\s*$"}
                ]
              }
''',
)

Path("tests/test_capture_codex_round47.py").write_text(
    '''"""Round 47 regressions for tokenizer and execution-policy provenance."""
from __future__ import annotations

import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_provenance as provenance
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend
from qsol_geo_reason.capture_common import CaptureContractError, _PRODUCTION_BACKEND_KEYS


class CaptureRound47RegressionTests(unittest.TestCase):
    def test_impossible_non_native_tokenizer_observation_is_rejected(self):
        observed = {
            "python_version": "3.12", "platform": "test", "torch_version": "2.0",
            "transformers_version": "5.0", "model_class": "Model",
            "tokenizer_class": "Tokenizer", "device": "cpu",
            "tokenizers_native_backend_active": False, "tokenizers_version": None,
            "tokenizers_package_file_count": None,
            "tokenizers_package_receipt_sha256": None,
        }
        request = {"capture": {"layers": [0]}, "backend": {"device": "cpu"}, "determinism": {"mode": "required"}}
        with patch.object(provenance, "_validate_torch_build_metadata"), patch.object(provenance, "_validate_python_package_provenance"):
            with self.assertRaisesRegex(CaptureContractError, "requires tokenizers_native_backend_active=true"):
                provenance._validate_production_metadata_shape(observed, request)

    def test_warn_only_state_is_observed_and_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = types.SimpleNamespace(is_deterministic_algorithms_warn_only_enabled=lambda: True)
        self.assertIs(backend._deterministic_warn_only_state(), True)
        request = {"determinism": {"mode": "required"}}
        with self.assertRaisesRegex(CaptureContractError, "warn_only_enabled=false"):
            provenance._validate_required_determinism(
                {"deterministic_algorithms_enabled": True, "deterministic_warn_only_enabled": True},
                request,
            )
        provenance._validate_required_determinism(
            {"deterministic_algorithms_enabled": True, "deterministic_warn_only_enabled": False},
            request,
        )

    def test_schema_and_exact_key_contract_bind_new_receipts(self):
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "capture-run-manifest.schema.json"
        production = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["backendObservedProduction"]
        required = set(production["required"])
        for field in ("mps_prefer_metal_env", "deterministic_warn_only_enabled"):
            self.assertIn(field, required)
            self.assertIn(field, _PRODUCTION_BACKEND_KEYS)
        self.assertEqual(production["properties"]["tokenizers_native_backend_active"], {"const": True})
        self.assertEqual(production["properties"]["deterministic_warn_only_enabled"], {"const": False})
        self.assertEqual(production["properties"]["tokenizers_package_file_count"]["type"], "integer")
        self.assertEqual(production["properties"]["tokenizers_package_receipt_sha256"]["$ref"], "#/$defs/sha256")

    def test_mps_metal_preference_is_rejected_by_verifier_shape(self):
        observed = {key: None for key in _PRODUCTION_BACKEND_KEYS}
        observed.update({
            "python_version": "3.12", "platform": "test", "torch_version": "2.0",
            "transformers_version": "5.0", "model_class": "Model", "tokenizer_class": "Tokenizer",
            "device": "mps", "tokenizers_native_backend_active": True,
            "tokenizers_version": "0.22", "tokenizers_package_file_count": 1,
            "tokenizers_package_receipt_sha256": "a" * 64,
            "safetensors_deserializer_active": True, "safetensors_package_file_count": 1,
            "safetensors_package_receipt_sha256": "b" * 64,
            "attention_implementation": "eager", "torch_num_threads": 1,
            "torch_num_interop_threads": 1, "mps_device_active": True,
            "mps_built": True, "mps_available": True, "autocast_disabled": True,
            "hidden_state_block_path": "layers", "hidden_state_count": 2,
            "observed_hidden_state_dtypes": {"0": ["float32"]},
            "mps_prefer_metal_env": "1",
        })
        request = {"capture": {"layers": [0]}, "backend": {"device": "mps"}, "determinism": {"mode": "required"}}
        with patch.object(provenance, "_validate_torch_build_metadata"), patch.object(provenance, "_validate_python_package_provenance"), patch.object(provenance, "_validate_dispatch_metadata"):
            with self.assertRaisesRegex(CaptureContractError, "Metal matmul preference"):
                provenance._validate_production_metadata_shape(observed, request)


if __name__ == "__main__":
    unittest.main()
''',
    encoding="utf-8",
)
