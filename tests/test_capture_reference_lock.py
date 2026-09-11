from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_capture_reference_environment.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("qsol_test_reference_lock_verifier", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load reference lock verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_platform_patches(verifier):
    return (
        mock.patch.object(verifier, "_current_python_version", return_value=(3, 11, 16)),
        mock.patch.object(verifier, "_current_python_implementation", return_value="CPython"),
        mock.patch.object(verifier, "_current_platform_system", return_value="Linux"),
        mock.patch.object(verifier, "_current_platform_machine", return_value="x86_64"),
    )


class CaptureReferenceLockTests(unittest.TestCase):
    def test_complete_lock_contains_resolved_transitive_runtime(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        self.assertEqual(len(locked), 28)
        for name, version in {
            "torch": "2.2.2+cpu",
            "filelock": "3.32.3",
            "fsspec": "2026.7.0",
            "jinja2": "3.1.6",
            "sympy": "1.14.0",
            "networkx": "3.6.1",
            "regex": "2026.9.10",
            "pyyaml": "6.0.3",
            "requests": "2.34.2",
            "certifi": "2026.7.22",
            "jsonschema-specifications": "2025.9.1",
            "referencing": "0.37.0",
            "rpds-py": "2026.6.3",
        }.items():
            self.assertEqual(locked[name][1], version)

    def test_verifier_rejects_unexpected_and_mismatched_runtime(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        actual = dict(locked)
        actual["requests"] = ("requests", "0.0.0")
        actual["surprise-package"] = ("surprise-package", "1.0.0")
        patches = _reference_platform_patches(verifier)
        with (
            mock.patch.object(
                verifier,
                "_installed_runtime_versions",
                return_value=actual,
            ),
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected=.*surprise-package"):
                verifier.verify_reference_environment()

    def test_verifier_rejects_non_python311_even_with_exact_distribution_lock(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        with (
            mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)),
            mock.patch.object(verifier, "_current_python_version", return_value=(3, 12, 9)),
            mock.patch.object(verifier, "_current_python_implementation", return_value="CPython"),
            mock.patch.object(verifier, "_current_platform_system", return_value="Linux"),
            mock.patch.object(verifier, "_current_platform_machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.11"):
                verifier.verify_reference_environment()

    def test_reference_receipt_is_self_hashed_and_complete(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        patches = _reference_platform_patches(verifier)
        with (
            mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)),
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            receipt = verifier.verify_reference_environment()
        self.assertEqual(receipt["distribution_count"], 28)
        self.assertEqual(receipt["python_version"], "3.11.16")
        self.assertEqual(receipt["platform_machine"], "x86_64")
        self.assertRegex(receipt["environment_receipt_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(receipt["distributions"]), 28)


if __name__ == "__main__":
    unittest.main()
