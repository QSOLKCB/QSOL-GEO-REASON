from __future__ import annotations

import contextlib
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from reference_environment_fixture import (
    capture_direct_package_provenance,
    capture_transitive_package_provenance,
    hub_package_provenance,
    hub_transport_package_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_capture_reference_environment.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("qsol_test_reference_lock_verifier", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load reference lock verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enter_reference_platform(
    stack: contextlib.ExitStack,
    verifier,
    *,
    python_version=(3, 11, 16),
    releaselevel="final",
    serial=0,
    transport=None,
    direct=None,
    transitive=None,
) -> None:
    stack.enter_context(mock.patch.object(verifier, "_current_python_version", return_value=python_version))
    stack.enter_context(mock.patch.object(verifier, "_current_python_releaselevel", return_value=releaselevel))
    stack.enter_context(mock.patch.object(verifier, "_current_python_serial", return_value=serial))
    stack.enter_context(mock.patch.object(verifier, "_current_python_implementation", return_value="CPython"))
    stack.enter_context(mock.patch.object(verifier, "_current_platform_system", return_value="Linux"))
    stack.enter_context(mock.patch.object(verifier, "_current_platform_machine", return_value="x86_64"))
    stack.enter_context(mock.patch.object(verifier, "_current_hub_package_provenance", return_value=hub_package_provenance()))
    stack.enter_context(mock.patch.object(
        verifier,
        "_current_hub_transport_package_provenance",
        return_value=transport if transport is not None else hub_transport_package_provenance(),
    ))
    stack.enter_context(mock.patch.object(
        verifier,
        "_current_capture_direct_package_provenance",
        return_value=direct if direct is not None else capture_direct_package_provenance(),
    ))
    stack.enter_context(mock.patch.object(
        verifier,
        "_current_capture_transitive_package_provenance",
        return_value=transitive if transitive is not None else capture_transitive_package_provenance(),
    ))


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
            "numpy": "1.26.4",
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
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=actual))
            _enter_reference_platform(stack, verifier)
            with self.assertRaisesRegex(RuntimeError, "unexpected=.*surprise-package"):
                verifier.verify_reference_environment()

    def test_verifier_rejects_non_python311_even_with_exact_distribution_lock(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier, python_version=(3, 12, 9))
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.11"):
                verifier.verify_reference_environment()

    def test_verifier_rejects_prerelease_python311_even_with_exact_lock(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier, releaselevel="candidate", serial=1)
            with self.assertRaisesRegex(RuntimeError, "requires a final CPython release"):
                verifier.verify_reference_environment()

    def test_reference_receipt_is_self_hashed_complete_and_content_binds_capture_closure(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier)
            receipt = verifier.verify_reference_environment()
        self.assertEqual(receipt["distribution_count"], 28)
        self.assertEqual(receipt["python_version"], "3.11.16")
        self.assertEqual(receipt["platform_machine"], "x86_64")
        self.assertEqual(receipt["huggingface_hub_package_file_count"], 137)
        self.assertEqual(receipt["huggingface_hub_package_receipt_sha256"], "7" * 64)
        self.assertEqual(receipt["hub_transport_package_provenance"], hub_transport_package_provenance())
        for required in ("filelock", "fsspec", "packaging", "pyyaml", "tqdm", "typing-extensions"):
            self.assertIn(required, receipt["hub_transport_package_provenance"])
        self.assertEqual(receipt["capture_direct_package_provenance"], capture_direct_package_provenance())
        self.assertEqual(
            set(receipt["capture_direct_package_provenance"]),
            {"torch", "transformers", "tokenizers", "safetensors"},
        )
        self.assertEqual(receipt["capture_transitive_package_provenance"], capture_transitive_package_provenance())
        for required in ("numpy", "regex", "pyyaml", "sympy", "typing-extensions"):
            self.assertIn(required, receipt["capture_transitive_package_provenance"])
        self.assertRegex(receipt["environment_receipt_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(receipt["distributions"]), 28)

    def test_reference_receipt_rejects_missing_transport_package_provenance(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        transport = hub_transport_package_provenance()
        transport.pop("filelock")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier, transport=transport)
            with self.assertRaisesRegex(RuntimeError, "execution package provenance keys"):
                verifier.verify_reference_environment()

    def test_reference_receipt_rejects_missing_direct_package_provenance(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        direct = capture_direct_package_provenance()
        direct.pop("torch")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier, direct=direct)
            with self.assertRaisesRegex(RuntimeError, "capture direct package provenance keys"):
                verifier.verify_reference_environment()

    def test_reference_receipt_rejects_missing_transitive_package_provenance(self) -> None:
        verifier = _load_verifier()
        locked = verifier._locked_versions()
        transitive = capture_transitive_package_provenance()
        transitive.pop("numpy")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(verifier, "_installed_runtime_versions", return_value=dict(locked)))
            _enter_reference_platform(stack, verifier, transitive=transitive)
            with self.assertRaisesRegex(RuntimeError, "capture transitive package provenance keys"):
                verifier.verify_reference_environment()


if __name__ == "__main__":
    unittest.main()
