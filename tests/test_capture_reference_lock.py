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
        with mock.patch.object(
            verifier,
            "_installed_runtime_versions",
            return_value=actual,
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected=.*surprise-package"):
                verifier.verify_reference_environment()


if __name__ == "__main__":
    unittest.main()
