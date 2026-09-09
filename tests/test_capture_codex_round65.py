"""Round 65/66 regressions for the latest canonical OBSERVATION trust findings."""
from __future__ import annotations

import inspect
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from qsol_geo_reason import capture_backend_core as core
from qsol_geo_reason import capture_backend_final as final_backend
from qsol_geo_reason import capture_backend_round65 as round65
from qsol_geo_reason import capture_backend_round66 as round66
from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_execute import resolve_implementation_revision


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_runner(root: Path, *args: str, **kwargs):
    return subprocess.run(["git", "-C", str(root), *args], **kwargs)


class CaptureRound65RegressionTests(unittest.TestCase):
    def test_snapshot_verifiers_are_reasserted_from_closure_bound_trust(self):
        trusted_core = core._snapshot_file_hashes
        trusted_final = final_backend._snapshot_file_hashes
        self.assertIs(trusted_core, trusted_final)

        def forged_snapshot(*_args, **_kwargs):
            return {"forged.safetensors": "0" * 64}

        try:
            core._snapshot_file_hashes = forged_snapshot
            final_backend._snapshot_file_hashes = forged_snapshot
            round65._reassert_snapshot_verifier_bindings_round65()
            self.assertIs(core._snapshot_file_hashes, trusted_core)
            self.assertIs(final_backend._snapshot_file_hashes, trusted_final)
        finally:
            core._snapshot_file_hashes = trusted_core
            final_backend._snapshot_file_hashes = trusted_final

        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        first = source.index("reassert_snapshot_verifiers()")
        inherited = source.index("original_init(self, request)")
        second = source.index("reassert_snapshot_verifiers()", first + 1)
        self.assertLess(first, inherited)
        self.assertLess(inherited, second)

    def test_transformers_stability_receipt_detects_byte_identical_replace_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "transformers"
            package.mkdir()
            target = package / "modeling.py"
            target.write_bytes(b"VALUE = 1\n")
            before = round65._package_stability_from_root_round65(
                package, "Transformers"
            )

            replacement = package / "replacement.tmp"
            replacement.write_bytes(b"VALUE = 1\n")
            os.replace(replacement, target)
            after = round65._package_stability_from_root_round65(
                package, "Transformers"
            )
            self.assertNotEqual(before, after)

        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertIn('preimport_stability("transformers", "Transformers")', source)
        self.assertIn("changed transiently during authenticated model loading", source)

    def test_deterministic_policy_callables_are_bound_after_construction(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        torch_module = types.ModuleType("torch")

        def setter(_enabled, warn_only=False):
            return None

        def enabled():
            return True

        def warn_only():
            return False

        torch_module.use_deterministic_algorithms = setter
        torch_module.are_deterministic_algorithms_enabled = enabled
        torch_module.is_deterministic_algorithms_warn_only_enabled = warn_only
        backend._torch = torch_module
        backend._canonical_deterministic_algorithms_enabled = True
        backend._canonical_deterministic_warn_only_enabled = False
        backend._determinism_mode = "required"
        round65._remember_round65_determinism_callables(
            backend, (setter, enabled, warn_only)
        )

        self.assertTrue(backend._deterministic_algorithms_state())
        self.assertFalse(backend._deterministic_warn_only_state())

        for attribute, operation in (
            (
                "use_deterministic_algorithms",
                backend._force_canonical_determinism_policy,
            ),
            (
                "are_deterministic_algorithms_enabled",
                backend._deterministic_algorithms_state,
            ),
            (
                "is_deterministic_algorithms_warn_only_enabled",
                backend._deterministic_warn_only_state,
            ),
        ):
            original = getattr(torch_module, attribute)
            try:
                setattr(torch_module, attribute, lambda *_args, **_kwargs: True)
                with self.assertRaisesRegex(
                    CaptureContractError, "deterministic-policy callable changed"
                ):
                    operation()
            finally:
                setattr(torch_module, attribute, original)

    def test_ignored_python_source_is_detected_even_when_git_status_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _git(repo, "init")
            _git(repo, "config", "user.email", "round66@example.invalid")
            _git(repo, "config", "user.name", "Round 66")
            package = repo / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "tracked package")

            exclude = repo / ".git" / "info" / "exclude"
            with exclude.open("a", encoding="utf-8") as handle:
                handle.write("\nsrc/qsol_geo_reason/hidden.py\n")
            (package / "hidden.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertEqual(_git(repo, "status", "--porcelain=v1", "--untracked-files=all"), "")

            ignored = round66._scan_ignored_python_sources_round66(repo, _git_runner)
            self.assertEqual(ignored, ("src/qsol_geo_reason/hidden.py",))

        resolver_source = inspect.getsource(resolve_implementation_revision)
        self.assertIn("ignored_guard()", resolver_source)
        self.assertIn("require_checkout", resolver_source)


if __name__ == "__main__":
    unittest.main()
