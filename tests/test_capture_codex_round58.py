from __future__ import annotations

import subprocess
import unittest

from qsol_geo_reason import capture
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import capture_backend_round60 as round60
from qsol_geo_reason import capture_backend_round66 as round66
from qsol_geo_reason import capture_execute
from qsol_geo_reason import provenance


class Round60ProvenanceWrapperTests(unittest.TestCase):
    def test_public_capture_binds_corrected_round60_resolver(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(
            capture_execute.resolve_implementation_revision,
            round66._resolve_implementation_revision_round66,
        )
        closure_values = tuple(
            cell.cell_contents
            for cell in capture_execute.resolve_implementation_revision.__closure__ or ()
        )
        self.assertIn(round60._resolve_implementation_revision_round60, closure_values)

        public_globals = round60._resolve_implementation_revision_round60.__globals__
        self.assertNotIn("_git_run", public_globals)
        self.assertNotIn("git_source_revision", public_globals)
        self.assertIsNot(public_globals, provenance.__dict__)

    def test_round44_wrapper_semantics_survive_sealed_graph_reconstruction(self):
        observed = round60._git_source_revision_round60(
            require_clean=False,
            reject_importable_bytecode=False,
        )
        self.assertIsInstance(observed, str)
        self.assertRegex(observed, r"^[0-9a-f]{40}$")

    def test_public_git_runner_rebind_cannot_forge_round60_source_identity(self):
        original = provenance._git_run

        def forged_git(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="f" * 40 + "\n", stderr="")

        try:
            provenance._git_run = forged_git
            observed = round60._git_source_revision_round60(
                require_clean=False,
                reject_importable_bytecode=False,
            )
            self.assertIsInstance(observed, str)
            self.assertRegex(observed, r"^[0-9a-f]{40}$")
            self.assertNotEqual(observed, "f" * 40)
        finally:
            provenance._git_run = original

    def test_planted_public_globals_cannot_rebind_round60_dependencies(self):
        resolver = round60._resolve_implementation_revision_round60
        public_globals = resolver.__globals__
        missing = object()
        old_runner = public_globals.get("_git_run", missing)
        old_source = public_globals.get("git_source_revision", missing)

        def forged_git(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="f" * 40 + "\n", stderr="")

        public_globals["_git_run"] = forged_git
        public_globals["git_source_revision"] = lambda **_kwargs: "f" * 40
        try:
            observed = round60._git_source_revision_round60(
                require_clean=False,
                reject_importable_bytecode=False,
            )
            self.assertRegex(observed or "", r"^[0-9a-f]{40}$")
            self.assertNotEqual(observed, "f" * 40)
        finally:
            if old_runner is missing:
                public_globals.pop("_git_run", None)
            else:
                public_globals["_git_run"] = old_runner
            if old_source is missing:
                public_globals.pop("git_source_revision", None)
            else:
                public_globals["git_source_revision"] = old_source


if __name__ == "__main__":
    unittest.main()
