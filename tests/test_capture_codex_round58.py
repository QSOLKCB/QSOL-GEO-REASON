from __future__ import annotations

import subprocess
import unittest

from qsol_geo_reason import capture
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import capture_backend_round58 as round58
from qsol_geo_reason import capture_backend_round60 as round60
from qsol_geo_reason import capture_execute
from qsol_geo_reason import provenance


class Round60ProvenanceWrapperTests(unittest.TestCase):
    def test_public_capture_binds_corrected_round60_resolver(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(
            capture_execute.resolve_implementation_revision,
            round60._resolve_implementation_revision_round60,
        )
        private_globals = round60._resolve_implementation_revision_round60.__globals__
        self.assertIs(private_globals["_git_run"], round58._git_run_round58)
        self.assertIs(
            private_globals["git_source_revision"],
            round60._git_source_revision_round60,
        )

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


if __name__ == "__main__":
    unittest.main()
