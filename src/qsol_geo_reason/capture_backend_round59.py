"""Round-59 compatibility surface for the sealed Round-58 capture boundary."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .capture_backend_round58 import HuggingFacePyTorchBackend
from . import capture_backend_round56 as _round56
from . import capture_backend_round58 as _round58
from . import provenance as _provenance


def _git_run_compat_round59(
    root: Path, *args: str, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Keep public provenance testable without weakening canonical capture.

    Canonical OBSERVATION source identity is already bound inside Round 58 to a
    private cloned provenance call graph whose Git runner is closure-sealed.  The
    public provenance module remains a general/testable API, so its compatibility
    runner deliberately resolves ``subprocess.run`` dynamically while retaining the
    same absolute-Git and dynamic-loader environment hardening.
    """
    git = _round56._trusted_git_executable()
    completed = subprocess.run(
        [str(git), "-C", str(root), *args],
        env=_round58._git_child_environment_round58(),
        **kwargs,
    )
    _round56._trusted_git_executable()
    return completed


# This mutable public slot is not a canonical trust root. capture_execute's
# resolve_implementation_revision global points at Round 58's private cloned call
# graph and therefore never resolves this binding during an OBSERVATION.
_provenance._git_run = _git_run_compat_round59

__all__ = ["HuggingFacePyTorchBackend"]
