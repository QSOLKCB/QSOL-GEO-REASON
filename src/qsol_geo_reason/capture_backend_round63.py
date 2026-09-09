"""Round-63 closure sealing for the exported Round-58 Git runner."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from . import capture_backend_round56 as _round56
from . import capture_backend_round58 as _round58
from . import provenance as _provenance


def _make_round63_git_runner():
    trusted_git = _round56._trusted_git_executable
    trusted_run = subprocess.run
    source_environment = os.environ
    devnull = os.devnull
    loader_prefixes = tuple(_round58._DYNAMIC_LOADER_ENV_PREFIXES)
    loader_names = frozenset(_round58._DYNAMIC_LOADER_ENV_NAMES)

    def child_environment() -> dict[str, str]:
        environment: dict[str, str] = {}
        for key, value in source_environment.items():
            upper = key.upper()
            if upper.startswith("GIT_"):
                continue
            if upper.startswith(loader_prefixes):
                continue
            if upper in loader_names:
                continue
            environment[key] = value
        environment.update(
            {
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        return environment

    sealed_child_environment = child_environment

    def run(
        root: Path, *args: str, **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        git = trusted_git()
        completed = trusted_run(
            [str(git), "-C", str(root), *args],
            env=sealed_child_environment(),
            **kwargs,
        )
        trusted_git()
        return completed

    return run


_git_run_round63 = _make_round63_git_runner()
del _make_round63_git_runner

# Replace the caller-visible Round-58 runner and the ordinary provenance compatibility
# binding. Round 60 independently rebuilds the canonical source-identity graph with
# its own closure-sealed runner, so neither path consults the writable Round-58
# environment-builder global after this point.
_round58._git_run_round58 = _git_run_round63
_provenance._git_run = _git_run_round63

__all__: list[str] = []
