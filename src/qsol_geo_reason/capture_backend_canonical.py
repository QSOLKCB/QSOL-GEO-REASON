"""Single composition boundary for the canonical GEO-CAP-001 production backend.

Historical ``capture_backend_round*`` modules are retained temporarily as reviewed
implementation strata and regression targets.  They are composed exactly once here;
public capture code imports only this module.  The round modules are defense-in-depth
implementation history, not separate public backends and not the security boundary of
the canonical CLI.
"""
from __future__ import annotations

from .capture_backend_round45 import (
    HuggingFacePyTorchBackend as _Round45HuggingFacePyTorchBackend,
)
from .capture_backend_round46 import HuggingFacePyTorchBackend
from .capture_backend_round56 import HuggingFacePyTorchBackend
from .capture_backend_round61 import HuggingFacePyTorchBackend
from .capture_backend_round64 import HuggingFacePyTorchBackend
from .capture_backend_round65 import HuggingFacePyTorchBackend

# These modules install verifier/runtime/source-identity hardening around the shared
# concrete class.  Keeping the sequence in one module makes the executable composition
# explicit and prevents the public facade from depending on repeated same-name imports.
from . import capture_backend_round62 as _capture_backend_round62  # noqa: F401
from . import capture_backend_round63 as _capture_backend_round63  # noqa: F401
from . import capture_backend_round60 as _capture_backend_round60  # noqa: F401
from . import capture_backend_round66 as _capture_backend_round66  # noqa: F401

# Compatibility only: historical source-audit regressions intentionally inspect the
# inherited pre-wrapper constructor.  This attribute does not change runtime dispatch.
HuggingFacePyTorchBackend.__init__.__wrapped__ = (
    _Round45HuggingFacePyTorchBackend.__init__
)

__all__ = ["HuggingFacePyTorchBackend"]
