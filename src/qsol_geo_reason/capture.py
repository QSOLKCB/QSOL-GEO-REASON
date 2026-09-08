"""Canonical local-model hidden-state capture API for QSOL-GEO-REASON Phase 2A."""

from .capture_common import (CAPTURE_PROTOCOL_ID, CAPTURE_SCHEMA_VERSION, CaptureBackend, CaptureBackendUnavailable, CaptureContractError, _pool_span, _require_hf_repo_id)
from .capture_validation import _quantization_reasons, _validate_loading_info, validate_capture_request
from .capture_provenance import _resolve_hidden_state_layout
from .capture_snapshot import _snapshot_file_hashes
from .capture_backend_round45 import HuggingFacePyTorchBackend as _Round45HuggingFacePyTorchBackend
from .capture_backend_round46 import HuggingFacePyTorchBackend

# Round 46 must retain its runtime try/finally so private authenticated loader
# redirects are restored on every construction exit, including BaseException paths.
# The long-standing source-invariant regression suite intentionally inspects the
# inherited production constructor, however. Mark the narrow cleanup wrapper as an
# introspection wrapper without changing which callable Python actually executes.
HuggingFacePyTorchBackend.__init__.__wrapped__ = _Round45HuggingFacePyTorchBackend.__init__

from .capture_execute import execute_capture
from .capture_verify import verify_capture_bundle
from .capture_publish import write_capture_bundle

__all__ = ["CAPTURE_PROTOCOL_ID", "CAPTURE_SCHEMA_VERSION", "CaptureBackend", "CaptureBackendUnavailable", "CaptureContractError", "HuggingFacePyTorchBackend", "execute_capture", "validate_capture_request", "verify_capture_bundle", "write_capture_bundle"]