"""Canonical local-model hidden-state capture API for QSOL-GEO-REASON Phase 2A."""

from .capture_common import (CAPTURE_PROTOCOL_ID, CAPTURE_SCHEMA_VERSION, CaptureBackend, CaptureBackendUnavailable, CaptureContractError, _pool_span, _require_hf_repo_id)
from .capture_validation import _quantization_reasons, _validate_loading_info, validate_capture_request
from .capture_provenance import _resolve_hidden_state_layout, _snapshot_file_hashes
from .capture_backend import HuggingFacePyTorchBackend
from .capture_execute import execute_capture
from .capture_verify import verify_capture_bundle
from .capture_publish import write_capture_bundle

__all__ = ["CAPTURE_PROTOCOL_ID", "CAPTURE_SCHEMA_VERSION", "CaptureBackend", "CaptureBackendUnavailable", "CaptureContractError", "HuggingFacePyTorchBackend", "execute_capture", "validate_capture_request", "verify_capture_bundle", "write_capture_bundle"]
