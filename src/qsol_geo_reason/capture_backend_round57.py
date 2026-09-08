"""Compatibility shim for the superseding Round-58 trust-boundary hardening."""

from .capture_backend_round58 import HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]
