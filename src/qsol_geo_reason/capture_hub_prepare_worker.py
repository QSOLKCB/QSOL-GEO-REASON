"""No-site isolated worker for trusted canonical-Hub preparation."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from typing import Any

from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_package import _python_package_provenance


def _preimport_hub_package_provenance() -> dict[str, Any]:
    if "huggingface_hub" in sys.modules:
        raise CaptureContractError(
            "Hub preparation requires huggingface_hub to be absent before its authenticated import"
        )
    try:
        spec = importlib.util.find_spec("huggingface_hub")
    except (ImportError, AttributeError, ValueError) as exc:
        raise CaptureBackendUnavailable(
            "Hub preparation requires the locked huggingface-hub distribution"
        ) from exc
    if spec is None or not isinstance(spec.origin, str) or not spec.origin.strip():
        raise CaptureBackendUnavailable(
            "Hub preparation cannot locate the locked huggingface-hub distribution"
        )
    probe = types.SimpleNamespace(__file__=spec.origin)
    return _python_package_provenance(probe, "Hugging Face Hub preparation")


def main() -> int:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise CaptureContractError(
            "Hub preparation worker requires CPython isolated mode with site disabled (-I -S)"
        )
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise CaptureContractError(
            "Hub preparation worker inherited a Python startup customization module"
        )

    before = _preimport_hub_package_provenance()
    try:
        import huggingface_hub
    except ImportError as exc:
        raise CaptureBackendUnavailable(
            "Hub preparation requires the locked huggingface-hub distribution"
        ) from exc
    after_import = _python_package_provenance(
        huggingface_hub,
        "Hugging Face Hub preparation",
    )
    if after_import != before:
        raise CaptureContractError(
            "Hugging Face Hub package changed while establishing the no-site preparation boundary"
        )

    from .capture_hub_tree import prepare_tree_receipts

    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise CaptureContractError("Hub preparation worker received malformed request JSON") from exc
    if not isinstance(request, dict):
        raise CaptureContractError("Hub preparation worker request must be a JSON object")

    receipts = prepare_tree_receipts(request)
    after_work = _python_package_provenance(
        huggingface_hub,
        "Hugging Face Hub preparation",
    )
    if after_work != before:
        raise CaptureContractError(
            "Hugging Face Hub package changed during trusted online preparation"
        )

    evidence = {
        **receipts,
        "huggingface_hub_package_file_count": before["file_count"],
        "huggingface_hub_package_receipt_sha256": before["receipt_sha256"],
    }
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureContractError, CaptureBackendUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
