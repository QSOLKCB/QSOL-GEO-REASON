"""No-site isolated worker for trusted canonical-Hub preparation."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from typing import Any

from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_package import _python_package_provenance
from .capture_reference_environment import HUB_TRANSPORT_PACKAGE_IMPORTS


def _preimport_package_provenance(import_name: str, where: str) -> dict[str, Any]:
    if import_name in sys.modules:
        raise CaptureContractError(
            f"Hub preparation requires {import_name} to be absent before its authenticated import"
        )
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise CaptureBackendUnavailable(
            f"Hub preparation requires the locked {import_name} distribution"
        ) from exc
    if spec is None or not isinstance(spec.origin, str) or not spec.origin.strip():
        raise CaptureBackendUnavailable(
            f"Hub preparation cannot locate the locked {import_name} distribution"
        )
    probe = types.SimpleNamespace(__file__=spec.origin)
    return _python_package_provenance(probe, where)


def _preimport_hub_package_provenance() -> dict[str, Any]:
    return _preimport_package_provenance(
        "huggingface_hub",
        "Hugging Face Hub preparation",
    )


def _preimport_transport_package_provenance() -> dict[str, dict[str, Any]]:
    return {
        canonical: _preimport_package_provenance(
            import_name,
            f"Hugging Face Hub transport dependency {canonical}",
        )
        for canonical, import_name in sorted(HUB_TRANSPORT_PACKAGE_IMPORTS.items())
    }


def _loaded_transport_package_provenance() -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for canonical, import_name in sorted(HUB_TRANSPORT_PACKAGE_IMPORTS.items()):
        try:
            module = importlib.import_module(import_name)
        except ImportError as exc:
            raise CaptureBackendUnavailable(
                f"Hub preparation requires the locked {import_name} distribution"
            ) from exc
        observed[canonical] = _python_package_provenance(
            module,
            f"Hugging Face Hub transport dependency {canonical}",
        )
    return observed


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
    transport_before = _preimport_transport_package_provenance()
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
    transport_after_import = _loaded_transport_package_provenance()
    if transport_after_import != transport_before:
        raise CaptureContractError(
            "Hugging Face Hub transport package content changed while establishing the no-site preparation boundary"
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
    transport_after_work = _loaded_transport_package_provenance()
    if transport_after_work != transport_before:
        raise CaptureContractError(
            "Hugging Face Hub transport package content changed during trusted online preparation"
        )

    evidence = {
        **receipts,
        "huggingface_hub_package_file_count": before["file_count"],
        "huggingface_hub_package_receipt_sha256": before["receipt_sha256"],
        "hub_transport_package_provenance": transport_before,
    }
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureContractError, CaptureBackendUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
