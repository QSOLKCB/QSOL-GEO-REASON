"""No-site isolated worker for trusted canonical-Hub preparation."""
from __future__ import annotations

import importlib
import json
import sys
from typing import Any

from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_package import _distribution_package_provenance
from .capture_reference_environment import (
    HUB_EXECUTION_PACKAGE_IMPORTS,
    HUB_TRANSPORT_PACKAGE_IMPORTS,
)


def _preimport_package_provenance(
    distribution_name: str,
    import_name: str,
    where: str,
) -> dict[str, Any]:
    if import_name in sys.modules:
        raise CaptureContractError(
            f"Hub preparation requires {import_name} to be absent before its authenticated import"
        )
    return _distribution_package_provenance(
        distribution_name,
        import_name,
        where,
    )


def _preimport_hub_package_provenance() -> dict[str, Any]:
    return _preimport_package_provenance(
        "huggingface-hub",
        "huggingface_hub",
        "Hugging Face Hub preparation",
    )


def _preimport_execution_package_provenance() -> dict[str, dict[str, Any]]:
    """Measure every external dependency Hub preparation may execute before import."""
    return {
        canonical: _preimport_package_provenance(
            canonical,
            import_name,
            f"Hugging Face Hub execution dependency {canonical}",
        )
        for canonical, import_name in sorted(HUB_EXECUTION_PACKAGE_IMPORTS.items())
    }


def _loaded_execution_package_provenance() -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for canonical, import_name in sorted(HUB_EXECUTION_PACKAGE_IMPORTS.items()):
        try:
            module = importlib.import_module(import_name)
        except ImportError as exc:
            raise CaptureBackendUnavailable(
                f"Hub preparation requires the locked {import_name} distribution"
            ) from exc
        observed[canonical] = _distribution_package_provenance(
            canonical,
            import_name,
            f"Hugging Face Hub execution dependency {canonical}",
            module=module,
        )
    return observed


def _transport_subset(
    provenance: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        canonical: provenance[canonical]
        for canonical in sorted(HUB_TRANSPORT_PACKAGE_IMPORTS)
    }


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
    execution_before = _preimport_execution_package_provenance()
    transport_before = _transport_subset(execution_before)
    try:
        import huggingface_hub
    except ImportError as exc:
        raise CaptureBackendUnavailable(
            "Hub preparation requires the locked huggingface-hub distribution"
        ) from exc
    after_import = _distribution_package_provenance(
        "huggingface-hub",
        "huggingface_hub",
        "Hugging Face Hub preparation",
        module=huggingface_hub,
    )
    if after_import != before:
        raise CaptureContractError(
            "Hugging Face Hub distribution-owned runtime changed while establishing the no-site preparation boundary"
        )
    execution_after_import = _loaded_execution_package_provenance()
    if execution_after_import != execution_before:
        raise CaptureContractError(
            "Hugging Face Hub execution dependency content changed while establishing the no-site preparation boundary"
        )

    from .capture_hub_tree import prepare_tree_receipts

    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise CaptureContractError("Hub preparation worker received malformed request JSON") from exc
    if not isinstance(request, dict):
        raise CaptureContractError("Hub preparation worker request must be a JSON object")

    receipts = prepare_tree_receipts(request)
    after_work = _distribution_package_provenance(
        "huggingface-hub",
        "huggingface_hub",
        "Hugging Face Hub preparation",
        module=huggingface_hub,
    )
    if after_work != before:
        raise CaptureContractError(
            "Hugging Face Hub distribution-owned runtime changed during trusted online preparation"
        )
    execution_after_work = _loaded_execution_package_provenance()
    if execution_after_work != execution_before:
        raise CaptureContractError(
            "Hugging Face Hub execution dependency content changed during trusted online preparation"
        )

    evidence = {
        **receipts,
        "huggingface_hub_package_file_count": before["file_count"],
        "huggingface_hub_package_receipt_sha256": before["receipt_sha256"],
        "hub_transport_package_provenance": transport_before,
        "hub_execution_package_provenance": execution_before,
    }
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureContractError, CaptureBackendUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)