"""Round-62 verifier binding for mapped MPS runtime provenance."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .capture_common import CaptureContractError
from . import capture_runtime as _runtime
from . import capture_provenance as _provenance


_MPS_RUNTIME_CONFIG_PREFIX = "QSOL_GEO_MPS_RUNTIME="
_CPU_RUNTIME_CONFIG_PREFIX = "QSOL_GEO_CPU_RUNTIME="
_MPS_RUNTIME_PROVENANCE_KEYS = frozenset(
    {
        "mps_runtime_library_file_count",
        "mps_runtime_library_receipt_sha256",
    }
)
_ORIGINAL_VALIDATE_TORCH_BUILD_METADATA = _runtime._validate_torch_build_metadata


def _mps_runtime_library_provenance_from_build_config_round62(
    config: str,
) -> dict[str, int | str] | None:
    lines = config.rstrip("\n").split("\n")
    records = [line for line in lines if line.startswith(_MPS_RUNTIME_CONFIG_PREFIX)]
    if not records:
        return None
    if len(records) != 1:
        raise CaptureContractError(
            "torch_build_config must contain exactly one MPS runtime provenance record"
        )
    record = records[0]
    index = lines.index(record)
    if (
        index != len(lines) - 2
        or not lines[-1].startswith(_CPU_RUNTIME_CONFIG_PREFIX)
    ):
        raise CaptureContractError(
            "MPS runtime provenance record must immediately precede the final CPU pooling runtime record"
        )
    try:
        payload = json.loads(record[len(_MPS_RUNTIME_CONFIG_PREFIX) :])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureContractError("MPS runtime provenance record is malformed") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"loaded_mps_runtime_libraries"}
        or not isinstance(payload["loaded_mps_runtime_libraries"], dict)
    ):
        raise CaptureContractError("MPS runtime provenance record is malformed")
    provenance = payload["loaded_mps_runtime_libraries"]
    if set(provenance) != _MPS_RUNTIME_PROVENANCE_KEYS:
        raise CaptureContractError("MPS runtime provenance record is malformed")
    count = provenance["mps_runtime_library_file_count"]
    receipt = provenance["mps_runtime_library_receipt_sha256"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(
            "MPS runtime library file count must be a positive integer"
        )
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(character not in "0123456789abcdef" for character in receipt)
    ):
        raise CaptureContractError(
            "MPS runtime library receipt must be a lowercase SHA-256 digest"
        )
    return {
        "mps_runtime_library_file_count": count,
        "mps_runtime_library_receipt_sha256": receipt,
    }


def _validate_torch_build_metadata_round62(observed: Mapping[str, Any]) -> None:
    _ORIGINAL_VALIDATE_TORCH_BUILD_METADATA(observed)
    config = observed.get("torch_build_config")
    if not isinstance(config, str):
        # The original validator provides the canonical diagnostic.
        return
    mps_runtime = _mps_runtime_library_provenance_from_build_config_round62(config)
    device = observed.get("device")
    if device == "mps":
        if mps_runtime is None:
            raise CaptureContractError(
                "MPS torch_build_config is missing the authenticated mapped MPS runtime receipt"
            )
    elif mps_runtime is not None:
        raise CaptureContractError(
            "MPS runtime library provenance must be absent outside MPS"
        )


# capture_provenance imported the validator by value, so harden both bindings before
# capture_execute freezes the canonical verifier surface.
_runtime._validate_torch_build_metadata = _validate_torch_build_metadata_round62
_provenance._validate_torch_build_metadata = _validate_torch_build_metadata_round62

__all__: list[str] = []
