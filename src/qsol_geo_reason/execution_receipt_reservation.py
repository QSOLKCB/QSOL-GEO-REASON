"""Crash-aware reservation for external execution-receipt destinations."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_bytes
from .capture_common import CaptureContractError
from .capture_publish import _ensure_parent_directory_durable, _fsync_directory


@dataclass(frozen=True)
class ExecutionReceiptReservation:
    path: Path
    marker: bytes


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )


def reserve_execution_receipt_destination(path: Path) -> ExecutionReceiptReservation:
    """Reserve the exact receipt name durably before any model/backend work starts."""
    destination = Path(path)
    try:
        _ensure_parent_directory_durable(destination.parent)
    except (CaptureContractError, OSError) as exc:
        raise RuntimeError(
            f"execution receipt destination parent is unavailable: {destination.parent}"
        ) from exc

    marker = (
        "QSOL-EXECUTION-RECEIPT-RESERVATION-1 "
        f"{os.getpid()} {secrets.token_hex(16)}\n"
    ).encode("ascii")
    published = False
    try:
        with destination.open("xb") as handle:
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
        published = True
        _fsync_directory(destination.parent)
    except FileExistsError as exc:
        raise RuntimeError(
            f"execution receipt destination already exists: {destination}"
        ) from exc
    except OSError as exc:
        if published:
            try:
                destination.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"unable to reserve execution receipt destination {destination}: {exc}"
        ) from exc

    return ExecutionReceiptReservation(path=destination, marker=marker)


def commit_execution_receipt_reservation(
    reservation: ExecutionReceiptReservation,
    receipt: Mapping[str, Any],
) -> None:
    """Atomically replace one intact reservation with the canonical receipt payload."""
    destination = reservation.path
    try:
        payload = canonical_json_bytes(receipt) + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError("execution receipt is not canonical JSON") from exc

    temporary = _temporary_sibling(destination)
    replaced = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            observed_marker = destination.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"unable to verify execution receipt reservation {destination}"
            ) from exc
        if observed_marker != reservation.marker:
            raise RuntimeError(
                f"execution receipt reservation changed before publication: {destination}"
            )
        os.replace(temporary, destination)
        replaced = True
        _fsync_directory(destination.parent)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"unable to publish reserved execution receipt {destination}: {exc}"
        ) from exc
    finally:
        if not replaced:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def release_execution_receipt_reservation(
    reservation: ExecutionReceiptReservation,
) -> None:
    """Best-effort removal of an intact pre-bundle reservation after ordinary failure."""
    destination = reservation.path
    try:
        if destination.read_bytes() != reservation.marker:
            return
        destination.unlink()
        _fsync_directory(destination.parent)
    except OSError:
        pass


__all__ = [
    "ExecutionReceiptReservation",
    "commit_execution_receipt_reservation",
    "release_execution_receipt_reservation",
    "reserve_execution_receipt_destination",
]
