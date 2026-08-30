"""Canonical JSON and hash helpers.

The simulation evidence contract uses canonical JSON with sorted keys, compact
separators, UTF-8, and disallowed NaN/Infinity so hashes are content-bound.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
