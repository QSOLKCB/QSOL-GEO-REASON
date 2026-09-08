"""Content receipts for external CPU math/runtime libraries used by capture."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_cuda_runtime import (
    _darwin_loaded_library_paths,
    _linux_loaded_library_paths,
    _sha256_file,
    _windows_loaded_library_paths,
)


_LINUX_CPU_LIBRARY = re.compile(
    r"^(?:libmkl[^/]*|libdnnl[^/]*|libmkldnn[^/]*|libiomp5[^/]*|libgomp[^/]*|libomp[^/]*|libopenblas[^/]*|libblas[^/]*|libcblas[^/]*|liblapack[^/]*)[.]so(?:[.].*)?$",
    re.IGNORECASE,
)
_WINDOWS_CPU_LIBRARY = re.compile(
    r"^(?:mkl[^/]*|dnnl[^/]*|mkldnn[^/]*|libiomp5(?:md)?|libgomp[^/]*|libomp[^/]*|openblas[^/]*|libopenblas[^/]*|blas[^/]*|lapack[^/]*|vcomp[0-9]+)[.]dll$",
    re.IGNORECASE,
)
_DARWIN_CPU_LIBRARY = re.compile(
    r"^(?:libmkl[^/]*|libdnnl[^/]*|libmkldnn[^/]*|libiomp5[^/]*|libomp[^/]*|libopenblas[^/]*|libblas[^/]*|liblapack[^/]*)[.]dylib$|^(?:Accelerate|vecLib)$",
    re.IGNORECASE,
)


def _is_cpu_runtime_library(path: str) -> bool:
    name = Path(path).name
    return bool(
        _LINUX_CPU_LIBRARY.fullmatch(name)
        or _WINDOWS_CPU_LIBRARY.fullmatch(name)
        or _DARWIN_CPU_LIBRARY.fullmatch(name)
    )


def _loaded_cpu_library_paths() -> list[Path]:
    if sys.platform.startswith("linux"):
        return _linux_loaded_library_paths(_is_cpu_runtime_library)
    if os.name == "nt":
        return _windows_loaded_library_paths(_is_cpu_runtime_library)
    if sys.platform == "darwin":
        return _darwin_loaded_library_paths(_is_cpu_runtime_library)
    raise CaptureContractError(
        "canonical CPU observation cannot enumerate loaded CPU math/runtime shared objects on this platform"
    )


def _cpu_runtime_library_receipt(paths: Iterable[Path]) -> tuple[int, str]:
    """Hash mapped external CPU math/runtime libraries by basename and content."""
    by_name: dict[str, str] = {}
    for raw_path in paths:
        try:
            path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise CaptureContractError(
                f"loaded CPU runtime library path is unavailable: {raw_path}"
            ) from exc
        if not path.is_file() or not _is_cpu_runtime_library(str(path)):
            continue
        name = path.name
        digest = _sha256_file(path)
        prior = by_name.get(name)
        if prior is not None and prior != digest:
            raise CaptureContractError(
                f"multiple loaded CPU runtime libraries share basename {name!r} with different content"
            )
        by_name[name] = digest
    # A statically linked PyTorch build can legitimately have no external CPU
    # math library in this set. The empty-set receipt is still an explicit,
    # canonical statement that enumeration succeeded and found none.
    return len(by_name), sha256_json(dict(sorted(by_name.items())))


def loaded_cpu_runtime_library_provenance() -> dict[str, int | str]:
    count, receipt = _cpu_runtime_library_receipt(_loaded_cpu_library_paths())
    return {
        "cpu_runtime_library_file_count": count,
        "cpu_runtime_library_receipt_sha256": receipt,
    }


__all__ = ["loaded_cpu_runtime_library_provenance"]
