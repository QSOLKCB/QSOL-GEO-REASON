from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/qsol_geo_reason/capture_cuda_runtime.py"
text = TARGET.read_text(encoding="utf-8")

old_import = "import stat\nimport sys\nfrom dataclasses import dataclass\nfrom pathlib import Path\n"
new_import = "import stat\nimport sys\nfrom pathlib import Path\n"
if text.count(old_import) != 1:
    raise RuntimeError("expected one generated dataclass import")
text = text.replace(old_import, new_import, 1)

old_class = '''@dataclass(frozen=True)
class _MappedLibrary:
    """A loaded library plus the stable object used to read its mapped bytes."""

    path: Path
    content_path: Path
    device: int | None = None
    inode: int | None = None
'''
new_class = '''class _MappedLibrary:
    """A loaded library plus the stable object used to read its mapped bytes."""

    __slots__ = ("path", "content_path", "device", "inode")

    def __init__(
        self,
        path: Path,
        content_path: Path,
        device: int | None = None,
        inode: int | None = None,
    ) -> None:
        # Keep this constructor source-backed. Canonical checkout provenance rejects
        # exec-generated callables, including dataclass-generated __init__ methods.
        self.path = Path(path)
        self.content_path = Path(content_path)
        self.device = device
        self.inode = inode
'''
if text.count(old_class) != 1:
    raise RuntimeError("expected one generated _MappedLibrary dataclass")
text = text.replace(old_class, new_class, 1)
TARGET.write_text(text, encoding="utf-8")
