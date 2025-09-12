from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

DICOM_PREFIXES = (b"DICM",)


def _looks_like_dicom(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(132)
        # Standard DICOM has preamble 128 bytes + 'DICM'
        if len(head) >= 132 and head[128:132] == b"DICM":
            return True
    except Exception:
        return False
    return False


def iter_dicom_paths(inputs: list[Path], recurse: bool) -> Iterable[Path]:
    """
    Yield candidate file paths; include .dcm and any file that looks like DICOM
    (DICOM prefix).
    """
    for root in inputs:
        if not root.exists():
            continue
        if root.is_file():
            # Ignore DICOMDIR index files
            if root.name.upper() == "DICOMDIR":
                continue
            if root.suffix.lower() == ".dcm" or _looks_like_dicom(root):
                yield root
            continue
        # directory
        if recurse:
            walker = root.rglob("*")
        else:
            walker = root.glob("*")
        for p in walker:
            if not p.is_file():
                continue
            # Ignore DICOMDIR index files
            if p.name.upper() == "DICOMDIR":
                continue
            if p.suffix.lower() == ".dcm" or _looks_like_dicom(p):
                yield p


def iter_pdf_paths(inputs: list[Path], recurse: bool) -> Iterable[Path]:
    """
    Yield PDF files in study folders.
    """
    for root in inputs:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".pdf":
            yield root
            continue
        walker = root.rglob("*.pdf") if recurse else root.glob("*.pdf")
        for p in walker:
            if p.is_file():
                yield p
