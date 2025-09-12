from __future__ import annotations

from pathlib import Path

from dicom_deid_gui.discovery import iter_dicom_paths


def test_iter_dicom_paths(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    (d / "b.dcm").write_bytes(b"\0" * 128 + b"DICM" + b"rest")

    res = list(iter_dicom_paths([d], recurse=False))
    assert (d / "b.dcm") in res
    assert (d / "a.txt") not in res

