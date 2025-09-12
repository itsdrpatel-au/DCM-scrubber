from __future__ import annotations

from pathlib import Path

from dicom_deid_gui.cli import main as cli_main


def test_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "x.dcm").write_bytes(b"\0" * 128 + b"DICM" + b"rest")
    out = tmp_path / "out"
    args = [
        "--input",
        str(inp),
        "--output",
        str(out),
        "--rows",
        "10",
        "--recurse",
        "--dry-run",
    ]
    rc = cli_main(args)
    assert rc == 0
    # Report should exist
    reports = list(out.glob("report_*.csv"))
    assert reports, "Expected a CSV report in dry-run"

