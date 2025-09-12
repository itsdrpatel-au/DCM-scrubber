from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO


def open_report(out_dir: Path) -> Path:
    """Return path to latest CSV report in out_dir if exists, else raise FileNotFoundError."""
    candidates = sorted(out_dir.glob("report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No report CSV found")
    return candidates[0]


class ReportWriter:
    """
    Context manager that opens a timestamped CSV and writes header:
    input_path,output_path,status,notes
    Supports .write_row(...)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.file: TextIO | None = None
        self.writer: Any | None = None
        self.path: Path | None = None

    def __enter__(self) -> ReportWriter:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"report_{ts}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["input_path", "output_path", "status", "notes"])
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.file:
            self.file.close()

    def write_row(
        self, input_path: Path, output_path: Path | None, status: str, notes: str
    ) -> None:
        if not self.writer or not self.file or not self.path:
            raise RuntimeError("ReportWriter is not open")
        out_str = str(output_path) if output_path is not None else ""
        # Ensure no PHI is included (only paths and generic notes)
        self.writer.writerow([str(input_path), out_str, status, notes])


def write_study_summary(out_dir: Path, rows: list[tuple[str, bool, str]]) -> Path:
    """Write a per-study summary CSV with columns: study,pii_detected,pii_images.

    - study: study folder name or identifier
    - pii_detected: YES/NO
    - pii_images: semicolon-separated list of relative image paths with detected PII
    Returns the path to the created CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"report_summary_{ts}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["study", "pii_detected", "pii_images"])  # header
        for study, detected, images in rows:
            w.writerow([study, "YES" if detected else "NO", images])
    return path
