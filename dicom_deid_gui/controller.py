from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .dicom_core import deidentify_and_mask, export_png, write_dicom
from .discovery import iter_dicom_paths
from .reporting import ReportWriter


def run_batch(
    inputs: list[Path],
    output_dir: Path,
    rows: int = 200,
    recurse: bool = True,
    workers: int | None = None,
    dry_run: bool = False,
    on_progress: Callable[[int, int, float, float], None] | None = None,
    on_log: Callable[[str, str], None] | None = None,
    cancelled_flag: threading.Event | None = None,
    export_pngs: bool = False,
) -> Path:
    """
    Discovers files, processes with ThreadPoolExecutor, writes outputs,
    creates CSV report, returns report path.
    Shared uid_cache ensures referential integrity across files.
    Honor cancelled_flag if set.
    """
    # Start time used for progress estimates
    _start_time = time.time()

    discovered = list(iter_dicom_paths(inputs, recurse=recurse))
    total = len(discovered)
    if on_log:
        on_log("info", f"Discovered {total} candidate files")
    if total == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Still emit an empty report
        with ReportWriter(output_dir) as rep:
            report_path = rep.path
        assert report_path is not None
        return report_path

    uid_cache: dict[tuple[str, str], str] = {}
    done = 0
    last_time = time.time()
    last_done = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / "png"

    with ReportWriter(output_dir) as report:
        report_path = report.path

        def update_progress() -> None:
            nonlocal last_time, last_done
            if on_progress:
                now = time.time()
                delta_done = done - last_done
                delta_t = max(now - last_time, 1e-6)
                rate = delta_done / delta_t  # files per second
                remaining = max(total - done, 0)
                eta_s = remaining / rate if rate > 0 else float("inf")
                last_time = now
                last_done = done
                on_progress(done, total, rate, eta_s)

        max_workers = workers or os.cpu_count() or 4
        if on_log:
            on_log("info", f"Starting processing with {max_workers} workers")

        def process_one(path: Path) -> tuple[Path, Path | None, str, str]:
            if cancelled_flag and cancelled_flag.is_set():
                return path, None, "cancelled", "Cancelled before start"
            ds, warns = deidentify_and_mask(path, rows_to_zero=rows, uid_cache=uid_cache)
            notes = "; ".join(warns)
            if ds is None:
                return path, None, "unreadable", notes
            out_path = output_dir / path.name
            if dry_run:
                # Optionally export PNG even in dry-run? We'll skip writing PNGs in dry-run.
                return path, out_path, "dry-run", notes
            try:
                write_dicom(ds, out_path)
                if export_pngs:
                    png_out = png_dir / (path.stem + ".png")
                    ok = export_png(ds, png_out)
                    if not ok and on_log:
                        on_log("warn", f"PNG export failed: {path}")
                return path, out_path, "ok", notes
            except Exception:
                return path, out_path, "write-failed", "Failed to write output"

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_one, p) for p in discovered]
            for fut in as_completed(futures):
                if cancelled_flag and cancelled_flag.is_set():
                    if on_log:
                        on_log("warn", "Cancellation requested; waiting for running tasks")
                    break
                try:
                    inp, outp, status, notes = fut.result()
                except Exception:
                    # Future failed harshly
                    inp = Path("")
                    outp = None
                    status = "error"
                    notes = "Unhandled exception"

                report.write_row(inp, outp, status, notes)
                if on_log:
                    on_log("info", f"{status}: {inp}")
                done += 1
                update_progress()

            # If cancelled, mark remaining as cancelled
            if cancelled_flag and cancelled_flag.is_set():
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
                remaining = total - done
                if on_log and remaining > 0:
                    on_log("warn", f"Cancelled with {remaining} files remaining")

    assert report_path is not None
    return report_path
