from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pydicom

from .dicom_core import deidentify_and_mask, export_png
from .discovery import iter_dicom_paths, iter_pdf_paths
from .pdf_core import extract_study_text_segment
from .pii_ocr import detect_pii_boxes, mask_boxes
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
    pii_ocr: bool = False,
    skip_first_of_study: bool = False,
) -> Path:
    """
    Discovers files, processes with ThreadPoolExecutor, writes outputs,
    creates CSV report, returns report path.
    Shared uid_cache ensures referential integrity across files.
    Honor cancelled_flag if set.
    """
    # Start time used for progress estimates
    _start_time = time.time()

    # Pre-processing cleanup: Remove temporary files starting with underscore from input directories
    if on_log:
        on_log("info", "Step 1/4: Scanning for temporary files to clean up...")
    cleanup_count = 0
    try:
        for root_dir in inputs:
            if not root_dir.exists():
                continue
            try:
                for temp_file in root_dir.rglob("_*"):
                    if temp_file.is_file():
                        try:
                            temp_file.unlink()
                            cleanup_count += 1
                        except Exception:
                            pass  # Skip files we can't delete
            except Exception:
                pass  # Continue with next directory if this one fails
        if cleanup_count > 0 and on_log:
            on_log("info", f"Cleaned up {cleanup_count} temporary files from input directories")
    except Exception:
        if on_log:
            on_log("warn", "Pre-processing cleanup encountered an error")

    # File discovery with progress feedback
    if on_log:
        on_log("info", "Step 2/4: Discovering DICOM and PDF files...")
    dicoms = list(iter_dicom_paths(inputs, recurse=recurse))
    pdfs = list(iter_pdf_paths(inputs, recurse=recurse))
    
    # Sort files naturally for numerical order processing
    def natural_sort_key(p: Path) -> tuple:
        """Sort by parent directory then filename naturally (handles numbers correctly)."""
        import re
        def atoi(text: str) -> int | str:
            return int(text) if text.isdigit() else text.lower()
        return (str(p.parent), [atoi(c) for c in re.split(r'(\d+)', p.name)])
    
    dicoms = sorted(dicoms, key=natural_sort_key)
    pdfs = sorted(pdfs, key=natural_sort_key)
    discovered = dicoms
    total = len(discovered) + len(pdfs)
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
    # PNG outputs will be nested under each processed study folder (per-file computed)

    # Pre-read original StudyInstanceUIDs for ordering/grouping (best-effort)
    if on_log:
        on_log("info", f"Step 3/4: Reading study metadata from {len(discovered)} files (this may take a few minutes)...")
    orig_study_uid_by_path: dict[Path, str] = {}
    metadata_progress_interval = max(100, len(discovered) // 20)  # Log every 5% or 100 files, whichever is more
    for idx, p in enumerate(discovered, 1):
        try:
            if cancelled_flag and cancelled_flag.is_set():
                if on_log:
                    on_log("warn", "Cancelled during metadata reading")
                return output_dir / "cancelled_report.csv"  # Return placeholder
            ds_tmp = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
            orig_study_uid_by_path[p] = str(getattr(ds_tmp, "StudyInstanceUID", ""))
        except Exception:
            orig_study_uid_by_path[p] = ""
        # Progress logging every N files
        if on_log and idx % metadata_progress_interval == 0:
            percent = (idx / len(discovered)) * 100
            on_log("info", f"Reading metadata... {idx}/{len(discovered)} ({percent:.1f}%)")

    # Resume capability: Check for already processed files
    if on_log:
        on_log("info", "Checking for already processed files (resume support)...")
    already_processed: set[Path] = set()
    
    def _match_root_and_rel_early(path: Path) -> tuple[Path, Path]:
        """Match path to input root and get relative path."""
        best_root: Path | None = None
        best_rel: Path | None = None
        best_len = -1
        for root in inputs:
            try:
                r = path.resolve().relative_to(root.resolve())
            except Exception:
                continue
            root_len = len(str(root))
            if root_len > best_len:
                best_len = root_len
                best_root = root
                best_rel = r
        if best_root is None or best_rel is None:
            return path.parent, Path(path.name)
        return best_root, best_rel
    
    def _processed_base_for_early(root: Path) -> Path:
        return output_dir / f"{root.name}_processed"
    
    resume_check_interval = max(100, len(discovered) // 10)  # Log every 10% or 100 files
    for idx, p in enumerate(discovered, 1):
        if cancelled_flag and cancelled_flag.is_set():
            if on_log:
                on_log("warn", "Cancelled during resume check")
            return output_dir / "cancelled_report.csv"
        root, rel = _match_root_and_rel_early(p)
        processed_base = _processed_base_for_early(root)
        expected_output = (processed_base / rel).with_suffix(".png")
        if expected_output.exists():
            already_processed.add(p)
        # Progress logging
        if on_log and len(discovered) > 500 and idx % resume_check_interval == 0:
            percent = (idx / len(discovered)) * 100
            on_log("info", f"Resume check... {idx}/{len(discovered)} ({percent:.0f}%)")
    
    if already_processed:
        if on_log:
            on_log("info", f"Found {len(already_processed)} already processed files - will skip them (resume mode)")
    elif on_log:
        on_log("info", "No already processed files found - processing all files")
    
    # Calculate max workers before starting
    max_workers = workers or os.cpu_count() or 4
    if on_log:
        on_log("info", f"Step 4/4: Processing {len(discovered) - len(already_processed)} files with {max_workers} workers...")
    
    with ReportWriter(output_dir) as report:
        report_path = report.path
        # Accumulate per-study PII detection info
        study_pii: dict[str, set[Path]] = {}

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

        def _match_root_and_rel(path: Path) -> tuple[Path, Path]:
            """Return (matched_root, relative_path) where relative_path is path relative to root.

            Falls back to (path.parent, Path(path.name)) if no root matches.
            """
            best_root: Path | None = None
            best_rel: Path | None = None
            best_len = -1
            for root in inputs:
                try:
                    r = path.resolve().relative_to(root.resolve())
                except Exception:
                    continue
                root_len = len(str(root))
                if root_len > best_len:
                    best_len = root_len
                    best_root = root
                    best_rel = r
            if best_root is None or best_rel is None:
                return path.parent, Path(path.name)
            return best_root, best_rel

        def _processed_base_for(root: Path) -> Path:
            return output_dir / f"{root.name}_processed"

        # Determine first image per study to skip entirely (e.g., I000001)
        skip_first: set[Path] = set()
        if skip_first_of_study:
            for p in discovered:
                root0, rel0 = _match_root_and_rel(p)
                if rel0.stem == "I000001":
                    skip_first.add(p)
        # Filter processing list: exclude skip_first and already_processed files, recompute total
        proc_dicoms = [p for p in discovered if p not in skip_first and p not in already_processed]
        total = len(proc_dicoms) + len(pdfs)
        
        # Log skipped counts
        if skip_first and on_log:
            on_log("info", f"Skipping {len(skip_first)} first-of-study images (I000001)")
        if already_processed and on_log:
            on_log("info", f"Resuming: Skipping {len(already_processed)} already processed files")

        def process_one(path: Path) -> tuple[Path, Path | None, str, str]:
            """Process a single DICOM file with robust error handling."""
            if cancelled_flag and cancelled_flag.is_set():
                return path, None, "cancelled", "Cancelled before start"
            
            try:
                ds, warns = deidentify_and_mask(path, rows_to_zero=rows, uid_cache=uid_cache)
                notes = "; ".join(warns)
                if ds is None:
                    return path, None, "unreadable", notes
                root, rel = _match_root_and_rel(path)
                processed_base = _processed_base_for(root)
                # We no longer write DICOMs to output; only PNGs and TXT
                out_path = (processed_base / rel).with_suffix(".png")
                if dry_run:
                    # Optionally export PNG even in dry-run? We'll skip writing PNGs in dry-run.
                    return path, out_path, "dry-run", notes
            except Exception as e:
                # Catch any errors during de-identification to prevent thread pool failure
                error_msg = f"De-identification error: {str(e)[:100]}"
                if on_log:
                    on_log("error", f"Failed to process {path.name}: {error_msg}")
                return path, None, "error", error_msg
            
            try:
                # Optional OCR-based PII scan and mask
                if pii_ocr:
                    try:
                        arr = ds.pixel_array  # numpy array
                        boxes = detect_pii_boxes(arr)
                        if boxes:
                            masked_arr = mask_boxes(arr, boxes)
                            ds.PixelData = masked_arr.tobytes()
                    except Exception:
                        pass
                # Always export PNG (only output artifact for images)
                png_out = out_path
                ok = export_png(ds, png_out)
                if not ok:
                    if on_log:
                        on_log("warn", f"PNG export failed: {path}")
                else:
                    # OCR text from PNG and write sidecar .txt
                    try:
                        from PIL import Image
                        try:
                            import pytesseract as _pyt
                        except Exception as e:
                            _pyt = None
                            if on_log:
                                on_log("warn", f"Pytesseract not available: {str(e)[:100]}")
                        if _pyt is not None:
                            img = Image.open(str(png_out))
                            text = _pyt.image_to_string(img)
                            txt_path = png_out.with_suffix(".txt")
                            txt_path.parent.mkdir(parents=True, exist_ok=True)
                            txt_path.write_text(text or "", encoding="utf-8")
                            # Track PII-like hints if detected via boxes
                            if pii_ocr:
                                try:
                                    arr = ds.pixel_array
                                    boxes = detect_pii_boxes(arr)
                                    if boxes:
                                        study_id = (
                                            orig_study_uid_by_path.get(path, "")
                                            or processed_base.name
                                        )
                                        study_pii.setdefault(study_id, set()).add(rel)
                                except Exception:
                                    pass
                    except Exception as e:
                        if on_log:
                            on_log("warn", f"PNG OCR failed: {str(e)[:100]}")
                return path, png_out, "ok", notes
            except Exception as e:
                # Robust error handling for PNG export/OCR failures
                error_msg = f"Output write error: {str(e)[:100]}"
                if on_log:
                    on_log("error", f"Failed to write output for {path.name}: {error_msg}")
                return path, out_path, "write-failed", error_msg

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit only non-skipped DICOMs
            futures = [ex.submit(process_one, p) for p in proc_dicoms]
            # Record skipped items in report (not counted in progress total)
            for sp in skip_first:
                report.write_row(sp, None, "skipped-first", "First image of study; not processed")
            # Record already processed items in report (resume mode)
            for ap in already_processed:
                root_ap, rel_ap = _match_root_and_rel(ap)
                out_ap = (_processed_base_for(root_ap) / rel_ap).with_suffix(".png")
                report.write_row(ap, out_ap, "skipped-resume", "Already processed; skipped in resume mode")
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

            # Process PDFs next
            if not (cancelled_flag and cancelled_flag.is_set()):
                def process_pdf(pdf_path: Path) -> tuple[Path, Path | None, str, str]:
                    if cancelled_flag and cancelled_flag.is_set():
                        return pdf_path, None, "cancelled", "Cancelled before start"
                    # Extract study text only (do not write redacted PDF)
                    root_pdf, rel_pdf = _match_root_and_rel(pdf_path)
                    processed_base_pdf = _processed_base_for(root_pdf)
                    # Target TXT sidecar path
                    txt_out = (processed_base_pdf / rel_pdf).with_suffix(".txt")
                    segment = extract_study_text_segment(pdf_path)
                    # Write text file alongside
                    notes = ""
                    if not dry_run:
                        try:
                            txt_out.parent.mkdir(parents=True, exist_ok=True)
                            txt_out.write_text(segment or "", encoding="utf-8")
                        except Exception:
                            notes = "Failed to write extracted text"
                    status = "ok" if not dry_run else "dry-run"
                    return pdf_path, (txt_out if not dry_run else txt_out), status, notes

                pdf_futures = [ex.submit(process_pdf, p) for p in pdfs]
                for fut in as_completed(pdf_futures):
                    if cancelled_flag and cancelled_flag.is_set():
                        if on_log:
                            on_log("warn", "Cancellation requested; stopping PDF tasks")
                        break
                    try:
                        inp, outp, status, notes = fut.result()
                    except Exception:
                        inp = Path("")
                        outp = None
                        status = "error"
                        notes = "Unhandled exception"
                    report.write_row(inp, outp, status, notes)
                    if on_log:
                        on_log("info", f"{status}: {inp}")
                    done += 1
                    update_progress()

            # After processing DICOMs (and PDFs), optionally delete first image per study
            if skip_first_of_study and not dry_run:
                # Build success lists per study in discovered order
                study_to_paths: dict[str, list[Path]] = {}
                for p in discovered:
                    study_uid = orig_study_uid_by_path.get(p, "")
                    if not study_uid:
                        continue
                    root_p, rel_p = _match_root_and_rel(p)
                    outp = _processed_base_for(root_p) / rel_p
                    if outp.exists():
                        study_to_paths.setdefault(study_uid, []).append(outp)
                # Delete first file per study
                for study_uid, out_list in study_to_paths.items():
                    if not out_list:
                        continue
                    # Prefer file named I000001.* if present
                    candidate: Path = out_list[0]
                    for fp in out_list:
                        if fp.stem == "I000001":
                            candidate = fp
                            break
                    try:
                        try:
                            candidate.unlink(missing_ok=True)
                        except TypeError:
                            if candidate.exists():
                                candidate.unlink()
                        if on_log:
                            on_log("info", f"Deleted first image for study {study_uid}")
                    except Exception:
                        if on_log:
                            on_log("warn", f"Failed deleting first image for study {study_uid}")

            # Study-level summary CSV disabled per request

            # If cancelled, mark remaining as cancelled
            if cancelled_flag and cancelled_flag.is_set():
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
                remaining = total - done
                if on_log and remaining > 0:
                    on_log("warn", f"Cancelled with {remaining} files remaining")
            else:
                # Successful completion
                if on_log:
                    elapsed = time.time() - _start_time
                    on_log("info", f"Processing completed: {done} files in {elapsed:.1f} seconds")

        # Post-processing cleanup: Remove temporary files starting with underscore from output directory
        if not dry_run:
            cleanup_count = 0
            try:
                for temp_file in output_dir.rglob("_*"):
                    if temp_file.is_file():
                        try:
                            temp_file.unlink()
                            cleanup_count += 1
                        except Exception:
                            pass  # Skip files we can't delete
                if cleanup_count > 0 and on_log:
                    on_log("info", f"Cleaned up {cleanup_count} temporary files from output directory")
            except Exception:
                if on_log:
                    on_log("warn", "Post-processing cleanup encountered an error")

    assert report_path is not None
    return report_path
