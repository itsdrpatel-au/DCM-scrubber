## DCM Scrubber – DICOM De‑Identification (GUI + CLI)

Cross‑platform HIPAA‑safe DICOM de‑identification utility with a PySide6 (Qt) GUI and a mirrored CLI. Performs aggressive metadata scrubbing, pixel masking, PNG previews, and per‑study PDF redaction/extraction, with multithreaded performance and robust reporting.

### Quick Setup

**For detailed installation instructions** (including system dependencies like Tesseract OCR), see [SETUP.md](SETUP.md).

**TL;DR:**
```bash
# macOS
brew install tesseract
pip3 install -e .
python3 -m dicom_deid_gui.app

# Windows
# Install Python and Tesseract (see SETUP.md)
pip install -e .
python -m dicom_deid_gui.app

# Linux
sudo apt install python3 python3-pip tesseract-ocr
pip3 install -e .
python3 -m dicom_deid_gui.app
```

### Developer Quickstart

```bash
# Create and activate a virtual environment
python3 -m venv .venv
. .venv/bin/activate

# Install (with dev tools for lint/tests)
python -m pip install -e '.[dev]'

# Launch GUI
python -m dicom_deid_gui

# CLI example
dicom-deid \
  --input /path/A --input /path/B \
  --output /path/out \
  --rows 200 \
  --recurse \
  --workers 8 \
  --export-png

# Run tests
pytest -q
```

### Key Features

- **Aggressive safe‑harbor de‑ID**: Removes private tags and PHI fields (names, IDs, addresses, phones, institutional/device fields), blanks all DA/TM/DT VRs, scrubs common sequences; preserves SOP Class UID; regenerates Study/Series/SOP Instance/FrameOfReference UIDs (stable mapping within a run); sets `PatientIdentityRemoved=YES`, `DeidentificationMethod=...`, and ensures `ImageType` includes `DERIVED`.
- **Pixel masking**: Zeros the top N rows (default 200) across mono/RGB, single/multi‑frame. If decoding isn’t available (e.g., JPEG/J2K codec missing), continues de‑ID and logs a generic warning; sets `BurnedInAnnotation=NO` after writing.
- **PDF handling (per study)**: Redacts the top third of each page and writes a sidecar `.txt` with text from “Study …” down to just before “Electronically Signed by”.
- **PNG previews**: Optional quick‑look PNGs normalized to 8‑bit (first frame), saved under `output/png/`.
- **Multithreaded**: ThreadPoolExecutor with configurable workers (default = CPU cores).
- **Discovery**: Recursively finds `.dcm` and files with DICOM preamble; also locates PDFs.
- **Reporting**: Timestamped CSV (`input_path,output_path,status,notes`) in the output folder.
- **Dry‑run mode**: Processes and reports without writing outputs.
- **Robust logging**: Rotating file logs plus GUI log pane; logs paths/status only (no PHI).
- **Thread‑safe GUI**: Progress/log updates via Qt signals; cancel button halts promptly.
- **OCR PII scan (no LLM)**: Optional OCR‑based scan that masks detected PII text regions (phones, emails, SSN‑like, MRN hints, dates, name hints) before writing.
- **Skip first image per study**: Automatically deletes the first image (I000001.dcm) from each study after processing (enabled by default).
- **Temporary file cleanup**: Automatically removes underscore-prefixed temporary files from input directories before processing and output directory after processing.
- **Numerical ordering**: Processes files in natural numerical order (I000001, I000002, etc.) for predictable output.
- **Resume capability**: Automatically detects and skips already processed files - if processing fails, simply restart and it will continue where it left off.
- **Batch processing optimized**: Designed for 500+ studies (30,000+ files) with progress feedback during all phases:
  - Step 1: Temporary file cleanup
  - Step 2: File discovery
  - Step 3: Metadata reading (with progress updates)
  - Step 4: De-identification and export
- **Robust error handling**: Individual file failures don't stop the batch - errors are logged and processing continues.

### GUI Overview

- **Inputs**: Add/remove multiple root folders; option to recurse into subfolders.
- **Output**: Folder picker (required).
- **Options**: Recurse, Rows to mask (default 200), Workers (default = CPU cores), Dry run, Export PNG previews.
- **Controls**: Start, Cancel, Open Report, Open Output Folder.
- **Feedback**: Live log pane, progress bar, processed/total with ETA, CSV report after completion or cancel.

### CLI Overview

```bash
dicom-deid \
  --input /path/A --input /path/B \
  --output /path/out \
  --rows 200 \
  --recurse \
  --workers 8 \
  --dry-run \
  --export-png \
  --pii-ocr \
  --skip-first-of-study
```

### Processing Details

- **DICOM**: Opens with `force=True`; scrubs PHI; stable UID remap per run; best‑effort pixel masking with `pylibjpeg` if available; writes with `write_like_original=False` under Implicit VR Little Endian and sane file meta.
- **PDF**: Overlays a white rectangle over the top third of every page; writes a `.txt` with the “Study …” segment extracted up to “Electronically Signed by”.

### Tooling

- Lint: `ruff check .`
- Format: `black .`
- Type‑check: `mypy dicom_deid_gui`
- Tests: `pytest -q`

### Platforms

- macOS, Windows, Linux (Python 3.10+)

### HIPAA Caveats

This tool applies common safe‑harbor techniques but cannot guarantee removal of all PHI in every possible DICOM or PDF. Review outputs per your organization’s policies. No PHI is logged.

Screenshots: (placeholder)


