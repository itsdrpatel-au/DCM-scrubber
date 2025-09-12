## DICOM De-identification GUI/CLI

Cross-platform HIPAA-safe DICOM de-identification utility with PySide6 GUI and CLI.

### Quickstart

```bash
uv venv && uv pip install -e .[dev]
python -m dicom_deid_gui              # launch GUI
dicom-deid --input /path/A --output /path/out --rows 200 --recurse
pytest -q
```

### Features

- Aggressive safe-harbor metadata cleaning and private tag removal
- Pixel masking of top N rows (default 200)
- Multithreaded processing with progress, cancel, and CSV report
- GUI and CLI, cross-platform

### HIPAA Caveats

This tool applies common safe-harbor techniques but cannot guarantee removal of all PHI in every possible DICOM. Review outputs per your organization policies. No PHI is logged.

Screenshots: (placeholder)


