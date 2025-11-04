# DCM Scrubber Setup Instructions

## System Requirements

- Python 3.10 or newer
- Tesseract OCR engine (for text extraction from images)

## Installation Instructions

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.10+** (if needed):
   ```bash
   brew install python@3.11
   ```

3. **Install Tesseract OCR**:
   ```bash
   brew install tesseract
   ```

4. **Clone the repository**:
   ```bash
   git clone https://github.com/itsdrpatel-au/DCM-scrubber.git
   cd DCM-scrubber
   ```

5. **Install Python dependencies**:
   ```bash
   pip3 install -e .
   ```

6. **Run the application**:
   ```bash
   # GUI version
   python3 -m dicom_deid_gui.app
   
   # CLI version
   dicom-deid --help
   ```

### Windows

1. **Install Python 3.10+**:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Install Tesseract OCR**:
   - Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to default location (`C:\Program Files\Tesseract-OCR`)
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

3. **Clone or download the repository**:
   ```bash
   git clone https://github.com/itsdrpatel-au/DCM-scrubber.git
   cd DCM-scrubber
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -e .
   ```

5. **Run the application**:
   ```bash
   # GUI version
   python -m dicom_deid_gui.app
   
   # CLI version
   dicom-deid --help
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and Tesseract**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip tesseract-ocr
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/itsdrpatel-au/DCM-scrubber.git
   cd DCM-scrubber
   ```

3. **Install Python dependencies**:
   ```bash
   pip3 install -e .
   ```

4. **Run the application**:
   ```bash
   # GUI version
   python3 -m dicom_deid_gui.app
   
   # CLI version
   dicom-deid --help
   ```

## Troubleshooting

### "pytesseract.pytesseract.TesseractNotFoundError"
- **macOS**: Run `brew install tesseract`
- **Windows**: Install Tesseract and add to PATH
- **Linux**: Run `sudo apt install tesseract-ocr`

### "python: command not found" (macOS/Linux)
- Use `python3` instead of `python`

### GUI doesn't launch
- Ensure PySide6 installed: `pip3 install PySide6`
- Check display settings

### Import errors
- Reinstall dependencies: `pip3 install -e . --force-reinstall`

### Command Line Tools outdated (macOS)

If you see errors about outdated Command Line Tools:

```bash
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --install
```

Or download manually from [Apple Developer Downloads](https://developer.apple.com/download/all/).

## Features

- **De-identification**: Removes PHI (Protected Health Information) from DICOM metadata
- **Pixel Masking**: Masks top rows of images to remove burned-in annotations
- **OCR Text Extraction**: Extracts text from images for review (requires Tesseract)
- **PDF Processing**: Handles study PDFs alongside DICOM files
- **Batch Processing**: Multi-threaded processing for large datasets
- **First Image Removal**: Automatically removes the first image (I000001.dcm) from each study
- **Temporary File Cleanup**: Removes underscore-prefixed temporary files before and after processing

## Default Settings

- **Rows to mask**: 75 pixels from top
- **Recurse subdirectories**: Enabled
- **Export PNG previews**: Enabled
- **OCR-based PII scan**: Enabled
- **Delete first image per study**: Enabled (I000001.dcm is typically a cover sheet)

## CLI Usage Examples

Basic usage:
```bash
dicom-deid -i /path/to/input -o /path/to/output
```

With custom settings:
```bash
dicom-deid -i /path/to/input -o /path/to/output --rows 100 --recurse --pii-ocr
```

Multiple input directories:
```bash
dicom-deid -i /path/to/input1 -i /path/to/input2 -o /path/to/output
```

Dry run (preview without writing):
```bash
dicom-deid -i /path/to/input -o /path/to/output --dry-run
```

