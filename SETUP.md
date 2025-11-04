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
- **Batch Processing**: Multi-threaded processing optimized for large datasets (500+ studies, 30,000+ files)
- **Resume Capability**: Automatically resumes interrupted processing - just restart and it continues where it left off
- **Numerical Ordering**: Processes files in natural order (I000001, I000002, etc.)
- **Progress Feedback**: Clear 4-step progress with updates during metadata reading for large batches
- **First Image Removal**: Automatically removes the first image (I000001.dcm) from each study
- **Temporary File Cleanup**: Removes underscore-prefixed temporary files before and after processing
- **Robust Error Handling**: Individual file errors don't stop the batch - processing continues

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

## Batch Processing (Large Datasets)

The system is optimized for processing large batches (500+ studies with thousands of files):

### Progress Phases

When you start processing, you'll see 4 distinct phases:

1. **Step 1/4: Scanning for temporary files** - Cleans up any leftover temp files
2. **Step 2/4: Discovering files** - Finds all DICOM and PDF files
3. **Step 3/4: Reading metadata** - Reads StudyInstanceUID from all files (shows progress every 5%)
4. **Step 4/4: Processing files** - De-identifies and exports files (shows files/sec and ETA)

### Resume Capability

If processing is interrupted (crash, power loss, cancellation), simply restart with the same input and output directories:

- The system automatically detects already processed files (checks for existing .png outputs)
- Skips those files and only processes remaining files
- Logs how many files were skipped: "Resuming: Skipping X already processed files"
- The CSV report includes all files (both newly processed and skipped)

**Example**: Processing 30,000 files interrupted at file 10,000:
- Restart the application
- Select the same input and output directories
- Processing resumes from file 10,001 automatically

### Performance Tips

- **Workers**: Default is CPU core count - usually optimal
- **For 30,000 files**: Metadata reading (Step 3) may take 5-10 minutes
- **Progress updates**: Watch the log panel for progress during metadata reading
- **Cancellation**: You can cancel anytime - use resume to continue later

