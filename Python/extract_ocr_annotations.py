import os
import pydicom
import pytesseract
from PIL import Image

def convert_dicom_to_pil(dicom_path):
    """
    Reads a DICOM file and converts it into a PIL Image.
    Converts the pixel data to an 8-bit RGB image.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        # Convert the pixel data to float and normalize to 0-255
        img_array = ds.pixel_array.astype("float")
        img_array = 255 * (img_array - img_array.min()) / (img_array.ptp() + 1e-8)
        img_array = img_array.astype("uint8")
        pil_img = Image.fromarray(img_array)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return None

def process_dicom_file(dicom_path):
    """
    Processes a single DICOM file:
      - Converts the DICOM to a PIL image.
      - Extracts OCR text using pytesseract.
      - Returns the extracted text.
    """
    image = convert_dicom_to_pil(dicom_path)
    if image is None:
        return ""
    ocr_text = pytesseract.image_to_string(image)
    return ocr_text.strip()

def process_study_folder(study_folder):
    """
    Iterates over all DICOM files in the study folder.
    For each DICOM file, extracts OCR text and writes it to a corresponding
    annotation file named <original_filename>_annotation.txt in the same folder.
    """
    for filename in os.listdir(study_folder):
        if filename.lower().endswith('.dcm'):
            dicom_path = os.path.join(study_folder, filename)
            print(f"Processing OCR for {dicom_path}")
            ocr_text = process_dicom_file(dicom_path)
            
            # Create a new filename for the annotation text file.
            base, _ = os.path.splitext(filename)
            annotation_filename = f"{base}_annotation.txt"
            annotation_path = os.path.join(study_folder, annotation_filename)
            
            with open(annotation_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            print(f"OCR annotation saved to {annotation_path}")

def process_all_study_folders(root_folder):
    """
    Iterates through each study folder in the root folder (pilot_dataset)
    and processes each one.
    """
    for folder in os.listdir(root_folder):
        study_path = os.path.join(root_folder, folder)
        if os.path.isdir(study_path):
            print(f"\nProcessing study folder: {study_path}")
            process_study_folder(study_path)

if __name__ == "__main__":
    # Assume that pilot_dataset is in the project root.
    root_folder = "pilot_dataset"
    if not os.path.isdir(root_folder):
        print(f"Folder '{root_folder}' not found in project root!")
    else:
        process_all_study_folders(root_folder)
