from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

PHI_KEYWORDS: set[str] = {
    # Patient/Person identifiers and names
    "PatientName",
    "PatientID",
    "OtherPatientIDs",
    "OtherPatientNames",
    "IssuerOfPatientID",
    "PatientBirthName",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "CountryOfResidence",
    "RegionOfResidence",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "MilitaryRank",
    "BranchOfService",
    "MedicalAlerts",
    "Allergies",
    # Visit/Institution
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "ReferringPhysicianTelephoneNumbers",
    "StationName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "PhysiciansOfRecord",
    "RequestingPhysician",
    "RequestingService",
    "StudyID",
    "IssuerOfAccessionNumberSequence",
    # Device / Operators
    "DeviceSerialNumber",
    "OperatorName",
    # Free text/comment fields commonly used
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "RequestedProcedureDescription",
    "PerformedProcedureStepDescription",
    "CommentsOnThePerformedProcedureStep",
    "ImageComments",
    "AdditionalPatientHistory",
}


DATE_VRS: set[str] = {"DA", "TM", "DT"}


def _blank_element_value(elem: pydicom.dataelem.DataElement) -> None:
    try:
        if elem.VR in DATE_VRS:
            elem.value = ""
        elif elem.VR in {"PN", "LO", "LT", "SH", "ST", "UT"}:
            elem.value = ""
        elif elem.VR in {"SQ"}:  # handled elsewhere
            pass
        else:
            # For safety, blank string for textual-like; leave numeric alone
            if isinstance(elem.value, str):
                elem.value = ""
    except Exception:
        # Best-effort blanking
        try:
            elem.value = ""
        except Exception:
            pass


def _scrub_dataset(ds: Dataset) -> None:
    # Remove private tags first
    ds.remove_private_tags()

    # Walk elements and blank PHI
    to_process: list[Dataset] = [ds]
    while to_process:
        cur = to_process.pop()
        for elem in list(cur.iterall()):
            try:
                if elem.VR == "SQ":
                    for item in elem.value:
                        if isinstance(item, Dataset):
                            to_process.append(item)
                    continue
                if elem.keyword in PHI_KEYWORDS:
                    _blank_element_value(elem)
                if elem.VR in DATE_VRS:
                    _blank_element_value(elem)
            except Exception:
                # Continue on any errors
                continue


def _get_or_make_uid(uid_cache: dict[tuple[str, str], str], key: str, old_uid: str | None) -> str:
    original = old_uid or ""
    cache_key = (key, original)
    if cache_key in uid_cache:
        return uid_cache[cache_key]
    new_uid = generate_uid(prefix=None)
    uid_cache[cache_key] = new_uid
    return new_uid


def _ensure_deid_flags(ds: Dataset) -> None:
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Basic profile; aggressive safe-harbor style"
    try:
        image_type = list(ds.get("ImageType", []))
        if "DERIVED" not in image_type:
            image_type.append("DERIVED")
        if image_type:
            ds.ImageType = image_type
    except Exception:
        pass


def _mask_top_rows(array: np.ndarray, rows_to_zero: int) -> np.ndarray:
    if rows_to_zero <= 0:
        return array
    masked: np.ndarray = array.copy()
    try:
        if masked.ndim == 2:
            masked[:rows_to_zero, :] = 0
        elif masked.ndim == 3:
            # H x W x C or F x H x W
            if masked.shape[-1] in (3, 4):  # likely channels last
                masked[:rows_to_zero, :, :] = 0
            else:  # interpret as F x H x W
                for f in range(masked.shape[0]):
                    masked[f, :rows_to_zero, :] = 0
        elif masked.ndim == 4:
            # F x H x W x C
            masked[:, :rows_to_zero, :, :] = 0
        else:
            # Unknown shape, best-effort on first two dims if present
            slices = [slice(None)] * masked.ndim
            if masked.ndim >= 2:
                slices[0] = slice(0, rows_to_zero)
                slices[1] = slice(None)
                masked[tuple(slices)] = 0
    except Exception:
        # do not fail masking entirely
        return array
    return masked


def deidentify_and_mask(
    in_path: Path,
    rows_to_zero: int,
    uid_cache: dict[tuple[str, str], str],
) -> tuple[FileDataset | None, list[str]]:
    warnings: list[str] = []
    try:
        ds = pydicom.dcmread(str(in_path), force=True)
    except Exception:
        warnings.append("Unreadable DICOM; skipped")
        return None, warnings

    # Remove PHI and private tags
    _scrub_dataset(ds)

    # Regenerate key UIDs with stable mapping per run
    try:
        for key, tag_name in (
            ("StudyInstanceUID", "StudyInstanceUID"),
            ("SeriesInstanceUID", "SeriesInstanceUID"),
            ("SOPInstanceUID", "SOPInstanceUID"),
            ("FrameOfReferenceUID", "FrameOfReferenceUID"),
        ):
            old_value = str(getattr(ds, tag_name, "")) or ""
            new_value = _get_or_make_uid(uid_cache, key, old_value)
            setattr(ds, tag_name, new_value)
    except Exception:
        warnings.append("Failed to regenerate some UIDs")

    # Pixel masking best-effort
    try:
        arr = ds.pixel_array  # pydicom provides numpy array
        masked: np.ndarray = _mask_top_rows(np.asarray(arr), rows_to_zero)
        # Assign back. pydicom will handle converting numpy array to bytes on save
        ds.PixelData = masked.tobytes()
        # Update Rows/Columns if needed (shape preserved)
        if masked.ndim == 2:
            ds.Rows, ds.Columns = masked.shape[0], masked.shape[1]
        elif masked.ndim == 3:
            # Could be HxWxC or FxHxW; keep existing Rows/Columns
            pass
        elif masked.ndim == 4:
            pass
        ds.BurnedInAnnotation = "NO"
    except Exception:
        warnings.append("Pixel decode/mask unavailable; metadata de-identified only")

    _ensure_deid_flags(ds)
    return ds, warnings


def write_dicom(
    ds: FileDataset,
    out_path: Path,
) -> None:
    # Ensure sane file meta and transfer syntax
    try:
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = FileMetaDataset()
        if not getattr(ds.file_meta, "TransferSyntaxUID", None):
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = True
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)


def export_png(ds: FileDataset, out_png_path: Path) -> bool:
    """Export a quick PNG preview from the dataset's pixel data.

    - Uses first frame if multi-frame
    - Converts to 8-bit for portability
    - Handles MONO and RGB
    Returns True on success, False otherwise.
    """
    try:
        from PIL import Image
    except Exception:
        return False

    try:
        arr = ds.pixel_array  # numpy ndarray
    except Exception:
        return False

    arr_np = np.asarray(arr)
    # Reduce dimensions to a single frame
    if arr_np.ndim == 4:  # F x H x W x C
        arr_np = arr_np[0]
    elif arr_np.ndim == 3 and arr_np.shape[-1] not in (3, 4):  # F x H x W
        arr_np = arr_np[0]

    mode = "L"
    if arr_np.ndim == 3 and arr_np.shape[-1] in (3, 4):
        img_arr = _to_uint8(arr_np[..., :3])
        mode = "RGB"
    else:
        img_arr = _to_uint8(arr_np)
        mode = "L"

    image = Image.fromarray(img_arr, mode=mode)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_png_path))
    return True


def _to_uint8(x: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 safely (clips if constant)."""
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - x_min) / (x_max - x_min)
    y = (y * 255.0).clip(0, 255)
    return y.astype(np.uint8)
