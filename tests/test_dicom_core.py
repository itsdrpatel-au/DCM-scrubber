from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from dicom_deid_gui.dicom_core import deidentify_and_mask


def make_synthetic(tmp_path: Path) -> Path:
    meta = pydicom.Dataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "a.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "John^Doe"
    ds.PatientID = "12345"
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.Modality = "OT"
    # Pixel data 2D
    arr = np.ones((10, 10), dtype=np.uint16) * 1000
    ds.Rows, ds.Columns = arr.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    p = tmp_path / "a.dcm"
    ds.save_as(str(p))
    return p


def test_uid_regeneration_and_masking(tmp_path: Path) -> None:
    p = make_synthetic(tmp_path)
    uid_cache: dict[tuple[str, str], str] = {}
    ds, warns = deidentify_and_mask(p, rows_to_zero=2, uid_cache=uid_cache)
    assert ds is not None
    # UIDs should be regenerated and consistent per original
    assert ds.StudyInstanceUID != "1.2.3.4"
    assert uid_cache[("StudyInstanceUID", "1.2.3.4")] == ds.StudyInstanceUID
    # PHI blanked
    assert str(ds.PatientName) == ""
    # Pixel masking
    arr = ds.pixel_array
    assert (arr[:2, :] == 0).all()
