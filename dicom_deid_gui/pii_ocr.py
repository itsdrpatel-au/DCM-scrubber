from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from PIL import Image

# Optional dependency: pytesseract
pytesseract: Any | None
try:
    import pytesseract as _pytesseract

    pytesseract = _pytesseract
except Exception:
    pytesseract = None


# Simple PII heuristics (no LLM): phone, email, SSN-like, MRN-like, dates
PHONE_RE = re.compile(r"\b(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
MRN_RE = re.compile(r"\bMRN[:#]?\s*\w+\b", re.IGNORECASE)
NAME_HINT_RE = re.compile(r"\b(Name|Patient)[:#]?\s+[A-Za-z][A-Za-z'\-]+\b", re.IGNORECASE)


def _to_pil_uint8(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # RGB(A)
        img = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")
        return img
    # grayscale normalize to uint8
    arrf = arr.astype(np.float32)
    mn = float(arrf.min())
    mx = float(arrf.max())
    if mx <= mn:
        out = np.zeros_like(arrf, dtype=np.uint8)
    else:
        out = ((arrf - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def detect_pii_boxes(image_array: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, w, h) of text regions that match PII patterns.

    Uses pytesseract.image_to_data for word-level boxes. If pytesseract is not
    available, returns an empty list.
    """
    if pytesseract is None:
        return []
    img = _to_pil_uint8(image_array)
    try:
        output_type = getattr(pytesseract, "Output").DICT
        data = pytesseract.image_to_data(img, output_type=output_type)
    except Exception:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    n = int(data.get("level", []) and len(data["level"]) or 0)
    for i in range(n):
        text = data["text"][i] if i < len(data.get("text", [])) else ""
        if not text:
            continue
        if _looks_like_pii(text):
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            boxes.append((x, y, w, h))
    return _merge_overlaps(boxes)


def _looks_like_pii(text: str) -> bool:
    if PHONE_RE.search(text):
        return True
    if EMAIL_RE.search(text):
        return True
    if SSN_RE.search(text):
        return True
    if DATE_RE.search(text):
        return True
    if MRN_RE.search(text):
        return True
    if NAME_HINT_RE.search(text):
        return True
    return False


def _merge_overlaps(boxes: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    # Simple merge by expanding overlapping rectangles
    merged: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        x2, y2 = x + w, y + h
        placed = False
        for j, (mx, my, mw, mh) in enumerate(merged):
            mx2, my2 = mx + mw, my + mh
            if not (x2 < mx or mx2 < x or y2 < my or my2 < y):
                # overlap, merge
                nx = min(x, mx)
                ny = min(y, my)
                nx2 = max(x2, mx2)
                ny2 = max(y2, my2)
                merged[j] = (nx, ny, nx2 - nx, ny2 - ny)
                placed = True
                break
        if not placed:
            merged.append((x, y, w, h))
    return merged


def mask_boxes(array: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy of array with rectangles masked to zero.

    Supports 2D (H, W), 3D (H, W, C) or (F, H, W), and 4D (F, H, W, C).
    Masks each frame if present using the same boxes.
    """
    if not boxes:
        return array
    masked = cast(np.ndarray, array.copy())
    if masked.ndim == 2:
        for (x, y, w, h) in boxes:
            masked[y : y + h, x : x + w] = 0
    elif masked.ndim == 3:
        if masked.shape[-1] in (3, 4):  # H, W, C
            for (x, y, w, h) in boxes:
                masked[y : y + h, x : x + w, :] = 0
        else:  # F, H, W
            for f in range(masked.shape[0]):
                for (x, y, w, h) in boxes:
                    masked[f, y : y + h, x : x + w] = 0
    elif masked.ndim == 4:  # F, H, W, C
        for f in range(masked.shape[0]):
            for (x, y, w, h) in boxes:
                masked[f, y : y + h, x : x + w, :] = 0
    return masked


