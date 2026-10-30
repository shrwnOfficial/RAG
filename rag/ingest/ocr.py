"""OpenCV preprocessing and Tesseract OCR."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytesseract
from PIL import Image

import config


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """Grayscale, deskew lightly, adaptive threshold."""
    bgr = _pil_to_cv(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    coords = np.column_stack(np.where(gray > 0))
    angle = 0.0
    if len(coords) > 100:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
    if abs(angle) > 0.5:
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(
            gray,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def ocr_image(image: Image.Image, psm: int = 3) -> str:
    processed = preprocess_for_ocr(image)
    config_str = f"--psm {psm} -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(
        processed,
        lang=config.TESSERACT_LANG,
        config=config_str,
    )
    return text.strip()


def ocr_bytes(data: bytes) -> str:
    image = Image.open(io.BytesIO(data))
    return ocr_image(image)


def page_needs_ocr(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < config.OCR_MIN_CHARS:
        return True
    alnum = sum(1 for c in stripped if c.isalnum())
    if len(stripped) == 0:
        return True
    return (alnum / len(stripped)) < config.OCR_MIN_ALNUM_RATIO
