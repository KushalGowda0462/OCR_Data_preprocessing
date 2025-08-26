# src/preprocessing/core.py
import cv2, numpy as np
from dataclasses import dataclass
from typing import Optional
from .models import ProcessingResult  # as in your repo

# --- NEW: safe helpers -------------------------------------------------
def _deskew(gray: np.ndarray) -> np.ndarray:
    # robust deskew using minAreaRect
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_safe(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    binv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8))
    return binv
# -----------------------------------------------------------------------

class DocumentPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, bgr: np.ndarray) -> ProcessingResult:
        """
        Returns a ProcessingResult with at least:
          - .ok (bool)
          - .binary (np.ndarray, uint8 0/255, text=white pixels)
          - .meta (dict)  # add any timing/flags here
        """
        try:
            if getattr(self.cfg, "SAFE_PREPROCESS", True):
                binv = preprocess_safe(bgr)
                return ProcessingResult(ok=True, binary=binv, meta={"mode": "safe"})
            else:
                # fall back to your ORIGINAL path (HSV isolate + line removal + inpaint)
                binv = self._original_pipeline(bgr)   # <-- call your existing method
                return ProcessingResult(ok=True, binary=binv, meta={"mode": "original"})
        except Exception as e:
            return ProcessingResult(ok=False, binary=None, meta={"error": f"{type(e).__name__}: {e}"})
