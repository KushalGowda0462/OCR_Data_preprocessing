# src/segmentation/lines_cv.py
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict, Tuple

def _to_binary_inv(bgr: np.ndarray) -> np.ndarray:
    """Text -> 255 (white), background -> 0 (black). Robust for phone photos."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    binv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 15
    )
    # light clean/reconnect
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
    return binv

def _suppress_ruled_lines(binv: np.ndarray) -> np.ndarray:
    """
    Remove LONG horizontal rulings *for segmentation only* (no inpaint).
    We subtract a morphology-opened mask of long horizontals from the text mask.
    """
    h, w = binv.shape
    klen = max(60, w // 5)                     # page-wide lines only
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    long_lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, hker)
    # Gentle dilation so dashed lines are captured
    long_lines = cv2.dilate(long_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (7,1)))
    # Subtract lines from text mask (clamp at 0)
    no_lines = cv2.subtract(binv, long_lines)
    return no_lines

def _bands_from_projection(mask: np.ndarray, smooth: int = 25, thr_ratio: float = 0.12) -> List[Tuple[int,int]]:
    """
    Horizontal projection → contiguous y-intervals ('bands') containing text.
    smooth: window size for 1D smoothing; thr_ratio: % of max to keep.
    """
    proj = mask.sum(axis=1).astype(np.float32) / 255.0
    # smooth with a box filter
    k = max(3, smooth | 1)  # odd
    sm = cv2.blur(proj.reshape(-1,1), (1,k)).ravel()
    thr = sm.max() * thr_ratio
    active = sm > thr

    # convert boolean run-lengths to [y0, y1) bands
    bands, y0 = [], None
    for y, on in enumerate(active):
        if on and y0 is None:
            y0 = y
        elif not on and y0 is not None:
            if y - y0 > 5:  # ignore tiny bands
                bands.append((y0, y))
            y0 = None
    if y0 is not None:
        bands.append((y0, len(active)))
    return bands

def _tighten_band(mask: np.ndarray, y0: int, y1: int, x_pad: int = 6, y_pad: int = 3) -> Tuple[int,int,int,int]:
    """
    Within a horizontal band, find a tight x-extent from connected components.
    Returns bbox (x0,y0,x1,y1).
    """
    band = mask[y0:y1, :]
    if band.size == 0 or band.max() == 0:
        return (0, y0, mask.shape[1], y1)
    # find connected components of text in the band
    num, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    xs0, xs1 = [], []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20 or h < 3:  # filter tiny specks
            continue
        xs0.append(x); xs1.append(x + w)
    if not xs0:
        return (0, y0, mask.shape[1], y1)
    x0 = max(0, min(xs0) - x_pad)
    x1 = min(mask.shape[1], max(xs1) + x_pad)
    return (x0, max(0, y0 - y_pad), x1, min(mask.shape[0], y1 + y_pad))

def detect_lines_cv(image_bgr: np.ndarray, line_len_frac: float = 0.20, smooth: int = 25, thr_ratio: float = 0.12) -> List[Dict]:
    """
    Pure-OpenCV line segmentation.
    Returns a list (top→bottom) of dicts:
      { "bbox": (x0,y0,x1,y1), "order": i }
    """
    binv = _to_binary_inv(image_bgr)
    mask = _suppress_ruled_lines(binv)
    bands = _bands_from_projection(mask, smooth, thr_ratio)

    lines = []
    for i, (y0, y1) in enumerate(bands):
        x0, y0x, x1, y1x = _tighten_band(mask, y0, y1)
        lines.append({"bbox": (int(x0), int(y0x), int(x1), int(y1x)), "order": i})
    return lines

def crop_lines(image_bgr: np.ndarray, lines: List[Dict]) -> List[np.ndarray]:
    crops = []
    for ln in lines:
        x0,y0,x1,y1 = ln["bbox"]
        crops.append(image_bgr[y0:y1, x0:x1].copy())
    return crops

def draw_overlays(image_bgr: np.ndarray, lines: List[Dict]) -> np.ndarray:
    """Draw bounding boxes around detected lines for visualization."""
    vis = image_bgr.copy()
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # Add line number
        cv2.putText(vis, str(ln["order"]), (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis
