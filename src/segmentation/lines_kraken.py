# src/segmentation/lines_kraken.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import cv2
from kraken import blla, binarization

def detect_lines(image_bgr: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bw_pil = binarization.nlbin(Image.fromarray(gray))
    seg = blla.segment(bw_pil)              # baselines + polygons + order
    arr = np.array(bw_pil)

    lines = []
    for i, ln in enumerate(seg.get("lines", [])):
        poly = ln.get("boundary") or ln.get("polygon")
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0,y0,x1,y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        lines.append({"bbox": (x0,y0,x1,y1), "polys": [np.array(poly)], "order": i})
    return lines
