from __future__ import annotations
from typing import List, Dict
import numpy as np
import cv2
from doctr.models import detection as det_models
from doctr.io import read_img

# Lazy global
_doctr_det = None

def _get_detector():
    global _doctr_det
    if _doctr_det is None:
        # DB-ResNet50 is a solid default; fast & accurate
        _doctr_det = det_models.db_resnet50(pretrained=True).eval()
    return _doctr_det

def detect_lines(image_bgr: np.ndarray, y_tol: int = 16) -> List[Dict]:
    """
    Returns list of {"bbox": (x0,y0,x1,y1), "order": i}
    """
    model = _get_detector()

    # docTR expects RGB float [0,1]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred = model.predict(np.expand_dims(rgb, 0))
    # pred[0].pages[0].blocks[].lines[].words[] or .artefacts; but simpler:
    # get boxes at word/granularity and group by y.
    # Convert Absolute (0..1) to pixels:
    h, w = rgb.shape[:2]
    words = []
    for block in pred[0].pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x0, y0, x1, y1) = word.geometry[0]  # ((x0, y0), (x1, y1)) normalized
                words.append({
                    "yc": (y0 + y1) / 2 * h,
                    "bbox": (int(x0*w), int(y0*h), int(x1*w), int(y1*h))
                })
    words.sort(key=lambda d: d["yc"])

    # group by vertical proximity
    lines = []
    for wd in words:
        placed = False
        for ln in lines:
            if abs(ln["yc"] - wd["yc"]) <= y_tol:
                ln["boxes"].append(wd["bbox"])
                ln["yc"] = (ln["yc"]*(len(ln["boxes"])-1) + wd["yc"]) / len(ln["boxes"])
                placed = True
                break
        if not placed:
            lines.append({"yc": wd["yc"], "boxes": [wd["bbox"]]})

    # tighten
    results = []
    for i, ln in enumerate(lines):
        x0 = min(b[0] for b in ln["boxes"]); y0 = min(b[1] for b in ln["boxes"])
        x1 = max(b[2] for b in ln["boxes"]); y1 = max(b[3] for b in ln["boxes"])
        results.append({"bbox": (x0, y0, x1, y1), "order": i})
    return results
