# src/segmentation/lines_mmocr.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from mmocr.apis import MMOCRInferencer

# Lazy global (so Streamlit/CLI reuse it)
_mmocr_det = None

def _get_det():
    global _mmocr_det
    if _mmocr_det is None:
        # DBNet++ is a good default; you can swap to "CRAFT" if needed
        _mmocr_det = MMOCRInferencer(det='DBNetpp', rec=None)
    return _mmocr_det

def detect_lines(image_bgr: np.ndarray, y_tol: int = 14) -> List[Dict]:
    """
    Returns: list of dicts:
      {"bbox": (x0,y0,x1,y1), "polys": [np.ndarray Nx2], "order": i}
    """
    det = _get_det()
    preds = det(image_bgr, return_vis=False)["predictions"][0]
    polys = preds.get("det_polygons", []) or []
    # Convert each polygon to its bbox + y-center
    items = []
    for poly in polys:
        poly = np.array(poly, dtype=np.float32)
        ys = poly[:,1]; xs = poly[:,0]
        items.append({"yc": float(np.mean(ys)),
                      "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                      "poly": poly})
    items.sort(key=lambda d: d["yc"])

    # group into lines by y-centroid distance
    lines: List[Dict] = []
    for it in items:
        placed = False
        for ln in lines:
            if abs(ln["yc"] - it["yc"]) <= y_tol:
                ln["boxes"].append(it["bbox"])
                ln["polys"].append(it["poly"])
                ln["yc"] = (ln["yc"]* (len(ln["boxes"])-1) + it["yc"]) / len(ln["boxes"])
                placed = True
                break
        if not placed:
            lines.append({"yc": it["yc"], "boxes": [it["bbox"]], "polys": [it["poly"]]})

    # tighten each line’s overall bbox
    results = []
    for idx, ln in enumerate(lines):
        x0 = min(b[0] for b in ln["boxes"]); y0 = min(b[1] for b in ln["boxes"])
        x1 = max(b[2] for b in ln["boxes"]); y1 = max(b[3] for b in ln["boxes"])
        results.append({"bbox": (x0,y0,x1,y1), "polys": ln["polys"], "order": idx})
    return results
