#!/usr/bin/env python3
from __future__ import annotations

# --- add this block at the very top ---
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.segmentation.lines_rule_based import (
    load_gray, page_deskew_by_rules, extract_horizontal_rule_mask,
    pick_rule_rows, build_bands_from_rules, text_ink_mask, crop_by_ruling
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(p: Path):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                yield f

def main():
    ap = argparse.ArgumentParser(description="Step-5: Rule-Based Line Segmentation (Horizontal Ruling Detection)")
    ap.add_argument("--input", "-i", required=True, help="Image file or folder")
    ap.add_argument("--output", "-o", default=os.path.expanduser("~/Downloads/rule_based_segmentation_output"), help="Output folder (default: ~/Downloads/rule_based_segmentation_output)")
    ap.add_argument("--no-deskew", action="store_true", help="Disable deskew (default: enabled)")
    ap.add_argument("--kernel", type=int, default=45, help="Horizontal kernel width for rule enhancement (default 45)")
    ap.add_argument("--min-gap", type=int, default=24, help="Minimum pixel gap between detected rules (default 24)")
    ap.add_argument("--min-strength", type=float, default=0.25, help="Peak strength threshold 0-1 (default 0.25)")
    ap.add_argument("--skip-blank", type=float, default=0.01, help="Min ink ratio to keep a band (default 0.01)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    total, total_lines = 0, 0
    for img_path in iter_images(in_path):
        total += 1
        page_out = out_root / img_path.stem
        page_out.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use the rule-based segmentation
            crop_by_ruling(
                img_path=img_path,
                out_dir=page_out,
                deskew=not args.no_deskew,
                kernel_w=args.kernel,
                min_gap=args.min_gap,
                min_strength=args.min_strength,
                skip_blank_ratio=args.skip_blank,
            )
            
            # Count the number of lines generated
            line_files = list(page_out.glob("line_*.png"))
            total_lines += len(line_files)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")
            continue

    print(f"[DONE] {total} page(s), {total_lines} total lines.")
    print(f"Output saved to: {out_root}")

if __name__ == "__main__":
    main()
