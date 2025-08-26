#!/usr/bin/env python3
from __future__ import annotations

# --- add this block at the very top ---
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

import argparse, json
from pathlib import Path
import cv2

from src.segmentation.lines_hybrid import detect_lines_hybrid, crop_lines, draw_overlays

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(p: Path):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                yield f

def main():
    ap = argparse.ArgumentParser(description="Step-5: Hybrid Line Segmentation (OpenCV + Rule-Based)")
    ap.add_argument("--input", "-i", required=True, help="Image file or folder")
    ap.add_argument("--output", "-o", default=os.path.expanduser("~/Downloads/hybrid_segmentation_output"), help="Output folder (default: ~/Downloads/hybrid_segmentation_output)")
    ap.add_argument("--line-len-frac", type=float, default=0.20, help="Fraction of width to consider a 'long' horizontal line")
    ap.add_argument("--smooth", type=int, default=25, help="Smoothing window for projection (odd)")
    ap.add_argument("--thr-ratio", type=float, default=0.12, help="Threshold ratio of projection peak")
    ap.add_argument("--kernel", type=int, default=45, help="Horizontal kernel width for rule enhancement")
    ap.add_argument("--min-gap", type=int, default=24, help="Minimum pixel gap between detected rules")
    ap.add_argument("--min-strength", type=float, default=0.25, help="Peak strength threshold 0-1")
    ap.add_argument("--no-deskew", action="store_true", help="Disable deskew")
    ap.add_argument("--no-singleline-snap", action="store_true", help="Disable single-line snap heuristic")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    report_fp = (out_root / "hybrid_lines.jsonl").open("w", encoding="utf-8")

    total, total_lines = 0, 0
    for img_path in iter_images(in_path):
        total += 1
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        # Use hybrid detection
        lines = detect_lines_hybrid(
            bgr,
            use_deskew=not args.no_deskew,
            line_len_frac=args.line_len_frac,
            smooth=args.smooth,
            thr_ratio=args.thr_ratio,
            kernel_w=args.kernel,
            min_gap=args.min_gap,
            min_strength=args.min_strength,
            enable_singleline_snap=not args.no_singleline_snap
        )
        total_lines += len(lines)

        # Save overlay with method information
        vis = draw_overlays(bgr, lines)
        stem = img_path.stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_hybrid_overlay.png"), vis)

        # Save crops
        crops = crop_lines(bgr, lines)
        for i, crop in enumerate(crops, 1):
            cv2.imwrite(str(out_dir / f"{stem}_hybrid_line_{i:02d}.png"), crop)

        # Write detailed JSONL report with method information
        record = {
            "image": str(img_path),
            "n_lines": int(len(lines)),
            "lines": []
        }
        
        for ln in lines:
            x0, y0, x1, y1 = ln["bbox"]
            line_info = {
                "order": int(ln.get("order", 0)),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "w": int(int(x1) - int(x0)),
                "h": int(int(y1) - int(y0)),
                "method": str(ln.get("method", "hybrid")),
                "confidence": float(ln.get("confidence", 0.0)),
                "area": int(ln.get("area", (int(x1) - int(x0)) * (int(y1) - int(y0))))
            }
            record["lines"].append(line_info)
        
        report_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Print summary with method breakdown
        method_counts = {}
        for ln in lines:
            method = ln.get("method", "hybrid")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        method_str = ", ".join([f"{k}: {v}" for k, v in method_counts.items()])
        print(f"[OK] {img_path.name}: {len(lines)} lines ({method_str})")

    report_fp.close()
    print(f"[DONE] {total} page(s), {total_lines} total lines.")
    print(f"Output saved to: {out_root}")
    print("\nMethod Legend:")
    print("  OpenCV: Green boxes - Projection-based detection")
    print("  Rule-based: Red boxes - Horizontal ruling detection") 
    print("  Hybrid: Blue boxes - Fused/combined results")

if __name__ == "__main__":
    main()
