# tools/step5_segment.py
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

from src.segmentation.lines_cv import detect_lines_cv, crop_lines, draw_overlays

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(p: Path):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                yield f

def main():
    ap = argparse.ArgumentParser(description="Step-5: Layout & Line Segmentation (OpenCV-only)")
    ap.add_argument("--input", "-i", required=True, help="Image file or folder")
    ap.add_argument("--output", "-o", default=os.path.expanduser("~/Downloads/line_segmentation_output"), help="Output folder (default: ~/Downloads/line_segmentation_output)")
    ap.add_argument("--line-len-frac", type=float, default=0.20, help="Fraction of width to consider a 'long' horizontal line")
    ap.add_argument("--smooth", type=int, default=25, help="Smoothing window for projection (odd)")
    ap.add_argument("--thr-ratio", type=float, default=0.12, help="Threshold ratio of projection peak")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output); out_root.mkdir(parents=True, exist_ok=True)
    report_fp = (out_root / "lines.jsonl").open("w", encoding="utf-8")

    total, total_lines = 0, 0
    for img_path in iter_images(in_path):
        total += 1
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        lines = detect_lines_cv(
            bgr,
            line_len_frac=args.line_len_frac,
            smooth=args.smooth,
            thr_ratio=args.thr_ratio
        )
        total_lines += len(lines)

        # Save overlay
        vis = draw_overlays(bgr, lines)
        stem = img_path.stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), vis)

        # Save crops
        crops = crop_lines(bgr, lines)
        for i, crop in enumerate(crops, 1):
            cv2.imwrite(str(out_dir / f"{stem}_line_{i:02d}.png"), crop)

        # Write JSONL
        record = {
            "image": str(img_path),
            "n_lines": len(lines),
            "lines": [{"order": ln["order"], "bbox": ln["bbox"],
                       "w": ln["bbox"][2]-ln["bbox"][0], "h": ln["bbox"][3]-ln["bbox"][1]} for ln in lines]
        }
        report_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[OK] {img_path.name}: {len(lines)} lines")
    report_fp.close()
    print(f"[DONE] {total} page(s), {total_lines} total lines.")

if __name__ == "__main__":
    main()
