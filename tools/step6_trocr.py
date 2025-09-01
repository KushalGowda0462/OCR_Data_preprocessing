# tools/step6_trocr.py
from __future__ import annotations
import sys, os, json, argparse
from pathlib import Path
import cv2

# Make 'src' importable when running from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.lines_cv import detect_lines_cv, crop_lines
from src.recognition.trocr_recognizer import TrOCRRecognizer


def draw_overlay(img_bgr, line_texts):
    """Draw green boxes with red text labels just above each box."""
    vis = img_bgr.copy()
    for ln in line_texts:
        (x0, y0, x1, y1) = ln["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = (ln["text"] or "").strip()
        if len(label) > 60:
            label = label[:60] + "…"
        y_text = max(12, y0 - 6)
        cv2.putText(vis, label, (x0, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def iter_images(p: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if p.is_file() and p.suffix.lower() in exts:
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                yield f


def main():
    ap = argparse.ArgumentParser(description="Step-6: Recognition with TrOCR only (with overlay output)")
    ap.add_argument("--input", "-i", required=True, help="Image file or folder")
    ap.add_argument("--output", "-o", required=True, help="Output folder")
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten",
                    help="HF model id (default: microsoft/trocr-base-handwritten)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    trocr = TrOCRRecognizer(model_name=args.model)

    report_fp = (out_root / "recognized.jsonl").open("w", encoding="utf-8")
    pages_done = 0

    for img_path in iter_images(in_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        # Step 5: segmentation
        lines = detect_lines_cv(img)
        crops = crop_lines(img, lines)

        # --- define the list BEFORE appending ---
        line_texts = []

        # Step 6: recognition
        for ln, crop in zip(lines, crops):
            txt = trocr.recognize_line(crop)
            line_texts.append({"order": ln["order"], "bbox": ln["bbox"], "text": txt})

        # Save JSONL record
        rec = {"image": str(img_path), "lines": line_texts}
        report_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Save overlay image
        vis = draw_overlay(img, line_texts)
        out_img_path = out_root / f"{img_path.stem}_overlay.png"
        cv2.imwrite(str(out_img_path), vis)

        print(f"[OK] {img_path.name}: {len(line_texts)} lines recognized → {out_img_path.name}")
        pages_done += 1

    report_fp.close()
    print(f"[DONE] {pages_done} page(s). Results in {out_root}\\recognized.jsonl and overlay PNGs.")


if __name__ == "__main__":
    main()
