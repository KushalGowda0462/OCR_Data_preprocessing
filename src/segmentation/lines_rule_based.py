#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


# --------------------- helpers ---------------------
def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def rotate_image_keep_size(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def page_deskew_by_rules(gray: np.ndarray, debug: dict | None = None) -> tuple[np.ndarray, float]:
    """
    Estimate skew from long horizontal lines using Hough, rotate to deskew.
    Returns (deskewed, angle_deg_applied).
    """
    # Light equalization improves line contrast
    eq = cv2.equalizeHist(gray)
    # Edge detection for lines
    edges = cv2.Canny(eq, 50, 150)

    # Prob. Hough lines (favor long horizontals)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=120,
        minLineLength=int(0.5 * gray.shape[1]), maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        if debug is not None: debug["edges"] = edges
        return gray, 0.0

    # compute angles (deg) of near-horizontal segments
    angs = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0: 
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # prefer near-horizontal lines
        if abs(angle) <= 20:
            angs.append(angle)

    if not angs:
        if debug is not None: debug["edges"] = edges
        return gray, 0.0

    # robust central angle (median)
    angle_med = float(np.median(angs))
    rotated = rotate_image_keep_size(gray, -angle_med)
    if debug is not None:
        debug["edges"] = edges
        debug["angle"] = angle_med
    return rotated, -angle_med


def extract_horizontal_rule_mask(gray: np.ndarray, kernel_w: int = 45) -> np.ndarray:
    """
    Create a mask of horizontal ruling lines using morphology.
    Returns a binary mask (255 = horizontal line pixels).
    """
    # Light denoise + contrast
    den = cv2.fastNlMeansDenoising(gray, h=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    den = clahe.apply(den)

    # Binary (text/lines dark) → invert for morphology
    thr = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    # Emphasize long horizontal structures
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horiz_kernel)
    # Optional thinning/closing keeps lines continuous
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))

    return lines  # white=255 on black


def pick_rule_rows(line_mask: np.ndarray, min_gap: int = 24, min_strength: float = 0.25) -> list[int]:
    """
    Pick y-rows that correspond to ruling lines using a vertical projection.
    - min_gap: minimum pixels between two detected rules (tune per DPI).
    - min_strength: peak height fraction vs max peak (0–1).
    """
    profile = line_mask.sum(axis=1).astype(np.float32)
    if profile.max() <= 0:
        return []

    # Normalize and threshold relative to max peak
    prof_norm = profile / (profile.max() + 1e-6)
    ys = []
    i = 1
    H = len(prof_norm)

    while i < H - 1:
        # local peak
        if prof_norm[i] > prof_norm[i - 1] and prof_norm[i] >= prof_norm[i + 1] and prof_norm[i] >= min_strength:
            # refine peak center in a small neighborhood
            start = max(0, i - 2)
            end = min(H, i + 3)
            local = prof_norm[start:end]
            peak = start + int(np.argmax(local))
            if not ys or (peak - ys[-1]) >= min_gap:
                ys.append(peak)
            i = peak + min_gap  # skip ahead by min_gap
        else:
            i += 1

    return ys


def build_bands_from_rules(ys: list[int], height: int, pad_top: int = 4, pad_bot: int = 4) -> list[tuple[int, int]]:
    """
    Convert ruling y-positions to [y0,y1] bands. Uses midpoints between rules.
    Adds top/bottom virtual bounds.
    """
    if not ys:
        return []

    ys = sorted(ys)
    # build boundaries: midpoints between successive rules
    bounds = [0]
    for a, b in zip(ys[:-1], ys[1:]):
        bounds.append((a + b) // 2)
    bounds.append(height)

    bands = []
    for i in range(len(bounds) - 1):
        y0 = max(0, bounds[i] - pad_top)
        y1 = min(height, bounds[i + 1] + pad_bot)
        if y1 - y0 >= 10:  # ignore tiny bands
            bands.append((y0, y1))
    return bands


def text_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of ink-like pixels (white=ink)."""
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    # remove small specks
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return thr


def crop_by_ruling(img_path: Path, out_dir: Path,
                   deskew: bool = True,
                   kernel_w: int = 45,
                   min_gap: int = 24,
                   min_strength: float = 0.25,
                   skip_blank_ratio: float = 0.01):
    """
    Full pipeline:
      - optional deskew
      - detect horizontal ruling
      - banding between rules
      - save crops (from ORIGINAL image)
      - skip bands with low ink
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original color for final crops
    orig_color = Image.open(img_path).convert("RGB")
    gray = load_gray(img_path)

    debug = {}
    if deskew:
        gray, angle_applied = page_deskew_by_rules(gray, debug)
    else:
        angle_applied = 0.0

    # Horizontal-line mask (rules)
    rule_mask = extract_horizontal_rule_mask(gray, kernel_w=kernel_w)
    rule_rows = pick_rule_rows(rule_mask, min_gap=min_gap, min_strength=min_strength)
    bands = build_bands_from_rules(rule_rows, height=gray.shape[0])

    # Ink mask (for blank filtering)
    ink = text_ink_mask(gray)

    saved = 0
    for idx, (y0, y1) in enumerate(bands, 1):
        band_h = y1 - y0
        # Require some ink pixels
        ink_count = int(ink[y0:y1, :].sum() / 255)
        if band_h > 0 and ink_count / (band_h * gray.shape[1]) >= skip_blank_ratio:
            crop = orig_color.crop((0, y0, orig_color.width, y1))
            crop.save(out_dir / f"line_{idx:03d}.png")
            saved += 1

    # Debug images (optional, uncomment to save)
    # cv2.imwrite(str(out_dir / "DEBUG_rule_mask.png"), rule_mask)
    # if "edges" in debug: cv2.imwrite(str(out_dir / "DEBUG_edges.png"), debug["edges"])

    print(f"[✓] {img_path.name}: saved {saved} crops → {out_dir} (deskew {angle_applied:.2f}°)")


# --------------------- CLI ---------------------
def gather_images(path: Path):
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])
    if path.is_file():
        return [path]
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Crop full notebook lines by detecting horizontal ruling."
    )
    ap.add_argument("input", help="Image file or directory")
    ap.add_argument("-o", "--out", default="rule_out", help="Output root folder")
    ap.add_argument("--no-deskew", action="store_true", help="Disable deskew (default: enabled)")
    ap.add_argument("--kernel", type=int, default=45, help="Horizontal kernel width for rule enhancement (default 45)")
    ap.add_argument("--min-gap", type=int, default=24, help="Minimum pixel gap between detected rules (default 24)")
    ap.add_argument("--min-strength", type=float, default=0.25, help="Peak strength threshold 0-1 (default 0.25)")
    ap.add_argument("--skip-blank", type=float, default=0.01, help="Min ink ratio to keep a band (default 0.01)")
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    images = gather_images(inp)
    if not images:
        print("No images found.")
        return

    for img in images:
        page_out = out_root / img.stem
        crop_by_ruling(
            img_path=img,
            out_dir=page_out,
            deskew=not args.no_deskew,
            kernel_w=args.kernel,
            min_gap=args.min_gap,
            min_strength=args.min_strength,
            skip_blank_ratio=args.skip_blank,
        )

    print(f"All done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
