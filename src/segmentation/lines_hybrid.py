#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Import both methods
from .lines_cv import _to_binary_inv, _suppress_ruled_lines, _bands_from_projection, _tighten_band
from .lines_rule_based import (
    load_gray, page_deskew_by_rules, extract_horizontal_rule_mask,
    pick_rule_rows, build_bands_from_rules, text_ink_mask
)

class HybridLineSegmenter:
    """
    Hybrid line segmentation combining:
    1. OpenCV projection-based method (good for text density)
    2. Rule-based method (good for structured documents)
    3. Intelligent fusion and conflict resolution
    """
    
    def __init__(self, 
                 use_deskew: bool = True,
                 confidence_threshold: float = 0.7,
                 overlap_threshold: float = 0.3,
                 fragment_height_ratio: float = 0.35,
                 merge_gap_ratio: float = 0.35,
                 singleline_height_ratio: float = 0.12):
        self.use_deskew = use_deskew
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        # Guardrails/heuristics
        self.fragment_height_ratio = fragment_height_ratio  # discard < 0.35 * median_h
        self.merge_gap_ratio = merge_gap_ratio              # merge gap threshold vs median_h
        self.singleline_height_ratio = singleline_height_ratio
        
    def detect_lines_hybrid(self, 
                           image_bgr: np.ndarray,
                           line_len_frac: float = 0.20,
                           smooth: int = 25,
                           thr_ratio: float = 0.12,
                           kernel_w: int = 45,
                           min_gap: int = 24,
                           min_strength: float = 0.25) -> List[Dict]:
        """
        Hybrid line detection combining both methods.
        Returns list of dicts with bbox, order, confidence, and method info.
        """
        H, W = image_bgr.shape[:2]
        # Step 1: Preprocessing and deskew
        if self.use_deskew:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray, angle_applied = page_deskew_by_rules(gray)
            # Apply same deskew to BGR image for consistency
            h, w = image_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle_applied, 1.0)
            image_bgr = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            angle_applied = 0.0
            
        # Step 2: Run both segmentation methods
        # Adaptive smoothing for handwriting to prevent over-splitting
        adaptive_smooth = max(25, int(round(0.02 * H)))
        opencv_lines = self._detect_opencv_lines(image_bgr, line_len_frac, adaptive_smooth, thr_ratio)
        rule_lines = self._detect_rule_lines(image_bgr, kernel_w, min_gap, min_strength)
        
        # Step 3: Intelligent fusion
        fused_lines = self._fuse_detections(opencv_lines, rule_lines, image_bgr.shape)
        
        # Step 4: Post-process and validate (merge fragments, guardrails, confidence)
        final_lines = self._post_process_lines(fused_lines, image_bgr)
        
        # Step 5: Single-line heuristic for pages that are essentially one line
        if final_lines:
            heights = [ln["bbox"][3] - ln["bbox"][1] for ln in final_lines]
            median_h = float(np.median(heights)) if heights else 0.0
            if median_h > self.singleline_height_ratio * H and len(final_lines) <= 3:
                final_lines = [{
                    "bbox": (0, 0, W, H),
                    "order": 0,
                    "method": "hybrid",
                    "confidence": 0.95,
                    "area": W * H,
                }]
        
        return final_lines
    
    def _detect_opencv_lines(self, image_bgr: np.ndarray, line_len_frac: float, smooth: int, thr_ratio: float) -> List[Dict]:
        """Run OpenCV projection-based detection."""
        try:
            binv = _to_binary_inv(image_bgr)
            mask = _suppress_ruled_lines(binv)
            bands = _bands_from_projection(mask, smooth, thr_ratio)
            
            lines = []
            for i, (y0, y1) in enumerate(bands):
                x0, y0x, x1, y1x = _tighten_band(mask, y0, y1)
                lines.append({
                    "bbox": (int(x0), int(y0x), int(x1), int(y1x)),
                    "order": i,
                    "method": "opencv",
                    "confidence": 0.8,  # Base confidence for OpenCV
                    "area": (x1-x0) * (y1x-y0x)
                })
            return lines
        except Exception as e:
            logging.warning(f"OpenCV detection failed: {e}")
            return []
    
    def _detect_rule_lines(self, image_bgr: np.ndarray, kernel_w: int, min_gap: int, min_strength: float) -> List[Dict]:
        """Run rule-based detection."""
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            rule_mask = extract_horizontal_rule_mask(gray, kernel_w=kernel_w)
            rule_rows = pick_rule_rows(rule_mask, min_gap, min_strength)
            bands = build_bands_from_rules(rule_rows, height=gray.shape[0])
            
            lines = []
            for i, (y0, y1) in enumerate(bands):
                # Use full width for rule-based lines
                lines.append({
                    "bbox": (0, y0, gray.shape[1], y1),
                    "order": i,
                    "method": "rule_based",
                    "confidence": 0.9,  # Higher confidence for rule-based
                    "area": gray.shape[1] * (y1-y0)
                })
            return lines
        except Exception as e:
            logging.warning(f"Rule-based detection failed: {e}")
            return []
    
    def _fuse_detections(self, opencv_lines: List[Dict], rule_lines: List[Dict], image_shape: Tuple) -> List[Dict]:
        """
        Intelligent fusion of both detection methods.
        - Resolves conflicts based on confidence and overlap
        - Prefers rule-based for structured documents
        - Falls back to OpenCV for complex layouts
        """
        if not opencv_lines and not rule_lines:
            return []
        
        if not opencv_lines:
            return rule_lines
        if not rule_lines:
            return opencv_lines
            
        # Calculate overlap matrix
        overlap_matrix = self._calculate_overlap_matrix(opencv_lines, rule_lines)
        
        # Fusion strategy: prefer rule-based when available, supplement with OpenCV
        fused_lines = []
        used_opencv = set()
        used_rule = set()
        
        # First, add high-confidence rule-based lines
        for i, rule_line in enumerate(rule_lines):
            if rule_line["confidence"] >= self.confidence_threshold:
                fused_lines.append(rule_line)
                used_rule.add(i)
        
        # Then, add OpenCV lines that don't significantly overlap with rule lines
        for i, opencv_line in enumerate(opencv_lines):
            if opencv_line["confidence"] >= self.confidence_threshold:
                # Check if this line significantly overlaps with any rule line
                significant_overlap = False
                for j, rule_line in enumerate(rule_lines):
                    if j in used_rule:
                        overlap = overlap_matrix[i][j]
                        if overlap > self.overlap_threshold:
                            significant_overlap = True
                            break
                
                if not significant_overlap:
                    fused_lines.append(opencv_line)
                    used_opencv.add(i)
        
        # Sort by y-coordinate (top to bottom)
        fused_lines.sort(key=lambda x: x["bbox"][1])
        
        # Reassign order numbers
        for i, line in enumerate(fused_lines):
            line["order"] = i
            line["method"] = "hybrid"  # Mark as fused result
            
        return fused_lines
    
    def _calculate_overlap_matrix(self, opencv_lines: List[Dict], rule_lines: List[Dict]) -> np.ndarray:
        """Calculate overlap between OpenCV and rule-based detections."""
        if not opencv_lines or not rule_lines:
            return np.array([])
            
        matrix = np.zeros((len(opencv_lines), len(rule_lines)))
        
        for i, opencv_line in enumerate(opencv_lines):
            for j, rule_line in enumerate(rule_lines):
                overlap = self._calculate_bbox_overlap(opencv_line["bbox"], rule_line["bbox"])
                matrix[i][j] = overlap
                
        return matrix
    
    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _post_process_lines(self, lines: List[Dict], image_bgr: np.ndarray) -> List[Dict]:
        """Post-process and validate detected lines: merge adjacent, guardrails (drop fragments, split tall), recompute order and confidence."""
        if not lines:
            return []
        
        # Filter degenerate boxes and sort
        valid = []
        for line in lines:
            x0, y0, x1, y1 = line["bbox"]
            if x1 > x0 and y1 > y0:
                valid.append(line)
        if not valid:
            return []
        valid.sort(key=lambda ln: ln["bbox"][1])
        
        # Compute median height
        heights = [ln["bbox"][3] - ln["bbox"][1] for ln in valid]
        median_h = float(np.median(heights)) if heights else 0.0
        
        # 1) Merge adjacent bands BEFORE height filtering when small gap and sufficient overlap
        merged_once = self._merge_adjacent_with_overlap(valid, median_h)
        
        # Recompute median after initial merge
        heights = [ln["bbox"][3] - ln["bbox"][1] for ln in merged_once]
        median_h = float(np.median(heights)) if heights else 0.0
        
        # Guardrail thresholds
        min_ok = 0.40 * median_h if median_h > 0 else 0
        max_ok = 1.90 * median_h if median_h > 0 else float('inf')
        
        # 2) Drop fragments below min_ok
        filtered = [ln for ln in merged_once if (ln["bbox"][3] - ln["bbox"][1]) >= min_ok]
        
        # 3) Split too-tall crops above max_ok using internal projection with smaller smoothing
        split_adjusted: List[Dict] = []
        H, W = image_bgr.shape[:2]
        for ln in filtered:
            h = ln["bbox"][3] - ln["bbox"][1]
            if h > max_ok and median_h > 0:
                split_lines = self._split_tall_line(image_bgr, ln, thr_ratio=0.12)
                if split_lines:
                    split_adjusted.extend(split_lines)
                else:
                    split_adjusted.append(ln)
            else:
                split_adjusted.append(ln)
        
        # Recompute confidence metrics (ink density and border darkness)
        enhanced = []
        gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        for ln in split_adjusted:
            x0, y0, x1, y1 = ln["bbox"]
            roi = gray_full[y0:y1, x0:x1]
            if roi.size == 0:
                ln["confidence"] = float(ln.get("confidence", 0.5)) * 0.7
                enhanced.append(ln)
                continue
            # Ink density
            _, binv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink_density = float(np.sum(binv > 0)) / float(binv.size)
            # Top/bottom border black ratio
            tb = max(1, min(3, roi.shape[0] // 20))
            top_strip = binv[:tb, :]
            bot_strip = binv[-tb:, :]
            border_black = 0.5 * (np.mean(top_strip > 0) + np.mean(bot_strip > 0))
            # Confidence adjustment
            base_conf = float(ln.get("confidence", 0.8))
            conf = base_conf * (0.6 + 0.3 * min(1.0, ink_density * 2.0)) * (0.8 + 0.2 * (1.0 - border_black))
            ln["confidence"] = float(min(1.0, max(0.0, conf)))
            ln["ink_density"] = float(ink_density)
            ln["border_black"] = float(border_black)
            enhanced.append(ln)
        
        # Sort by y, reassign order
        enhanced.sort(key=lambda x: x["bbox"][1])
        for i, ln in enumerate(enhanced):
            ln["order"] = i
        
        return enhanced

    def _merge_adjacent_with_overlap(self, lines: List[Dict], median_h: float) -> List[Dict]:
        if len(lines) <= 1:
            return lines
        lines_sorted = sorted(lines, key=lambda ln: ln["bbox"][1])
        merged: List[Dict] = []
        gap_thr = self.merge_gap_ratio * median_h if median_h > 0 else 0
        
        def overlap_over_narrower(a, b) -> float:
            ax0, _, ax1, _ = a
            bx0, _, bx1, _ = b
            inter = max(0, min(ax1, bx1) - max(ax0, bx0))
            narrower = max(1, min(ax1 - ax0, bx1 - bx0))
            return inter / narrower
        
        current = lines_sorted[0]
        for nxt in lines_sorted[1:]:
            x0, y0, x1, y1 = current["bbox"]
            nx0, ny0, nx1, ny1 = nxt["bbox"]
            vgap = max(0, ny0 - y1)
            if gap_thr > 0 and vgap <= gap_thr and overlap_over_narrower(current["bbox"], nxt["bbox"]) >= 0.5:
                merged_bbox = (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1))
                current = {
                    "bbox": merged_bbox,
                    "order": current.get("order", 0),
                    "method": "hybrid",
                    "confidence": max(float(current.get("confidence", 0.7)), float(nxt.get("confidence", 0.7))),
                    "area": (merged_bbox[2]-merged_bbox[0]) * (merged_bbox[3]-merged_bbox[1])
                }
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged

    def _split_tall_line(self, image_bgr: np.ndarray, line: Dict, thr_ratio: float = 0.12) -> List[Dict]:
        """Split a too-tall line by running projection inside the crop with smaller smoothing to produce 2-3 bands."""
        x0, y0, x1, y1 = line["bbox"]
        roi = image_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        h = roi.shape[0]
        # Smaller smoothing window for splitting (more sensitive)
        smooth_small = max(15, int(round(0.012 * h)))
        try:
            binv = _to_binary_inv(roi)
            # Suppress very long horizontals only (use parent width proportion: keep as is since _suppress_ruled_lines uses wide kernel)
            mask = _suppress_ruled_lines(binv)
            bands = _bands_from_projection(mask, smooth_small, thr_ratio)
            parts: List[Dict] = []
            for i, (ry0, ry1) in enumerate(bands):
                tx0, ty0, tx1, ty1 = _tighten_band(mask, ry0, ry1)
                # Map back to page coordinates
                bx0 = int(x0 + tx0)
                by0 = int(y0 + ty0)
                bx1 = int(x0 + tx1)
                by1 = int(y0 + ty1)
                if by1 - by0 > max(8, h // 50):
                    parts.append({
                        "bbox": (bx0, by0, bx1, by1),
                        "order": 0,
                        "method": "hybrid",
                        "confidence": float(line.get("confidence", 0.7))
                    })
            # If no meaningful split, return empty to keep original
            return parts if len(parts) >= 2 else []
        except Exception:
            return []

def detect_lines_hybrid(image_bgr: np.ndarray, 
                       use_deskew: bool = True,
                       line_len_frac: float = 0.20,
                       smooth: int = 25,
                       thr_ratio: float = 0.12,
                       kernel_w: int = 45,
                       min_gap: int = 24,
                       min_strength: float = 0.25,
                       enable_singleline_snap: bool = True) -> List[Dict]:
    """
    Convenience function for hybrid line detection.
    Returns list of dicts with bbox, order, confidence, and method info.
    """
    segmenter = HybridLineSegmenter(use_deskew=use_deskew)
    lines = segmenter.detect_lines_hybrid(
        image_bgr, line_len_frac, smooth, thr_ratio, 
        kernel_w, min_gap, min_strength
    )
    if not enable_singleline_snap:
        # If disabled, and we collapsed to exactly 1 full-page line, try to undo by simple split using projection bands
        H, W = image_bgr.shape[:2]
        if len(lines) == 1 and lines[0]["bbox"] == (0, 0, W, H):
            # run only opencv projection to get candidate bands with adaptive smoothing
            binv = _to_binary_inv(image_bgr)
            mask = _suppress_ruled_lines(binv)
            bands = _bands_from_projection(mask, max(25, int(round(0.02 * H))), thr_ratio)
            recovered = []
            for i, (y0, y1) in enumerate(bands):
                x0, y0x, x1, y1x = _tighten_band(mask, y0, y1)
                if y1x - y0x > max(8, H // 100):
                    recovered.append({
                        "bbox": (int(x0), int(y0x), int(x1), int(y1x)),
                        "order": i,
                        "method": "hybrid",
                        "confidence": 0.7,
                        "area": (x1-x0)*(y1x-y0x)
                    })
            if recovered:
                return recovered
    return lines

def crop_lines(image_bgr: np.ndarray, lines: List[Dict]) -> List[np.ndarray]:
    """Crop individual lines from the image."""
    crops = []
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        crop = image_bgr[y0:y1, x0:x1].copy()
        crops.append(crop)
    return crops

def draw_overlays(image_bgr: np.ndarray, lines: List[Dict]) -> np.ndarray:
    """Draw bounding boxes around detected lines with method information."""
    vis = image_bgr.copy()
    
    # Color coding for different methods
    colors = {
        "opencv": (0, 255, 0),      # Green
        "rule_based": (255, 0, 0),  # Red
        "hybrid": (0, 0, 255)       # Blue
    }
    
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        method = line.get("method", "hybrid")
        confidence = line.get("confidence", 0.0)
        color = colors.get(method, (0, 0, 255))
        
        # Draw bounding box
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        
        # Add line number and confidence
        label = f"{line['order']} ({confidence:.2f})"
        cv2.putText(vis, label, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add method indicator
        method_label = method[:3].upper()
        cv2.putText(vis, method_label, (x0+5, y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis
