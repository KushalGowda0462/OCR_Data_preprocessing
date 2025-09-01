# src/recognition/trocr_recognizer.py
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2

class TrOCRRecognizer:
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = None,
        max_length: int = 256,
        num_beams: int = 1,         # 1 = greedy (fast/deterministic)
        use_fast: bool = True       # try to avoid the slow-processor warning
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use fast image processor if available; falls back silently
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.num_beams = num_beams

        # Optional: half-precision on CUDA for speed/memory (safe no-op on CPU)
        self._fp16 = False
        if self.device == "cuda":
            try:
                self.model.half()
                self._fp16 = True
            except Exception:
                self._fp16 = False  # stay in fp32 if half() not supported

    def recognize_line(self, img_bgr) -> str:
        """Takes a single line crop (OpenCV BGR) → returns recognized text string"""
        if img_bgr is None or img_bgr.size == 0:
            return ""
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                do_sample=False
            )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # normalize whitespace a bit
        return " ".join(text.strip().split())

    # Optional speed-up: batch a list of crops in one forward pass.
    # You don't have to use this; recognize_line() above remains unchanged.
    def recognize_lines(self, crops_bgr) -> list[str]:
        if not crops_bgr:
            return []
        pil_list = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = self.processor(images=pil_list, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                do_sample=False
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [" ".join(t.strip().split()) for t in texts]

    @staticmethod
    def draw_overlay(img, line_texts):
        """
        Draw green boxes with red text labels just above each box.
        line_texts: list of dicts like {"order": i, "bbox": (x0,y0,x1,y1), "text": "..."}
        """
        vis = img.copy()
        for ln in line_texts:
            (x0, y0, x1, y1) = ln["bbox"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = (ln.get("text") or "").strip()
            if len(label) > 60:
                label = label[:60] + "…"
            y_text = max(12, y0 - 6)
            try:
                cv2.putText(vis, label, (x0, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            except cv2.error:
                # OpenCV's putText is ASCII-oriented; fall back to ASCII-only if needed
                safe = label.encode("ascii", "ignore").decode("ascii")
                cv2.putText(vis, safe, (x0, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return vis












# # src/recognition/trocr_recognizer.py
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import torch
# import cv2

# class TrOCRRecognizer:
#     def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: str = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.processor = TrOCRProcessor.from_pretrained(model_name)
#         self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#     def recognize_line(self, img_bgr) -> str:
#         """Takes a single line crop (OpenCV BGR) → returns recognized text string"""
#         pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#         pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.device)
#         with torch.no_grad():
#             generated_ids = self.model.generate(pixel_values, max_length=256)
#         text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         return text.strip()

#     def draw_overlay(img, line_texts):
#     vis = img.copy()
#     for ln in line_texts:
#         (x0, y0, x1, y1) = ln["bbox"]
#         cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
#         # put text above the box (truncate if too long)
#         label = ln["text"][:40] + ("…" if len(ln["text"]) > 40 else "")
#         cv2.putText(vis, label, (x0, max(0, y0 - 5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
#     return vis
