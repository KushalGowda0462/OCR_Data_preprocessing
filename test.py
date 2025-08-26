# scratch_test.py
import cv2, json
from src.preprocessing.core import preprocess_safe
from src.segmentation.lines_kraken import detect_lines as kraken_lines
from src.segmentation.lines_mmocr import detect_lines as mmocr_lines

bgr = cv2.imread("your_postprocessed_or_original.jpg")
binv = preprocess_safe(bgr)
cv2.imwrite("binv_debug.png", binv)

print("Kraken:", len(kraken_lines(bgr)))
print("MMOCR :", len(mmocr_lines(bgr)))
