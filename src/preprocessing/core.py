# src/preprocessing/core.py
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from .models import PreprocessingConfig, InkColor, PreprocessingError

logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """Main class for document preprocessing operations"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.line_kernel_width % 2 == 0:
            self.config.line_kernel_width += 1
        if self.config.adaptive_block_size % 2 == 0:
            self.config.adaptive_block_size += 1
    
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Main preprocessing pipeline for document images"""
        try:
            original_shape = image_array.shape
            processed = image_array.copy()
            
            # Step 1: Deskew the image
            if self.config.deskew_enabled:
                processed = self._deskew_image(processed)
            
            # Step 2: Improve illumination and contrast
            processed = self._enhance_contrast(processed)
            
            # Step 3: Isolate ink (blue or black)
            ink_mask = self._isolate_ink(processed)
            
            # Step 4: Remove ruled lines if requested
            if self.config.line_removal_enabled:
                processed = self._remove_ruled_lines(processed, ink_mask)
            else:
                processed = cv2.bitwise_and(processed, processed, mask=ink_mask)
            
            # Step 5: Binarize and thicken strokes
            processed = self._binarize_and_thicken(processed)
            
            logger.info(f"Preprocessing completed. Original shape: {original_shape}, Processed shape: {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise PreprocessingError(f"Image preprocessing failed: {str(e)}")
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image skew using Hough line detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        lines = cv2.HoughLinesP(
            binary, 1, np.pi/180, threshold=100, 
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None or len(lines) < 5:
            logger.warning("Not enough lines detected for deskewing")
            return image
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        angles = np.array(angles)
        median_angle = np.median(angles)
        std_angle = np.std(angles)
        filtered_angles = angles[(angles >= median_angle - 2*std_angle) & 
                                (angles <= median_angle + 2*std_angle)]
        
        if len(filtered_angles) == 0:
            return image
        
        median_skew = np.median(filtered_angles)
        if abs(median_skew) > self.config.max_deskew_angle:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_skew, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (width, height), 
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.info(f"Deskewed image by {median_skew:.2f} degrees")
        return rotated
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE and mild sharpening to enhance contrast"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
        return denoised
    
    def _isolate_ink(self, image: np.ndarray) -> np.ndarray:
        """Create a mask for blue or black ink regions"""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            blue_lower = np.array([self.config.blue_hue_range[0], 
                                 self.config.blue_sat_threshold, 
                                 self.config.blue_val_threshold])
            blue_upper = np.array([self.config.blue_hue_range[1], 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            _, _, v = cv2.split(hsv)
            black_mask = cv2.threshold(v, self.config.black_val_threshold, 255, 
                                     cv2.THRESH_BINARY_INV)[1]
            
            ink_mask = cv2.bitwise_or(blue_mask, black_mask)
        else:
            _, ink_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        
        return ink_mask
    
    def _remove_ruled_lines(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove horizontal ruled lines using morphological operations"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.config.line_kernel_width, self.config.line_kernel_height)
        )
        
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        result = cv2.inpaint(
            image, lines, self.config.line_inpaint_radius, 
            cv2.INPAINT_TELEA
        )
        
        return result
    
    def _binarize_and_thicken(self, image: np.ndarray) -> np.ndarray:
        """Binarize image and thicken faint strokes"""
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.config.adaptive_block_size, 
            self.config.adaptive_c
        )
        
        kernel = np.ones((self.config.close_kernel_size, 
                         self.config.close_kernel_size), np.uint8)
        thickened = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, 
            iterations=self.config.thicken_iterations
        )
        
        return 255 - thickened

    def detect_ink_color(self, image: np.ndarray) -> InkColor:
        """Detect the predominant ink color in the document"""
        if len(image.shape) != 3:
            return InkColor.UNKNOWN
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        blue_lower = np.array([self.config.blue_hue_range[0], 
                             self.config.blue_sat_threshold, 
                             self.config.blue_val_threshold])
        blue_upper = np.array([self.config.blue_hue_range[1], 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        _, _, v = cv2.split(hsv)
        black_mask = cv2.threshold(v, self.config.black_val_threshold, 255, 
                                 cv2.THRESH_BINARY_INV)[1]
        
        blue_pixels = np.count_nonzero(blue_mask)
        black_pixels = np.count_nonzero(black_mask)
        
        if blue_pixels > black_pixels * 1.5:
            return InkColor.BLUE
        elif black_pixels > blue_pixels * 1.5:
            return InkColor.BLACK
        else:
            return InkColor.UNKNOWN