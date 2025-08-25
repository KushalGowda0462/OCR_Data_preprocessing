# src/preprocessing/models.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import numpy as np

class PreprocessingError(Exception):
    """Custom exception for preprocessing failures"""
    pass

class InkColor(Enum):
    BLUE = "blue"
    BLACK = "black"
    UNKNOWN = "unknown"

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations"""
    # Deskew parameters
    deskew_enabled: bool = True
    max_deskew_angle: float = 15.0
    
    # Ink isolation parameters
    blue_hue_range: Tuple[int, int] = (90, 140)
    blue_sat_threshold: int = 30
    blue_val_threshold: int = 30
    black_val_threshold: int = 50
    
    # Line removal parameters
    line_removal_enabled: bool = True
    line_kernel_width: int = 51
    line_kernel_height: int = 1
    line_inpaint_radius: int = 3
    
    # Morphology parameters
    close_kernel_size: int = 3
    thicken_iterations: int = 1
    
    # Adaptive threshold parameters
    adaptive_block_size: int = 31
    adaptive_c: int = 15

@dataclass
class ProcessingResult:
    """Result of document preprocessing"""
    document_id: str
    processing_time: float
    ink_color: InkColor
    output_path: str
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    status: str = "success"
    error_message: Optional[str] = None