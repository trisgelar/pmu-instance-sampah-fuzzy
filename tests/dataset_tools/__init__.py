"""
Dataset Tools Module

This module contains tools for diagnosing and fixing dataset issues,
particularly for YOLO training with COCO annotations.

Tools:
- diagnose_dataset.py: Diagnose dataset configuration issues
- fix_dataset_classes.py: Fix class configuration issues manually
- fix_dataset_ultralytics.py: Fix using Ultralytics conversion tools
- dataset_validator.py: Validate dataset structure and format
- extract_and_check_dataset.py: Extract and verify dataset structure
- final_verification.py: Final dataset verification for YOLO training
- fix_yolo_coordinates.py: Fix coordinate normalization issues
- check_segmentation_format.py: Check dataset format for segmentation
- fix_segmentation_labels.py: Convert labels to segmentation format
"""

from .diagnose_dataset import diagnose_dataset
from .fix_dataset_classes import fix_dataset_classes
from .fix_dataset_ultralytics import fix_dataset_ultralytics
from .dataset_validator import validate_dataset
from .extract_and_check_dataset import extract_and_check_dataset
from .final_verification import final_verification
from .fix_yolo_coordinates import fix_yolo_coordinates
from .check_segmentation_format import check_segmentation_format
from .fix_segmentation_labels import convert_to_segmentation_labels
from .test_segmentation_integration import test_segmentation_integration

__all__ = [
    'diagnose_dataset',
    'fix_dataset_classes', 
    'fix_dataset_ultralytics',
    'validate_dataset',
    'extract_and_check_dataset',
    'final_verification',
    'fix_yolo_coordinates',
    'check_segmentation_format',
    'convert_to_segmentation_labels',
    'test_segmentation_integration'
] 