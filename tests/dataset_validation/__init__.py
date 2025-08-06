"""
Dataset Validation Module

This module provides tools for dataset validation and verification.
"""

# Import dataset validation modules
try:
    from .dataset_validator import DatasetValidator
    DATASET_VALIDATOR_AVAILABLE = True
except ImportError as e:
    DatasetValidator = None
    DATASET_VALIDATOR_AVAILABLE = False

try:
    from .final_verification import FinalVerification
    FINAL_VERIFICATION_AVAILABLE = True
except ImportError as e:
    FinalVerification = None
    FINAL_VERIFICATION_AVAILABLE = False

# Define available exports
__all__ = []

if DATASET_VALIDATOR_AVAILABLE:
    __all__.append('DatasetValidator')

if FINAL_VERIFICATION_AVAILABLE:
    __all__.append('FinalVerification')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 