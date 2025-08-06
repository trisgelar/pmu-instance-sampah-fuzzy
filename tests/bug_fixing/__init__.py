"""
Bug Fixing Module

This module provides tools for bug detection, fixing, and validation.
It includes detectors for common bugs, automatic fixers, and validation tools.
"""

# Import bug fixing modules
try:
    from .bug_detector import BugDetector
    BUG_DETECTOR_AVAILABLE = True
except ImportError as e:
    BugDetector = None
    BUG_DETECTOR_AVAILABLE = False

try:
    from .bug_fixer import BugFixer
    BUG_FIXER_AVAILABLE = True
except ImportError as e:
    BugFixer = None
    BUG_FIXER_AVAILABLE = False

try:
    from .bug_validator import BugValidator
    BUG_VALIDATOR_AVAILABLE = True
except ImportError as e:
    BugValidator = None
    BUG_VALIDATOR_AVAILABLE = False

# Define available exports
__all__ = []

if BUG_DETECTOR_AVAILABLE:
    __all__.append('BugDetector')

if BUG_FIXER_AVAILABLE:
    __all__.append('BugFixer')

if BUG_VALIDATOR_AVAILABLE:
    __all__.append('BugValidator')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 