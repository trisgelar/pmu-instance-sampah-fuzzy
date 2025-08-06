"""
Training Tools Module

This module provides training systems and tools for the waste detection system.
"""

# Import training tool modules
try:
    from .enhanced_training_system import EnhancedTrainingSystem
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError as e:
    EnhancedTrainingSystem = None
    ENHANCED_TRAINING_AVAILABLE = False

try:
    from .simple_enhanced_training import SimpleEnhancedTrainingSystem
    SIMPLE_TRAINING_AVAILABLE = True
except ImportError as e:
    SimpleEnhancedTrainingSystem = None
    SIMPLE_TRAINING_AVAILABLE = False

try:
    from .test_training_imports import TestTrainingImports
    TRAINING_IMPORTS_AVAILABLE = True
except ImportError as e:
    TestTrainingImports = None
    TRAINING_IMPORTS_AVAILABLE = False

# Define available exports
__all__ = []

if ENHANCED_TRAINING_AVAILABLE:
    __all__.append('EnhancedTrainingSystem')

if SIMPLE_TRAINING_AVAILABLE:
    __all__.append('SimpleEnhancedTrainingSystem')

if TRAINING_IMPORTS_AVAILABLE:
    __all__.append('TestTrainingImports')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 