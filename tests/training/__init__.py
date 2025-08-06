"""
Training Test Module

This module contains enhanced training systems for testing purposes.
"""

# Import simple training system (no external dependencies)
try:
    from .simple_enhanced_training import SimpleEnhancedTrainingSystem
    SIMPLE_AVAILABLE = True
except ImportError as e:
    SimpleEnhancedTrainingSystem = None
    SIMPLE_AVAILABLE = False

# Import enhanced training system (has external dependencies)
try:
    from .enhanced_training_system import EnhancedTrainingSystem
    ENHANCED_AVAILABLE = True
except ImportError as e:
    EnhancedTrainingSystem = None
    ENHANCED_AVAILABLE = False

# Define available exports
__all__ = []

if SIMPLE_AVAILABLE:
    __all__.append('SimpleEnhancedTrainingSystem')

if ENHANCED_AVAILABLE:
    __all__.append('EnhancedTrainingSystem') 