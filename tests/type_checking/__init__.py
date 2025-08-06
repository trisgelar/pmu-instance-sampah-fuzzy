"""
Type Checking Module

This module provides tools for type validation, checking, and fixing.
It includes validators for data types, type compatibility checks, and type fixing utilities.
"""

# Import type checking modules
try:
    from .type_validator import TypeValidator
    TYPE_VALIDATOR_AVAILABLE = True
except ImportError as e:
    TypeValidator = None
    TYPE_VALIDATOR_AVAILABLE = False

try:
    from .type_fixer import TypeFixer
    TYPE_FIXER_AVAILABLE = True
except ImportError as e:
    TypeFixer = None
    TYPE_FIXER_AVAILABLE = False

try:
    from .type_checker import TypeChecker
    TYPE_CHECKER_AVAILABLE = True
except ImportError as e:
    TypeChecker = None
    TYPE_CHECKER_AVAILABLE = False

# Define available exports
__all__ = []

if TYPE_VALIDATOR_AVAILABLE:
    __all__.append('TypeValidator')

if TYPE_FIXER_AVAILABLE:
    __all__.append('TypeFixer')

if TYPE_CHECKER_AVAILABLE:
    __all__.append('TypeChecker')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 