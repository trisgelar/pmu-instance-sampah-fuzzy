"""
Validation Module

This module provides tools for validating models, data, and configurations.
It includes validators for model files, dataset formats, and configuration files.
"""

# Import validation modules
try:
    from .model_validator import ModelValidator
    MODEL_VALIDATOR_AVAILABLE = True
except ImportError as e:
    ModelValidator = None
    MODEL_VALIDATOR_AVAILABLE = False

try:
    from .data_validator import DataValidator
    DATA_VALIDATOR_AVAILABLE = True
except ImportError as e:
    DataValidator = None
    DATA_VALIDATOR_AVAILABLE = False

try:
    from .config_validator import ConfigValidator
    CONFIG_VALIDATOR_AVAILABLE = True
except ImportError as e:
    ConfigValidator = None
    CONFIG_VALIDATOR_AVAILABLE = False

# Define available exports
__all__ = []

if MODEL_VALIDATOR_AVAILABLE:
    __all__.append('ModelValidator')

if DATA_VALIDATOR_AVAILABLE:
    __all__.append('DataValidator')

if CONFIG_VALIDATOR_AVAILABLE:
    __all__.append('ConfigValidator')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 