"""
ONNX Testing Module

This module provides structured testing for ONNX models and conversions.
It includes environment checks, model validation, conversion testing, and inference testing.
"""

# Import ONNX testing modules
try:
    from .check_onnx_environment import ONNXEnvironmentChecker
    ONNX_ENV_AVAILABLE = True
except ImportError as e:
    ONNXEnvironmentChecker = None
    ONNX_ENV_AVAILABLE = False

try:
    from .check_onnx_models import ONNXModelChecker
    ONNX_MODEL_AVAILABLE = True
except ImportError as e:
    ONNXModelChecker = None
    ONNX_MODEL_AVAILABLE = False

try:
    from .check_onnx_conversion import ONNXConversionChecker
    ONNX_CONVERSION_AVAILABLE = True
except ImportError as e:
    ONNXConversionChecker = None
    ONNX_CONVERSION_AVAILABLE = False

try:
    from .check_onnx_inference import ONNXInferenceChecker
    ONNX_INFERENCE_AVAILABLE = True
except ImportError as e:
    ONNXInferenceChecker = None
    ONNX_INFERENCE_AVAILABLE = False

try:
    from .check_onnx_rknn_environment import ONNXRKNNEnvironmentChecker
    ONNX_RKNN_ENV_AVAILABLE = True
except ImportError as e:
    ONNXRKNNEnvironmentChecker = None
    ONNX_RKNN_ENV_AVAILABLE = False

# Define available exports
__all__ = []

if ONNX_ENV_AVAILABLE:
    __all__.append('ONNXEnvironmentChecker')

if ONNX_MODEL_AVAILABLE:
    __all__.append('ONNXModelChecker')

if ONNX_CONVERSION_AVAILABLE:
    __all__.append('ONNXConversionChecker')

if ONNX_INFERENCE_AVAILABLE:
    __all__.append('ONNXInferenceChecker')

if ONNX_RKNN_ENV_AVAILABLE:
    __all__.append('ONNXRKNNEnvironmentChecker')

# Version info
__version__ = "1.0.0"
__author__ = "PMU Instance Sampah Fuzzy Team" 