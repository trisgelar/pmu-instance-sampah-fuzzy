# file: modules/exceptions.py

class WasteDetectionError(Exception):
    """Base exception class for waste detection system."""
    pass

class DatasetError(WasteDetectionError):
    """Exception raised for dataset-related errors."""
    pass

class ModelError(WasteDetectionError):
    """Exception raised for model-related errors."""
    pass

class FuzzyLogicError(WasteDetectionError):
    """Exception raised for fuzzy logic computation errors."""
    pass

class InferenceError(WasteDetectionError):
    """Exception raised for inference-related errors."""
    pass

class ConfigurationError(WasteDetectionError):
    """Exception raised for configuration-related errors."""
    pass

class FileOperationError(WasteDetectionError):
    """Exception raised for file operation errors."""
    pass

class APIError(WasteDetectionError):
    """Exception raised for API-related errors."""
    pass

class ValidationError(WasteDetectionError):
    """Exception raised for validation errors."""
    pass 