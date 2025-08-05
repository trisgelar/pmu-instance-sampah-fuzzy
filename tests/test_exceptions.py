#!/usr/bin/env python3
"""
Unit tests for the custom exception hierarchy.
Tests exception inheritance and error handling.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.exceptions import (
    WasteDetectionError, DatasetError, ModelError, FuzzyLogicError,
    InferenceError, ConfigurationError, FileOperationError, APIError, ValidationError
)


class TestExceptionHierarchy(unittest.TestCase):
    """Test cases for exception hierarchy."""

    def test_base_exception_inheritance(self):
        """Test that all exceptions inherit from WasteDetectionError."""
        exceptions = [
            DatasetError,
            ModelError,
            FuzzyLogicError,
            InferenceError,
            ConfigurationError,
            FileOperationError,
            APIError,
            ValidationError
        ]
        
        for exception_class in exceptions:
            self.assertTrue(issubclass(exception_class, WasteDetectionError))

    def test_waste_detection_error_inheritance(self):
        """Test that WasteDetectionError inherits from Exception."""
        self.assertTrue(issubclass(WasteDetectionError, Exception))

    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated with messages."""
        test_message = "Test error message"
        
        exceptions = [
            WasteDetectionError(test_message),
            DatasetError(test_message),
            ModelError(test_message),
            FuzzyLogicError(test_message),
            InferenceError(test_message),
            ConfigurationError(test_message),
            FileOperationError(test_message),
            APIError(test_message),
            ValidationError(test_message)
        ]
        
        for exception in exceptions:
            self.assertEqual(str(exception), test_message)

    def test_exception_without_message(self):
        """Test that exceptions can be instantiated without messages."""
        exceptions = [
            WasteDetectionError(),
            DatasetError(),
            ModelError(),
            FuzzyLogicError(),
            InferenceError(),
            ConfigurationError(),
            FileOperationError(),
            APIError(),
            ValidationError()
        ]
        
        for exception in exceptions:
            self.assertIsInstance(exception, Exception)

    def test_exception_with_cause(self):
        """Test that exceptions can be raised with a cause."""
        original_error = ValueError("Original error")
        
        try:
            raise DatasetError("Dataset error") from original_error
        except DatasetError as e:
            self.assertEqual(str(e), "Dataset error")
            self.assertEqual(e.__cause__, original_error)

    def test_exception_chaining(self):
        """Test exception chaining functionality."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise DatasetError("Outer error") from e
        except DatasetError as e:
            self.assertEqual(str(e), "Outer error")
            self.assertIsInstance(e.__cause__, ValueError)

    def test_exception_types(self):
        """Test that exceptions are of correct types."""
        # Test base exception
        base_exception = WasteDetectionError("Base error")
        self.assertIsInstance(base_exception, WasteDetectionError)
        self.assertIsInstance(base_exception, Exception)
        
        # Test specific exceptions
        dataset_exception = DatasetError("Dataset error")
        self.assertIsInstance(dataset_exception, DatasetError)
        self.assertIsInstance(dataset_exception, WasteDetectionError)
        
        model_exception = ModelError("Model error")
        self.assertIsInstance(model_exception, ModelError)
        self.assertIsInstance(model_exception, WasteDetectionError)

    def test_exception_docstrings(self):
        """Test that exceptions have proper docstrings."""
        exceptions = [
            WasteDetectionError,
            DatasetError,
            ModelError,
            FuzzyLogicError,
            InferenceError,
            ConfigurationError,
            FileOperationError,
            APIError,
            ValidationError
        ]
        
        for exception_class in exceptions:
            self.assertIsNotNone(exception_class.__doc__)
            self.assertGreater(len(exception_class.__doc__), 0)

    def test_exception_equality(self):
        """Test exception equality."""
        error1 = DatasetError("Same message")
        error2 = DatasetError("Same message")
        error3 = DatasetError("Different message")
        
        # Same message should be equal
        self.assertEqual(str(error1), str(error2))
        # Different messages should not be equal
        self.assertNotEqual(str(error1), str(error3))

    def test_exception_repr(self):
        """Test exception string representation."""
        message = "Test error message"
        exception = DatasetError(message)
        
        # Test string representation
        self.assertIn(message, str(exception))
        self.assertIn("DatasetError", repr(exception))

    def test_exception_inheritance_chain(self):
        """Test the complete inheritance chain."""
        # Create an instance of each exception type
        exceptions = [
            WasteDetectionError("test"),
            DatasetError("test"),
            ModelError("test"),
            FuzzyLogicError("test"),
            InferenceError("test"),
            ConfigurationError("test"),
            FileOperationError("test"),
            APIError("test"),
            ValidationError("test")
        ]
        
        for exception in exceptions:
            # All should be instances of the base exception
            self.assertIsInstance(exception, WasteDetectionError)
            # All should be instances of Exception
            self.assertIsInstance(exception, Exception)
            # All should be instances of BaseException
            self.assertIsInstance(exception, BaseException)

    def test_exception_with_formatting(self):
        """Test exceptions with formatted messages."""
        value = 42
        message = f"Error with value: {value}"
        
        exception = ModelError(message)
        self.assertEqual(str(exception), message)

    def test_exception_with_multiple_arguments(self):
        """Test exceptions with multiple arguments."""
        try:
            raise DatasetError("Error", "Additional", "Arguments")
        except DatasetError as e:
            # Should handle multiple arguments gracefully
            self.assertIsInstance(e, DatasetError)

    def test_exception_context(self):
        """Test exception context preservation."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError:
                raise DatasetError("Outer error")
        except DatasetError as e:
            # Should have context from the original ValueError
            self.assertIsNotNone(e.__context__)

    def test_exception_traceback(self):
        """Test that exceptions have proper traceback."""
        try:
            raise ModelError("Test error")
        except ModelError as e:
            # Should have traceback information
            self.assertIsNotNone(e.__traceback__)

    def test_exception_attributes(self):
        """Test that exceptions have proper attributes."""
        message = "Test message"
        exception = ConfigurationError(message)
        
        # Test basic attributes
        self.assertEqual(exception.args, (message,))
        self.assertEqual(str(exception), message)

    def test_exception_subclassing(self):
        """Test that exceptions can be subclassed."""
        class CustomDatasetError(DatasetError):
            """Custom dataset error."""
            pass
        
        custom_error = CustomDatasetError("Custom error")
        self.assertIsInstance(custom_error, CustomDatasetError)
        self.assertIsInstance(custom_error, DatasetError)
        self.assertIsInstance(custom_error, WasteDetectionError)

    def test_exception_with_none_message(self):
        """Test exceptions with None message."""
        exception = ValidationError(None)
        self.assertEqual(str(exception), "None")

    def test_exception_with_empty_message(self):
        """Test exceptions with empty message."""
        exception = APIError("")
        self.assertEqual(str(exception), "")

    def test_exception_with_unicode_message(self):
        """Test exceptions with unicode messages."""
        unicode_message = "Erro de teste com caracteres especiais: áéíóú"
        exception = FileOperationError(unicode_message)
        self.assertEqual(str(exception), unicode_message)

    def test_exception_with_long_message(self):
        """Test exceptions with very long messages."""
        long_message = "A" * 1000
        exception = InferenceError(long_message)
        self.assertEqual(str(exception), long_message)

    def test_exception_with_special_characters(self):
        """Test exceptions with special characters."""
        special_message = "Error with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        exception = FuzzyLogicError(special_message)
        self.assertEqual(str(exception), special_message)


class TestExceptionUsage(unittest.TestCase):
    """Test cases for exception usage patterns."""

    def test_raise_and_catch(self):
        """Test raising and catching exceptions."""
        try:
            raise DatasetError("Test error")
        except DatasetError as e:
            self.assertEqual(str(e), "Test error")
        else:
            self.fail("Exception should have been raised")

    def test_exception_not_raised(self):
        """Test that exception is not raised when not expected."""
        try:
            # Do something that doesn't raise an exception
            pass
        except DatasetError:
            self.fail("DatasetError should not have been raised")

    def test_multiple_exception_types(self):
        """Test handling multiple exception types."""
        exceptions = [
            DatasetError("Dataset error"),
            ModelError("Model error"),
            ConfigurationError("Configuration error")
        ]
        
        for exception in exceptions:
            try:
                raise exception
            except WasteDetectionError as e:
                self.assertIsInstance(e, type(exception))

    def test_exception_in_function(self):
        """Test exceptions raised in functions."""
        def function_that_raises():
            raise ModelError("Function error")
        
        try:
            function_that_raises()
        except ModelError as e:
            self.assertEqual(str(e), "Function error")

    def test_exception_in_class_method(self):
        """Test exceptions raised in class methods."""
        class TestClass:
            def method_that_raises(self):
                raise ValidationError("Method error")
        
        obj = TestClass()
        try:
            obj.method_that_raises()
        except ValidationError as e:
            self.assertEqual(str(e), "Method error")

    def test_exception_with_context_manager(self):
        """Test exceptions with context managers."""
        class TestContextManager:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    raise ConfigurationError("Context error") from exc_val
        
        with self.assertRaises(ConfigurationError):
            with TestContextManager():
                raise ValueError("Original error")

    def test_exception_propagation(self):
        """Test exception propagation through multiple levels."""
        def level3():
            raise APIError("Level 3 error")
        
        def level2():
            try:
                level3()
            except APIError as e:
                raise FileOperationError("Level 2 error") from e
        
        def level1():
            try:
                level2()
            except FileOperationError as e:
                raise InferenceError("Level 1 error") from e
        
        try:
            level1()
        except InferenceError as e:
            self.assertEqual(str(e), "Level 1 error")
            self.assertIsInstance(e.__cause__, FileOperationError)
            self.assertIsInstance(e.__cause__.__cause__, APIError)


if __name__ == '__main__':
    unittest.main() 