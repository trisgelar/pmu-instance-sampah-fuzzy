#!/usr/bin/env python3
"""
Unit tests for the fuzzy area classifier module.
Tests fuzzy logic classification functionality and error handling.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fuzzy_area_classifier import FuzzyAreaClassifier
from modules.exceptions import FuzzyLogicError, ValidationError


class TestFuzzyAreaClassifier(unittest.TestCase):
    """Test cases for FuzzyAreaClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = FuzzyAreaClassifier()

    def test_init_default_config(self):
        """Test FuzzyAreaClassifier initialization with default configuration."""
        classifier = FuzzyAreaClassifier()
        
        # Verify default configuration
        self.assertIn('sedikit', classifier.area_percent_ranges)
        self.assertIn('sedang', classifier.area_percent_ranges)
        self.assertIn('banyak', classifier.area_percent_ranges)
        self.assertEqual(classifier.sedikit_threshold, 33.0)
        self.assertEqual(classifier.sedang_threshold, 66.0)

    def test_init_custom_config(self):
        """Test FuzzyAreaClassifier initialization with custom configuration."""
        custom_config = {
            'area_percent_ranges': {
                'sedikit': [0, 0, 3],
                'sedang': [2, 8, 15],
                'banyak': [12, 100, 100]
            },
            'sedikit_threshold': 25.0,
            'sedang_threshold': 75.0
        }
        
        classifier = FuzzyAreaClassifier(custom_config)
        
        # Verify custom configuration
        self.assertEqual(classifier.area_percent_ranges['sedikit'], [0, 0, 3])
        self.assertEqual(classifier.sedikit_threshold, 25.0)
        self.assertEqual(classifier.sedang_threshold, 75.0)

    def test_validate_input_valid(self):
        """Test input validation with valid values."""
        # Test valid inputs
        valid_inputs = [0, 1, 50, 100, 25.5, 75.8]
        
        for value in valid_inputs:
            try:
                self.classifier._validate_input(value)
            except ValidationError:
                self.fail(f"ValidationError raised for valid input: {value}")

    def test_validate_input_invalid_type(self):
        """Test input validation with invalid types."""
        invalid_inputs = ["string", [], {}, None]
        
        for value in invalid_inputs:
            with self.assertRaises(ValidationError):
                self.classifier._validate_input(value)

    def test_validate_input_negative(self):
        """Test input validation with negative values."""
        with self.assertRaises(ValidationError):
            self.classifier._validate_input(-1)

    def test_validate_input_exceeds_100(self):
        """Test input validation with values exceeding 100."""
        # Should cap to 100 and log warning
        with patch('modules.fuzzy_area_classifier.logger') as mock_logger:
            self.classifier._validate_input(150)
            mock_logger.warning.assert_called_once()

    def test_classify_area_sedikit(self):
        """Test area classification for 'sedikit' category."""
        # Test values that should be classified as 'sedikit'
        sedikit_values = [0, 1, 2, 3, 4, 5]
        
        for value in sedikit_values:
            result = self.classifier.classify_area(value)
            self.assertEqual(result, "sedikit", f"Value {value} should be classified as 'sedikit'")

    def test_classify_area_sedang(self):
        """Test area classification for 'sedang' category."""
        # Test values that should be classified as 'sedang'
        sedang_values = [6, 10, 15, 20]
        
        for value in sedang_values:
            result = self.classifier.classify_area(value)
            self.assertEqual(result, "sedang", f"Value {value} should be classified as 'sedang'")

    def test_classify_area_banyak(self):
        """Test area classification for 'banyak' category."""
        # Test values that should be classified as 'banyak'
        banyak_values = [25, 50, 75, 100]
        
        for value in banyak_values:
            result = self.classifier.classify_area(value)
            self.assertEqual(result, "banyak", f"Value {value} should be classified as 'banyak'")

    def test_classify_area_with_fuzzy_system(self):
        """Test area classification using fuzzy system."""
        # Mock the fuzzy system to return predictable results
        with patch.object(self.classifier, 'area_ctrl') as mock_ctrl:
            mock_ctrl.input = {}
            mock_ctrl.output = {'classification_score': 30.0}  # Should result in 'sedikit'
            
            result = self.classifier.classify_area(10)
            self.assertEqual(result, "sedikit")

    def test_classify_area_fuzzy_system_unavailable(self):
        """Test area classification when fuzzy system is unavailable."""
        # Set area_ctrl to None to simulate fuzzy system unavailability
        self.classifier.area_ctrl = None
        
        with patch('modules.fuzzy_area_classifier.logger') as mock_logger:
            result = self.classifier.classify_area(5)
            
            # Should use fallback classification
            self.assertEqual(result, "sedikit")
            mock_logger.warning.assert_called_once()

    def test_classify_area_fuzzy_computation_error(self):
        """Test area classification when fuzzy computation fails."""
        # Mock fuzzy system to raise ValueError
        with patch.object(self.classifier, 'area_ctrl') as mock_ctrl:
            mock_ctrl.input = {}
            mock_ctrl.compute.side_effect = ValueError("Fuzzy computation error")
            
            with patch('modules.fuzzy_area_classifier.logger') as mock_logger:
                result = self.classifier.classify_area(10)
                
                # Should use fallback classification
                self.assertEqual(result, "sedang")
                mock_logger.error.assert_called_once()
                mock_logger.info.assert_called_once()

    def test_classify_area_unexpected_error(self):
        """Test area classification with unexpected errors."""
        # Mock fuzzy system to raise unexpected error
        with patch.object(self.classifier, 'area_ctrl') as mock_ctrl:
            mock_ctrl.input = {}
            mock_ctrl.compute.side_effect = Exception("Unexpected error")
            
            with self.assertRaises(FuzzyLogicError):
                self.classifier.classify_area(10)

    def test_fallback_classification(self):
        """Test fallback classification method."""
        # Test fallback thresholds
        self.assertEqual(self.classifier._fallback_classification(0), "sedikit")
        self.assertEqual(self.classifier._fallback_classification(5), "sedang")
        self.assertEqual(self.classifier._fallback_classification(15), "banyak")

    def test_determine_classification(self):
        """Test classification determination based on fuzzy score."""
        # Test different fuzzy scores
        self.assertEqual(self.classifier._determine_classification(20), "sedikit")
        self.assertEqual(self.classifier._determine_classification(50), "sedang")
        self.assertEqual(self.classifier._determine_classification(80), "banyak")

    def test_get_membership_functions(self):
        """Test getting membership function parameters."""
        result = self.classifier.get_membership_functions()
        
        # Verify result structure
        self.assertIn('area_percent', result)
        self.assertIn('classification_score', result)
        self.assertIn('thresholds', result)
        self.assertIn('fallback_thresholds', result)

    def test_get_membership_functions_fuzzy_system_unavailable(self):
        """Test getting membership functions when fuzzy system is unavailable."""
        self.classifier.area_ctrl = None
        
        result = self.classifier.get_membership_functions()
        self.assertIn('error', result)

    def test_update_membership_functions(self):
        """Test updating membership function parameters."""
        new_area_params = {'sedikit': [0, 0, 3]}
        new_score_params = {'low': [0, 0, 25]}
        new_thresholds = {'sedikit': 25.0}
        
        success = self.classifier.update_membership_functions(
            new_area_params, new_score_params, new_thresholds
        )
        
        self.assertTrue(success)
        self.assertEqual(self.classifier.area_percent_ranges['sedikit'], [0, 0, 3])
        self.assertEqual(self.classifier.classification_score_ranges['low'], [0, 0, 25])
        self.assertEqual(self.classifier.sedikit_threshold, 25.0)

    def test_update_membership_functions_failure(self):
        """Test updating membership functions when it fails."""
        # Mock _setup_fuzzy_system to raise an exception
        with patch.object(self.classifier, '_setup_fuzzy_system', side_effect=Exception("Setup failed")):
            success = self.classifier.update_membership_functions({}, {}, {})
            self.assertFalse(success)

    def test_get_configuration(self):
        """Test getting current configuration."""
        config = self.classifier.get_configuration()
        
        # Verify configuration structure
        self.assertIn('area_percent_ranges', config)
        self.assertIn('classification_score_ranges', config)
        self.assertIn('sedikit_threshold', config)
        self.assertIn('sedang_threshold', config)
        self.assertIn('fallback_sedikit_threshold', config)
        self.assertIn('fallback_sedang_threshold', config)

    def test_setup_fuzzy_system(self):
        """Test fuzzy system setup."""
        # Test that fuzzy system is created successfully
        fuzzy_system = self.classifier._setup_fuzzy_system()
        self.assertIsNotNone(fuzzy_system)

    def test_setup_fuzzy_system_failure(self):
        """Test fuzzy system setup failure."""
        # Mock skfuzzy to raise an exception
        with patch('modules.fuzzy_area_classifier.fuzz') as mock_fuzz:
            mock_fuzz.trimf.side_effect = Exception("Fuzzy setup failed")
            
            with self.assertRaises(FuzzyLogicError):
                self.classifier._setup_fuzzy_system()

    def test_setup_configuration(self):
        """Test configuration setup."""
        custom_config = {
            'area_percent_ranges': {'sedikit': [0, 0, 3]},
            'sedikit_threshold': 25.0
        }
        
        self.classifier._setup_configuration(custom_config)
        
        # Verify custom configuration was applied
        self.assertEqual(self.classifier.area_percent_ranges['sedikit'], [0, 0, 3])
        self.assertEqual(self.classifier.sedikit_threshold, 25.0)

    def test_setup_configuration_partial(self):
        """Test configuration setup with partial configuration."""
        partial_config = {
            'sedikit_threshold': 30.0
        }
        
        self.classifier._setup_configuration(partial_config)
        
        # Verify only specified values were changed
        self.assertEqual(self.classifier.sedikit_threshold, 30.0)
        # Default values should remain unchanged
        self.assertEqual(self.classifier.sedang_threshold, 66.0)

    def test_classify_area_edge_cases(self):
        """Test area classification with edge cases."""
        edge_cases = [
            (0, "sedikit"),
            (100, "banyak"),
            (33.0, "sedikit"),
            (66.0, "sedang"),
            (99.9, "banyak")
        ]
        
        for value, expected in edge_cases:
            result = self.classifier.classify_area(value)
            self.assertEqual(result, expected, f"Value {value} should be classified as '{expected}'")

    def test_classify_area_boundary_values(self):
        """Test area classification with boundary values."""
        # Test values at classification boundaries
        boundaries = [
            (self.classifier.fallback_sedikit_threshold - 0.1, "sedikit"),
            (self.classifier.fallback_sedikit_threshold, "sedang"),
            (self.classifier.fallback_sedang_threshold - 0.1, "sedang"),
            (self.classifier.fallback_sedang_threshold, "banyak")
        ]
        
        for value, expected in boundaries:
            result = self.classifier.classify_area(value)
            self.assertEqual(result, expected, f"Boundary value {value} should be classified as '{expected}'")

    def test_classify_area_float_values(self):
        """Test area classification with float values."""
        float_values = [0.5, 1.7, 10.3, 25.8, 50.2, 75.9, 99.1]
        
        for value in float_values:
            try:
                result = self.classifier.classify_area(value)
                self.assertIn(result, ["sedikit", "sedang", "banyak"])
            except Exception as e:
                self.fail(f"Classification failed for float value {value}: {e}")

    def test_classify_area_high_precision(self):
        """Test area classification with high precision values."""
        high_precision_values = [0.001, 1.234, 10.567, 25.890, 50.123, 75.456, 99.999]
        
        for value in high_precision_values:
            try:
                result = self.classifier.classify_area(value)
                self.assertIn(result, ["sedikit", "sedang", "banyak"])
            except Exception as e:
                self.fail(f"Classification failed for high precision value {value}: {e}")


class TestFuzzyAreaClassifierIntegration(unittest.TestCase):
    """Integration tests for FuzzyAreaClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = FuzzyAreaClassifier()

    def test_full_classification_pipeline(self):
        """Test the complete classification pipeline."""
        test_cases = [
            (0, "sedikit"),
            (5, "sedang"),
            (15, "banyak"),
            (50, "banyak"),
            (100, "banyak")
        ]
        
        for input_value, expected_output in test_cases:
            result = self.classifier.classify_area(input_value)
            self.assertEqual(result, expected_output, 
                           f"Input {input_value} should produce '{expected_output}', got '{result}'")

    def test_configuration_persistence(self):
        """Test that configuration changes persist."""
        # Get initial configuration
        initial_config = self.classifier.get_configuration()
        
        # Update configuration
        new_threshold = 40.0
        self.classifier.sedikit_threshold = new_threshold
        
        # Verify change persisted
        updated_config = self.classifier.get_configuration()
        self.assertEqual(updated_config['sedikit_threshold'], new_threshold)
        self.assertNotEqual(initial_config['sedikit_threshold'], new_threshold)

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test with invalid input that should trigger fallback
        with patch.object(self.classifier, 'area_ctrl') as mock_ctrl:
            mock_ctrl.input = {}
            mock_ctrl.compute.side_effect = ValueError("Simulated error")
            
            # Should not crash and should use fallback
            result = self.classifier.classify_area(10)
            self.assertIn(result, ["sedikit", "sedang", "banyak"])

    def test_performance_with_large_inputs(self):
        """Test performance with large number of inputs."""
        import time
        
        # Test classification speed
        start_time = time.time()
        
        for i in range(1000):
            self.classifier.classify_area(i % 101)  # Values 0-100
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second)
        self.assertLess(duration, 1.0, f"Classification took too long: {duration:.2f} seconds")


if __name__ == '__main__':
    unittest.main() 