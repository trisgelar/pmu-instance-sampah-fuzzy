#!/usr/bin/env python3
"""
Unit tests for the configuration manager module.
Tests configuration loading, validation, and management functionality.
"""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config_manager import (
    ConfigManager, ModelConfig, DatasetConfig, FuzzyConfig, 
    LoggingConfig, SystemConfig, Environment
)
from modules.exceptions import ConfigurationError, ValidationError


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Sample valid configuration
        self.valid_config = {
            'model': {
                'supported_versions': ['v8n', 'v10n', 'v11n'],
                'default_epochs': 50,
                'default_batch_size': 16,
                'default_img_size': [640, 640],
                'default_conf_threshold': 0.25
            },
            'dataset': {
                'roboflow_project': 'test_project',
                'roboflow_version': '1',
                'default_dataset_dir': 'datasets',
                'default_model_dir': 'runs'
            },
            'fuzzy': {
                'area_percent_ranges': {
                    'sedikit': [0, 0, 5],
                    'sedang': [3, 10, 20],
                    'banyak': [15, 100, 100]
                },
                'sedikit_threshold': 33.0,
                'sedang_threshold': 66.0
            },
            'logging': {
                'level': 'INFO',
                'file_logging': True,
                'console_logging': True
            },
            'system': {
                'environment': 'colab',
                'num_workers': 4,
                'request_timeout': 300
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_init_with_default_config(self):
        """Test ConfigManager initialization with default configuration."""
        with patch('modules.config_manager.ConfigManager._create_default_config') as mock_create:
            config_manager = ConfigManager()
            
            # Verify default configuration is created
            mock_create.assert_called_once()
            
            # Verify default values
            self.assertEqual(config_manager.model_config.default_epochs, 50)
            self.assertEqual(config_manager.model_config.default_batch_size, 16)
            self.assertEqual(config_manager.dataset_config.default_dataset_dir, "datasets")

    def test_init_with_custom_config_path(self):
        """Test ConfigManager initialization with custom config path."""
        # Create a temporary config file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_path)
        
        # Verify custom configuration is loaded
        self.assertEqual(config_manager.model_config.default_epochs, 50)
        self.assertEqual(config_manager.dataset_config.roboflow_project, 'test_project')

    def test_init_with_environment(self):
        """Test ConfigManager initialization with specific environment."""
        config_manager = ConfigManager(environment="production")
        
        # Verify environment is set
        self.assertEqual(config_manager.system_config.environment, Environment.PRODUCTION)

    def test_load_configuration_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        non_existent_path = "/non/existent/config.yaml"
        config_manager = ConfigManager(non_existent_path)
        
        # Should create default configuration
        self.assertIsNotNone(config_manager.model_config)
        self.assertIsNotNone(config_manager.dataset_config)

    def test_load_configuration_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        invalid_config = "invalid: yaml: content: ["
        
        with patch('builtins.open', mock_open(read_data=invalid_config)):
            with patch('modules.config_manager.ConfigManager._create_default_config') as mock_create:
                config_manager = ConfigManager(self.config_path)
                
                # Should fall back to default configuration
                mock_create.assert_called_once()

    def test_validate_configuration_valid(self):
        """Test configuration validation with valid parameters."""
        config_manager = ConfigManager()
        
        # Should not raise any exceptions
        try:
            config_manager._validate_configuration()
        except ValidationError:
            self.fail("ValidationError raised unexpectedly!")

    def test_validate_configuration_invalid_epochs(self):
        """Test configuration validation with invalid epochs."""
        config_manager = ConfigManager()
        config_manager.model_config.default_epochs = 2000  # Invalid
        
        with self.assertRaises(ValidationError):
            config_manager._validate_configuration()

    def test_validate_configuration_invalid_batch_size(self):
        """Test configuration validation with invalid batch size."""
        config_manager = ConfigManager()
        config_manager.model_config.default_batch_size = 200  # Invalid
        
        with self.assertRaises(ValidationError):
            config_manager._validate_configuration()

    def test_validate_configuration_invalid_img_size(self):
        """Test configuration validation with invalid image size."""
        config_manager = ConfigManager()
        config_manager.model_config.default_img_size = (100, 100)  # Not in supported sizes
        
        with self.assertRaises(ValidationError):
            config_manager._validate_configuration()

    def test_validate_configuration_invalid_fuzzy_thresholds(self):
        """Test configuration validation with invalid fuzzy thresholds."""
        config_manager = ConfigManager()
        config_manager.fuzzy_config.sedikit_threshold = 150  # Invalid (> 100)
        
        with self.assertRaises(ValidationError):
            config_manager._validate_configuration()

    def test_update_config_valid(self):
        """Test updating configuration with valid parameters."""
        config_manager = ConfigManager()
        
        # Update model configuration
        success = config_manager.update_config("model", "default_epochs", 100)
        self.assertTrue(success)
        self.assertEqual(config_manager.model_config.default_epochs, 100)

    def test_update_config_invalid_section(self):
        """Test updating configuration with invalid section."""
        config_manager = ConfigManager()
        
        success = config_manager.update_config("invalid_section", "key", "value")
        self.assertFalse(success)

    def test_update_config_invalid_key(self):
        """Test updating configuration with invalid key."""
        config_manager = ConfigManager()
        
        success = config_manager.update_config("model", "invalid_key", "value")
        self.assertFalse(success)

    def test_update_config_environment(self):
        """Test updating configuration environment."""
        config_manager = ConfigManager()
        
        success = config_manager.update_config("system", "environment", "production")
        self.assertTrue(success)
        self.assertEqual(config_manager.system_config.environment, Environment.PRODUCTION)

    def test_get_config_summary(self):
        """Test getting configuration summary."""
        config_manager = ConfigManager()
        summary = config_manager.get_config_summary()
        
        # Verify summary structure
        self.assertIn('environment', summary)
        self.assertIn('model', summary)
        self.assertIn('dataset', summary)
        self.assertIn('fuzzy', summary)
        self.assertIn('system', summary)

    def test_create_environment_config(self):
        """Test creating environment-specific configuration."""
        config_manager = ConfigManager()
        
        # Test production environment
        with patch('modules.config_manager.ConfigManager._save_configuration') as mock_save:
            config_manager.create_environment_config(Environment.PRODUCTION)
            
            # Verify configuration changes
            self.assertEqual(config_manager.logging_config.level, "WARNING")
            self.assertEqual(config_manager.system_config.num_workers, 8)
            self.assertEqual(config_manager.model_config.default_batch_size, 32)
            
            # Verify save was called
            mock_save.assert_called_once()

    def test_create_environment_config_development(self):
        """Test creating development environment configuration."""
        config_manager = ConfigManager()
        
        with patch('modules.config_manager.ConfigManager._save_configuration'):
            config_manager.create_environment_config(Environment.DEVELOPMENT)
            
            # Verify configuration changes
            self.assertEqual(config_manager.logging_config.level, "DEBUG")
            self.assertEqual(config_manager.system_config.num_workers, 2)
            self.assertEqual(config_manager.model_config.default_batch_size, 8)

    def test_create_environment_config_testing(self):
        """Test creating testing environment configuration."""
        config_manager = ConfigManager()
        
        with patch('modules.config_manager.ConfigManager._save_configuration'):
            config_manager.create_environment_config(Environment.TESTING)
            
            # Verify configuration changes
            self.assertEqual(config_manager.logging_config.level, "INFO")
            self.assertEqual(config_manager.system_config.num_workers, 1)
            self.assertEqual(config_manager.model_config.default_epochs, 2)
            self.assertEqual(config_manager.model_config.default_batch_size, 4)

    def test_save_configuration(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager()
        
        with patch('builtins.open', mock_open()) as mock_file:
            config_manager._save_configuration()
            
            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_apply_configuration(self):
        """Test applying configuration data."""
        config_manager = ConfigManager()
        
        # Test applying model configuration
        model_data = {'default_epochs': 75, 'default_batch_size': 32}
        config_data = {'model': model_data}
        
        config_manager._apply_configuration(config_data)
        
        # Verify changes were applied
        self.assertEqual(config_manager.model_config.default_epochs, 75)
        self.assertEqual(config_manager.model_config.default_batch_size, 32)

    def test_apply_configuration_invalid_data(self):
        """Test applying configuration with invalid data."""
        config_manager = ConfigManager()
        
        # Test with invalid data that should not crash
        invalid_data = {'model': {'invalid_key': 'value'}}
        
        try:
            config_manager._apply_configuration(invalid_data)
        except Exception as e:
            self.fail(f"Applying invalid configuration should not crash: {e}")


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig dataclass."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        self.assertEqual(config.default_epochs, 50)
        self.assertEqual(config.default_batch_size, 16)
        self.assertEqual(config.default_img_size, (640, 640))
        self.assertEqual(config.default_conf_threshold, 0.25)
        self.assertIn('v8n', config.supported_versions)
        self.assertIn('v10n', config.supported_versions)
        self.assertIn('v11n', config.supported_versions)

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            default_epochs=100,
            default_batch_size=32,
            default_img_size=(832, 832),
            default_conf_threshold=0.5
        )
        
        self.assertEqual(config.default_epochs, 100)
        self.assertEqual(config.default_batch_size, 32)
        self.assertEqual(config.default_img_size, (832, 832))
        self.assertEqual(config.default_conf_threshold, 0.5)


class TestDatasetConfig(unittest.TestCase):
    """Test cases for DatasetConfig dataclass."""

    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        config = DatasetConfig()
        
        self.assertEqual(config.roboflow_project, "abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation")
        self.assertEqual(config.roboflow_version, "1")
        self.assertEqual(config.default_dataset_dir, "datasets")
        self.assertEqual(config.default_model_dir, "runs")

    def test_dataset_config_custom_values(self):
        """Test DatasetConfig with custom values."""
        config = DatasetConfig(
            roboflow_project="custom_project",
            roboflow_version="2",
            default_dataset_dir="custom_datasets",
            default_model_dir="custom_runs"
        )
        
        self.assertEqual(config.roboflow_project, "custom_project")
        self.assertEqual(config.roboflow_version, "2")
        self.assertEqual(config.default_dataset_dir, "custom_datasets")
        self.assertEqual(config.default_model_dir, "custom_runs")


class TestFuzzyConfig(unittest.TestCase):
    """Test cases for FuzzyConfig dataclass."""

    def test_fuzzy_config_defaults(self):
        """Test FuzzyConfig default values."""
        config = FuzzyConfig()
        
        self.assertEqual(config.sedikit_threshold, 33.0)
        self.assertEqual(config.sedang_threshold, 66.0)
        self.assertIn('sedikit', config.area_percent_ranges)
        self.assertIn('sedang', config.area_percent_ranges)
        self.assertIn('banyak', config.area_percent_ranges)

    def test_fuzzy_config_custom_values(self):
        """Test FuzzyConfig with custom values."""
        custom_ranges = {
            'sedikit': [0, 0, 3],
            'sedang': [2, 8, 15],
            'banyak': [12, 100, 100]
        }
        
        config = FuzzyConfig(
            area_percent_ranges=custom_ranges,
            sedikit_threshold=25.0,
            sedang_threshold=75.0
        )
        
        self.assertEqual(config.sedikit_threshold, 25.0)
        self.assertEqual(config.sedang_threshold, 75.0)
        self.assertEqual(config.area_percent_ranges, custom_ranges)


class TestLoggingConfig(unittest.TestCase):
    """Test cases for LoggingConfig dataclass."""

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        self.assertEqual(config.level, "INFO")
        self.assertTrue(config.file_logging)
        self.assertTrue(config.console_logging)
        self.assertEqual(config.log_file, "waste_detection_system.log")

    def test_logging_config_custom_values(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            file_logging=False,
            console_logging=True,
            log_file="custom.log"
        )
        
        self.assertEqual(config.level, "DEBUG")
        self.assertFalse(config.file_logging)
        self.assertTrue(config.console_logging)
        self.assertEqual(config.log_file, "custom.log")


class TestSystemConfig(unittest.TestCase):
    """Test cases for SystemConfig dataclass."""

    def test_system_config_defaults(self):
        """Test SystemConfig default values."""
        config = SystemConfig()
        
        self.assertEqual(config.environment, Environment.COLAB)
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertTrue(config.mixed_precision)

    def test_system_config_custom_values(self):
        """Test SystemConfig with custom values."""
        config = SystemConfig(
            environment=Environment.PRODUCTION,
            num_workers=8,
            pin_memory=False,
            mixed_precision=False
        )
        
        self.assertEqual(config.environment, Environment.PRODUCTION)
        self.assertEqual(config.num_workers, 8)
        self.assertFalse(config.pin_memory)
        self.assertFalse(config.mixed_precision)


class TestEnvironment(unittest.TestCase):
    """Test cases for Environment enum."""

    def test_environment_values(self):
        """Test Environment enum values."""
        self.assertEqual(Environment.DEVELOPMENT.value, "development")
        self.assertEqual(Environment.PRODUCTION.value, "production")
        self.assertEqual(Environment.TESTING.value, "testing")
        self.assertEqual(Environment.COLAB.value, "colab")

    def test_environment_from_string(self):
        """Test creating Environment from string."""
        self.assertEqual(Environment("development"), Environment.DEVELOPMENT)
        self.assertEqual(Environment("production"), Environment.PRODUCTION)
        self.assertEqual(Environment("testing"), Environment.TESTING)
        self.assertEqual(Environment("colab"), Environment.COLAB)

    def test_environment_invalid_string(self):
        """Test creating Environment from invalid string."""
        with self.assertRaises(ValueError):
            Environment("invalid")


if __name__ == '__main__':
    unittest.main() 