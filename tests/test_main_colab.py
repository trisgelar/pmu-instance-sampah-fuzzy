#!/usr/bin/env python3
"""
Unit tests for the main orchestrator class.
Tests system initialization, configuration management, and error handling.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_colab import WasteDetectionSystemColab
from modules.exceptions import (
    ConfigurationError, ValidationError, DatasetError, ModelError
)


class TestWasteDetectionSystemColab(unittest.TestCase):
    """Test cases for WasteDetectionSystemColab class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock component initialization
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Verify ConfigManager was initialized
                mock_config_manager.assert_called_once()
                
                # Verify configuration sections were retrieved
                mock_instance.get_model_config.assert_called_once()
                mock_instance.get_dataset_config.assert_called_once()
                mock_instance.get_fuzzy_config.assert_called_once()
                mock_instance.get_logging_config.assert_called_once()
                mock_instance.get_system_config.assert_called_once()

    def test_init_with_custom_config_path(self):
        """Test initialization with custom configuration path."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock component initialization
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab(config_path=self.config_path)
                
                # Verify ConfigManager was initialized with custom path
                mock_config_manager.assert_called_once_with(self.config_path, None)

    def test_init_with_environment(self):
        """Test initialization with specific environment."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock component initialization
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab(environment="production")
                
                # Verify ConfigManager was initialized with environment
                mock_config_manager.assert_called_once_with(None, "production")

    def test_init_configuration_error(self):
        """Test initialization with configuration error."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_config_manager.side_effect = ConfigurationError("Config error")
            
            with self.assertRaises(ConfigurationError):
                WasteDetectionSystemColab()

    def test_init_component_initialization_error(self):
        """Test initialization with component initialization error."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock component initialization to fail
            with patch('main_colab.DatasetManager', side_effect=Exception("Component error")):
                with self.assertRaises(ConfigurationError):
                    WasteDetectionSystemColab()

    def test_train_and_export_model_valid(self):
        """Test training and export with valid parameters."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_model_config = MagicMock()
            mock_model_config.default_epochs = 50
            mock_model_config.default_batch_size = 16
            mock_model_config.supported_versions = ["v8n", "v10n", "v11n"]
            mock_model_config.min_epochs = 1
            mock_model_config.max_epochs = 1000
            mock_model_config.min_batch_size = 1
            mock_model_config.max_batch_size = 128
            
            mock_instance.get_model_config.return_value = mock_model_config
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Mock training process
                with patch.object(system, 'model_processor') as mock_model_processor:
                    mock_model_processor.train_model.return_value = "runs/train/exp1"
                    
                    result = system.train_and_export_model("v8n", epochs=10, batch_size=8)
                    
                    self.assertEqual(result, "runs/train/exp1")
                    mock_model_processor.train_model.assert_called_once()

    def test_train_and_export_model_invalid_version(self):
        """Test training with invalid model version."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_model_config = MagicMock()
            mock_model_config.supported_versions = ["v8n", "v10n", "v11n"]
            
            mock_instance.get_model_config.return_value = mock_model_config
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                with self.assertRaises(ValidationError):
                    system.train_and_export_model("invalid_version")

    def test_train_and_export_model_invalid_epochs(self):
        """Test training with invalid epochs."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_model_config = MagicMock()
            mock_model_config.default_epochs = 50
            mock_model_config.supported_versions = ["v8n", "v10n", "v11n"]
            mock_model_config.min_epochs = 1
            mock_model_config.max_epochs = 1000
            
            mock_instance.get_model_config.return_value = mock_model_config
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                with self.assertRaises(ValidationError):
                    system.train_and_export_model("v8n", epochs=2000)

    def test_train_and_export_model_invalid_batch_size(self):
        """Test training with invalid batch size."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_model_config = MagicMock()
            mock_model_config.default_batch_size = 16
            mock_model_config.supported_versions = ["v8n", "v10n", "v11n"]
            mock_model_config.min_batch_size = 1
            mock_model_config.max_batch_size = 128
            
            mock_instance.get_model_config.return_value = mock_model_config
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                with self.assertRaises(ValidationError):
                    system.train_and_export_model("v8n", batch_size=200)

    def test_get_system_status(self):
        """Test getting system status."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            mock_instance.get_config_summary.return_value = {"test": "config"}
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Mock file existence checks
                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    status = system.get_system_status()
                    
                    # Verify status structure
                    self.assertIn('configuration', status)
                    self.assertIn('dataset_ready', status)
                    self.assertIn('model_dir_exists', status)
                    self.assertIn('components_initialized', status)
                    self.assertEqual(status['configuration'], {"test": "config"})

    def test_update_configuration_valid(self):
        """Test updating configuration with valid parameters."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            mock_instance.update_config.return_value = True
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Mock component re-initialization
                with patch.object(system, '_initialize_components'):
                    success = system.update_configuration("model", "default_epochs", 100)
                    
                    self.assertTrue(success)
                    mock_instance.update_config.assert_called_once_with("model", "default_epochs", 100)

    def test_update_configuration_invalid(self):
        """Test updating configuration with invalid parameters."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            mock_instance.update_config.return_value = False
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                success = system.update_configuration("invalid", "key", "value")
                
                self.assertFalse(success)

    def test_get_configuration_summary(self):
        """Test getting configuration summary."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            mock_instance.get_config_summary.return_value = {"test": "summary"}
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                summary = system.get_configuration_summary()
                
                self.assertEqual(summary, {"test": "summary"})

    def test_run_inference_and_visualization(self):
        """Test running inference and visualization."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_model_config = MagicMock()
            mock_model_config.default_conf_threshold = 0.25
            
            mock_instance.get_model_config.return_value = mock_model_config
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Mock inference visualizer
                with patch.object(system, 'inference_visualizer') as mock_visualizer:
                    mock_visualizer.run_inference.return_value = {"results": "test"}
                    
                    result = system.run_inference_and_visualization("runs/train/exp1", "v8n")
                    
                    self.assertEqual(result, {"results": "test"})
                    mock_visualizer.run_inference.assert_called_once()

    def test_analyze_training_run(self):
        """Test analyzing training run."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                system = WasteDetectionSystemColab()
                
                # Mock metrics analyzer
                with patch.object(system, 'metrics_analyzer') as mock_analyzer:
                    mock_analyzer.analyze_training_metrics.return_value = {"metrics": "test"}
                    
                    result = system.analyze_training_run("runs/train/exp1", "v8n")
                    
                    self.assertEqual(result, {"metrics": "test"})
                    mock_analyzer.analyze_training_metrics.assert_called_once()


class TestWasteDetectionSystemColabIntegration(unittest.TestCase):
    """Integration tests for WasteDetectionSystemColab."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_full_system_workflow(self):
        """Test the complete system workflow."""
        # Create test configuration
        test_config = {
            'model': {
                'supported_versions': ['v8n'],
                'default_epochs': 2,
                'default_batch_size': 4
            },
            'dataset': {
                'roboflow_project': 'test_project',
                'roboflow_version': '1'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Mock all external dependencies
        with patch('main_colab.ConfigManager') as mock_config_manager, \
             patch('main_colab.DatasetManager') as mock_dataset_manager, \
             patch('main_colab.ModelProcessor') as mock_model_processor, \
             patch('main_colab.RknnConverter') as mock_rknn_converter, \
             patch('main_colab.TrainingMetricsAnalyzer') as mock_metrics_analyzer, \
             patch('main_colab.InferenceVisualizer') as mock_inference_visualizer:
            
            # Mock configuration manager
            mock_config_instance = MagicMock()
            mock_config_manager.return_value = mock_config_instance
            
            # Mock configuration sections
            mock_config_instance.get_model_config.return_value = MagicMock()
            mock_config_instance.get_dataset_config.return_value = MagicMock()
            mock_config_instance.get_fuzzy_config.return_value = MagicMock()
            mock_config_instance.get_logging_config.return_value = MagicMock()
            mock_config_instance.get_system_config.return_value = MagicMock()
            mock_config_instance.get_config_summary.return_value = {"test": "config"}
            mock_config_instance.update_config.return_value = True
            
            # Mock components
            mock_dataset_instance = MagicMock()
            mock_dataset_manager.return_value = mock_dataset_instance
            
            mock_model_instance = MagicMock()
            mock_model_processor.return_value = mock_model_instance
            mock_model_instance.train_model.return_value = "runs/train/exp1"
            
            mock_metrics_instance = MagicMock()
            mock_metrics_analyzer.return_value = mock_metrics_instance
            mock_metrics_instance.analyze_training_metrics.return_value = {"metrics": "test"}
            
            mock_inference_instance = MagicMock()
            mock_inference_visualizer.return_value = mock_inference_instance
            mock_inference_instance.run_inference.return_value = {"inference": "test"}
            
            # Initialize system
            system = WasteDetectionSystemColab(self.config_path)
            
            # Test configuration update
            success = system.update_configuration("model", "default_epochs", 5)
            self.assertTrue(success)
            
            # Test training
            result = system.train_and_export_model("v8n", epochs=2, batch_size=4)
            self.assertEqual(result, "runs/train/exp1")
            
            # Test analysis
            analysis = system.analyze_training_run("runs/train/exp1", "v8n")
            self.assertEqual(analysis, {"metrics": "test"})
            
            # Test inference
            inference = system.run_inference_and_visualization("runs/train/exp1", "v8n")
            self.assertEqual(inference, {"inference": "test"})
            
            # Test status
            status = system.get_system_status()
            self.assertIn('configuration', status)
            self.assertIn('components_initialized', status)

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock component initialization to fail
            with patch('main_colab.DatasetManager', side_effect=DatasetError("Dataset error")):
                with self.assertRaises(ConfigurationError):
                    WasteDetectionSystemColab()

    def test_performance_under_load(self):
        """Test system performance under load."""
        import time
        
        with patch('main_colab.ConfigManager') as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            
            # Mock configuration sections
            mock_instance.get_model_config.return_value = MagicMock()
            mock_instance.get_dataset_config.return_value = MagicMock()
            mock_instance.get_fuzzy_config.return_value = MagicMock()
            mock_instance.get_logging_config.return_value = MagicMock()
            mock_instance.get_system_config.return_value = MagicMock()
            
            # Mock components
            with patch('main_colab.DatasetManager'), \
                 patch('main_colab.ModelProcessor'), \
                 patch('main_colab.RknnConverter'), \
                 patch('main_colab.TrainingMetricsAnalyzer'), \
                 patch('main_colab.InferenceVisualizer'):
                
                # Test initialization performance
                start_time = time.time()
                
                for _ in range(100):
                    system = WasteDetectionSystemColab()
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Should complete within reasonable time
                self.assertLess(duration, 10.0, f"Initialization took too long: {duration:.2f} seconds")


if __name__ == '__main__':
    unittest.main() 