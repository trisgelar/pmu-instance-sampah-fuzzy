#!/usr/bin/env python3
"""
Test for dynamic directory naming fix in ModelProcessor.

This test verifies that the ModelProcessor can correctly find training directories
that Ultralytics creates with dynamic suffixes (e.g., segment_train_v8n2).
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.model_processor import ModelProcessor
from modules.exceptions import ModelError


class TestDynamicDirectoryFix(unittest.TestCase):
    """Test cases for dynamic directory naming fix."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "results", "runs")
        self.onnx_dir = os.path.join(self.temp_dir, "results", "onnx")
        self.img_size = (640, 640)
        
        # Create directory structure
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)
        
        # Initialize ModelProcessor
        self.model_processor = ModelProcessor(
            model_dir=self.model_dir,
            onnx_model_dir=self.onnx_dir,
            img_size=self.img_size
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_training_directory(self, model_version: str, suffix: str = ""):
        """Create a mock training directory structure."""
        base_name = f"segment_train_{model_version}"
        dir_name = f"{base_name}{suffix}"
        
        # Create the training directory structure
        train_dir = os.path.join(self.model_dir, "segment", dir_name)
        weights_dir = os.path.join(train_dir, "weights")
        
        os.makedirs(weights_dir, exist_ok=True)
        
        # Create a mock best.pt file
        best_pt_path = os.path.join(weights_dir, "best.pt")
        with open(best_pt_path, 'w') as f:
            f.write("mock model content")
        
        return train_dir, weights_dir, best_pt_path
    
    def test_find_training_directory_no_suffix(self):
        """Test finding training directory without suffix."""
        # Create directory: segment_train_v8n
        train_dir, weights_dir, best_pt_path = self.create_mock_training_directory("v8n")
        
        # Test get_model_paths
        paths = self.model_processor.get_model_paths("v8n")
        
        self.assertNotIn("error", paths)
        self.assertEqual(paths["pytorch_model"], best_pt_path)
        self.assertEqual(paths["weights_dir"], weights_dir)
        self.assertEqual(paths["run_dir"], train_dir)
    
    def test_find_training_directory_with_suffix(self):
        """Test finding training directory with suffix (e.g., v8n2)."""
        # Create directory: segment_train_v8n2
        train_dir, weights_dir, best_pt_path = self.create_mock_training_directory("v8n", "2")
        
        # Test get_model_paths
        paths = self.model_processor.get_model_paths("v8n")
        
        self.assertNotIn("error", paths)
        self.assertEqual(paths["pytorch_model"], best_pt_path)
        self.assertEqual(paths["weights_dir"], weights_dir)
        self.assertEqual(paths["run_dir"], train_dir)
    
    def test_find_training_directory_with_large_suffix(self):
        """Test finding training directory with large suffix (e.g., v8n15)."""
        # Create directory: segment_train_v8n15
        train_dir, weights_dir, best_pt_path = self.create_mock_training_directory("v8n", "15")
        
        # Test get_model_paths
        paths = self.model_processor.get_model_paths("v8n")
        
        self.assertNotIn("error", paths)
        self.assertEqual(paths["pytorch_model"], best_pt_path)
        self.assertEqual(paths["weights_dir"], weights_dir)
        self.assertEqual(paths["run_dir"], train_dir)
    
    def test_find_training_directory_multiple_versions(self):
        """Test finding training directories for multiple model versions."""
        # Create directories for different versions
        v8n_dir, v8n_weights, v8n_best = self.create_mock_training_directory("v8n", "2")
        v10n_dir, v10n_weights, v10n_best = self.create_mock_training_directory("v10n", "3")
        v11n_dir, v11n_weights, v11n_best = self.create_mock_training_directory("v11n", "1")
        
        # Test all versions
        for version in ["v8n", "v10n", "v11n"]:
            paths = self.model_processor.get_model_paths(version)
            self.assertNotIn("error", paths)
            self.assertTrue(os.path.exists(paths["pytorch_model"]))
    
    def test_no_training_directory_found(self):
        """Test behavior when no training directory is found."""
        # Don't create any training directories
        
        # Test get_model_paths
        paths = self.model_processor.get_model_paths("v8n")
        
        self.assertIn("error", paths)
        self.assertIn("Training directory not found", paths["error"])
    
    def test_zip_weights_folder_with_suffix(self):
        """Test zip_weights_folder with dynamic directory naming."""
        # Create directory: segment_train_v8n2
        train_dir, weights_dir, best_pt_path = self.create_mock_training_directory("v8n", "2")
        
        # Test zip_weights_folder
        result = self.model_processor.zip_weights_folder("v8n")
        
        self.assertTrue(result)
        
        # Check if zip file was created
        expected_zip = f"segment_v8n_weights.zip"
        self.assertTrue(os.path.exists(expected_zip))
        
        # Clean up zip file
        if os.path.exists(expected_zip):
            os.remove(expected_zip)
    
    def test_zip_weights_folder_no_directory(self):
        """Test zip_weights_folder when no directory exists."""
        # Don't create any training directories
        
        # Test zip_weights_folder
        result = self.model_processor.zip_weights_folder("v8n")
        
        self.assertFalse(result)
    
    @patch('modules.model_processor.YOLO')
    def test_train_yolo_model_dynamic_directory(self, mock_yolo):
        """Test that train_yolo_model uses dynamic directory finding."""
        # Mock YOLO model and training
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Mock training results
        mock_results = MagicMock()
        mock_model.train.return_value = mock_results
        
        # Mock model export
        mock_model.export.return_value = None
        
        # Create a mock training directory that would be created by Ultralytics
        train_dir, weights_dir, best_pt_path = self.create_mock_training_directory("v8n", "2")
        
        # Test train_yolo_model
        results, run_dir = self.model_processor.train_yolo_model(
            model_version="v8n",
            data_yaml_path="mock_data.yaml",
            epochs=1,
            batch_size=1
        )
        
        # Verify that the correct directory was found
        self.assertEqual(run_dir, train_dir)
        self.assertTrue(os.path.exists(best_pt_path))
    
    def test_directory_pattern_matching(self):
        """Test that directory pattern matching works correctly."""
        # Create multiple directories with different patterns
        self.create_mock_training_directory("v8n", "")  # segment_train_v8n
        self.create_mock_training_directory("v8n", "2")  # segment_train_v8n2
        self.create_mock_training_directory("v10n", "1")  # segment_train_v10n1
        
        # Test that each version finds the correct directory
        v8n_paths = self.model_processor.get_model_paths("v8n")
        v10n_paths = self.model_processor.get_model_paths("v10n")
        
        self.assertNotIn("error", v8n_paths)
        self.assertNotIn("error", v10n_paths)
        
        # Should find the first matching directory (alphabetical order)
        self.assertIn("segment_train_v8n", v8n_paths["run_dir"])
        self.assertIn("segment_train_v10n1", v10n_paths["run_dir"])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 