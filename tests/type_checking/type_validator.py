#!/usr/bin/env python3
"""
Type Validator

This module validates data types and structures in the project.
"""

import os
import sys
from pathlib import Path
import logging
import yaml
import json
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TypeValidator:
    """
    Validator for data types and structures.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            'config_types_ok': False,
            'model_types_ok': False,
            'data_types_ok': False,
            'path_types_ok': False,
            'parameter_types_ok': False
        }
    
    def print_header(self, title):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_section(self, title):
        """Print a formatted section."""
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def validate_config_types(self):
        """Validate configuration file types."""
        print_section("Configuration Types Validation")
        
        config_file = self.project_root / "config.yaml"
        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            self.results['config_types_ok'] = False
            return
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate expected types
            type_issues = []
            
            # Check model configuration
            if 'model' in config:
                model_config = config['model']
                if not isinstance(model_config, dict):
                    type_issues.append("model config should be a dictionary")
                else:
                    if 'default_img_size' in model_config and not isinstance(model_config['default_img_size'], int):
                        type_issues.append("default_img_size should be an integer")
                    if 'batch_size' in model_config and not isinstance(model_config['batch_size'], int):
                        type_issues.append("batch_size should be an integer")
                    if 'epochs' in model_config and not isinstance(model_config['epochs'], int):
                        type_issues.append("epochs should be an integer")
            
            # Check dataset configuration
            if 'dataset' in config:
                dataset_config = config['dataset']
                if not isinstance(dataset_config, dict):
                    type_issues.append("dataset config should be a dictionary")
                else:
                    if 'path' in dataset_config and not isinstance(dataset_config['path'], str):
                        type_issues.append("dataset path should be a string")
                    if 'classes' in dataset_config and not isinstance(dataset_config['classes'], list):
                        type_issues.append("dataset classes should be a list")
            
            # Check training configuration
            if 'training' in config:
                training_config = config['training']
                if not isinstance(training_config, dict):
                    type_issues.append("training config should be a dictionary")
                else:
                    if 'device' in training_config and not isinstance(training_config['device'], str):
                        type_issues.append("training device should be a string")
                    if 'workers' in training_config and not isinstance(training_config['workers'], int):
                        type_issues.append("training workers should be an integer")
            
            if type_issues:
                print("⚠️ Configuration type issues found:")
                for issue in type_issues:
                    print(f"   - {issue}")
                self.results['config_types_ok'] = False
            else:
                print("✅ All configuration types are valid")
                self.results['config_types_ok'] = True
                
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            self.results['config_types_ok'] = False
    
    def validate_model_types(self):
        """Validate model-related data types."""
        print_section("Model Types Validation")
        
        # Check model processor types
        try:
            sys.path.append(str(self.project_root))
            from modules.model_processor import ModelProcessor
            
            # Validate ModelProcessor attributes
            processor = ModelProcessor()
            type_issues = []
            
            # Check path attributes
            if not isinstance(processor.MODEL_DIR, str):
                type_issues.append("MODEL_DIR should be a string")
            if not isinstance(processor.ONNX_MODEL_DIR, str):
                type_issues.append("ONNX_MODEL_DIR should be a string")
            if not isinstance(processor.RKNN_MODEL_DIR, str):
                type_issues.append("RKNN_MODEL_DIR should be a string")
            
            # Check configuration attributes
            if not isinstance(processor.img_size, int):
                type_issues.append("img_size should be an integer")
            if not isinstance(processor.batch_size, int):
                type_issues.append("batch_size should be an integer")
            if not isinstance(processor.epochs, int):
                type_issues.append("epochs should be an integer")
            
            if type_issues:
                print("⚠️ Model type issues found:")
                for issue in type_issues:
                    print(f"   - {issue}")
                self.results['model_types_ok'] = False
            else:
                print("✅ All model types are valid")
                self.results['model_types_ok'] = True
                
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            self.results['model_types_ok'] = False
    
    def validate_data_types(self):
        """Validate dataset and data types."""
        print_section("Data Types Validation")
        
        # Check dataset structure
        dataset_dir = self.project_root / "datasets"
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            self.results['data_types_ok'] = False
            return
        
        type_issues = []
        
        # Check for required dataset files
        required_files = ['train', 'val', 'test']
        for split in required_files:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                type_issues.append(f"Dataset split '{split}' directory not found")
            else:
                # Check for images and labels
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                
                if not images_dir.exists():
                    type_issues.append(f"Images directory not found for {split} split")
                if not labels_dir.exists():
                    type_issues.append(f"Labels directory not found for {split} split")
        
        # Check data.yaml file
        data_yaml = dataset_dir / "data.yaml"
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                
                # Validate data.yaml structure
                if not isinstance(data_config, dict):
                    type_issues.append("data.yaml should contain a dictionary")
                else:
                    if 'train' in data_config and not isinstance(data_config['train'], str):
                        type_issues.append("train path in data.yaml should be a string")
                    if 'val' in data_config and not isinstance(data_config['val'], str):
                        type_issues.append("val path in data.yaml should be a string")
                    if 'nc' in data_config and not isinstance(data_config['nc'], int):
                        type_issues.append("nc (number of classes) in data.yaml should be an integer")
                    if 'names' in data_config and not isinstance(data_config['names'], list):
                        type_issues.append("names in data.yaml should be a list")
                        
            except Exception as e:
                type_issues.append(f"data.yaml validation failed: {e}")
        else:
            type_issues.append("data.yaml file not found")
        
        if type_issues:
            print("⚠️ Data type issues found:")
            for issue in type_issues:
                print(f"   - {issue}")
            self.results['data_types_ok'] = False
        else:
            print("✅ All data types are valid")
            self.results['data_types_ok'] = True
    
    def validate_path_types(self):
        """Validate path and directory types."""
        print_section("Path Types Validation")
        
        type_issues = []
        
        # Check required directories
        required_dirs = [
            "results",
            "results/runs",
            "results/onnx_models",
            "results/rknn_models",
            "datasets",
            "modules",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                type_issues.append(f"Required directory not found: {dir_path}")
            elif not full_path.is_dir():
                type_issues.append(f"Path exists but is not a directory: {dir_path}")
        
        # Check file permissions
        try:
            # Test write permissions for results directory
            results_dir = self.project_root / "results"
            if results_dir.exists():
                test_file = results_dir / ".test_write"
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception:
                    type_issues.append("Results directory is not writable")
        except Exception as e:
            type_issues.append(f"Path permission check failed: {e}")
        
        if type_issues:
            print("⚠️ Path type issues found:")
            for issue in type_issues:
                print(f"   - {issue}")
            self.results['path_types_ok'] = False
        else:
            print("✅ All path types are valid")
            self.results['path_types_ok'] = True
    
    def validate_parameter_types(self):
        """Validate function parameter types."""
        print_section("Parameter Types Validation")
        
        type_issues = []
        
        try:
            sys.path.append(str(self.project_root))
            
            # Check main_colab.py parameter types
            from main_colab import WasteDetectionSystemColab
            
            # Validate class attributes
            system = WasteDetectionSystemColab()
            
            # Check that required attributes exist and have correct types
            if not hasattr(system, 'model_processor'):
                type_issues.append("WasteDetectionSystemColab should have model_processor attribute")
            elif not hasattr(system.model_processor, 'train_yolo_model'):
                type_issues.append("ModelProcessor should have train_yolo_model method")
            
            if not hasattr(system, 'analyze_training_run'):
                type_issues.append("WasteDetectionSystemColab should have analyze_training_run method")
            
            if not hasattr(system, 'run_inference_and_visualization'):
                type_issues.append("WasteDetectionSystemColab should have run_inference_and_visualization method")
            
            if not hasattr(system, 'convert_and_zip_rknn_models'):
                type_issues.append("WasteDetectionSystemColab should have convert_and_zip_rknn_models method")
            
            if type_issues:
                print("⚠️ Parameter type issues found:")
                for issue in type_issues:
                    print(f"   - {issue}")
                self.results['parameter_types_ok'] = False
            else:
                print("✅ All parameter types are valid")
                self.results['parameter_types_ok'] = True
                
        except Exception as e:
            print(f"❌ Parameter validation failed: {e}")
            self.results['parameter_types_ok'] = False
    
    def run_all_validations(self):
        """Run all type validations."""
        self.print_header("Type Validator")
        
        self.validate_config_types()
        self.validate_model_types()
        self.validate_data_types()
        self.validate_path_types()
        self.validate_parameter_types()
        
        self.print_header("Validation Summary")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all validations."""
        print("📊 Type Validation Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_checks}/{total_checks} validations passed")
        
        if passed_checks == total_checks:
            print("🎉 All type validations passed! Data types are consistent.")
        else:
            print("⚠️ Some validations failed. Please review the issues above.")

def main():
    """Run type validator."""
    validator = TypeValidator()
    results = validator.run_all_validations()
    
    print("\n💡 For type fixing, use the TypeFixer module")

if __name__ == "__main__":
    main() 