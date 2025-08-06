# Test Structure

This directory contains all tests for the Waste Detection System, organized into logical categories for better maintainability and clarity.

## 📁 Directory Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_config_manager.py
│   ├── test_exceptions.py
│   ├── test_fuzzy_area_classifier.py
│   ├── test_secrets_validation.py
│   ├── test_dynamic_directory_fix.py
│   └── test_dynamic_directory_logic.py
├── integration/             # Integration tests for system-wide functionality
│   ├── test_main_colab.py
│   ├── test_dataset_fix.py
│   ├── test_dataset_fix_verification.py
│   ├── test_path_fix.py
│   └── test_coco_structure_verification.py

├── diagnostic/              # Diagnostic scripts for troubleshooting
│   ├── check_categories.py
│   ├── check_dataset_annotations.py
│   ├── check_polygon_segmentation.py
│   ├── check_cuda_versions.py
│   └── check_yolo_label_matching.py
├── fixes/                   # Fix verification scripts
│   ├── test_config_fix.py
│   ├── test_onnx_export_fix.py
│   ├── test_sampah_normalization.py
│   └── test_sampah_only.py
├── dataset_tools/           # Dataset diagnosis and fix tools
│   ├── diagnose_dataset.py
│   ├── fix_dataset_classes.py
│   ├── fix_dataset_ultralytics.py
│   ├── dataset_validator.py
│   ├── extract_and_check_dataset.py
│   ├── final_verification.py
│   ├── fix_yolo_coordinates.py
│   ├── test_dataset_fix_integration.py
│   ├── run_dataset_fix_test.py
│   └── README.md
└── utils/                   # Utility test scripts
    ├── test_cuda.py
    └── verify_python311_compatibility.py
└── training/                # Training system tests
    ├── enhanced_training_system.py
    ├── simple_enhanced_training.py
    └── README.md
└── existing_results/        # Existing results usage tests
    ├── use_existing_training.py
    ├── use_existing_results.py
    ├── use_existing_results_safe.py
    └── README.md
```

## 🧪 Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual modules and components in isolation
- **Scope**: Single functions, classes, or small groups of related functionality
- **Examples**: Configuration management, exception handling, fuzzy logic, dynamic directory naming

### Integration Tests (`integration/`)
- **Purpose**: Test how different modules work together
- **Scope**: End-to-end workflows and system-wide functionality
- **Examples**: Main system initialization, dataset processing workflows, dataset fix verification, path fixing



### Diagnostic Tests (`diagnostic/`)
- **Purpose**: Troubleshoot specific issues and verify system state
- **Scope**: Problem identification and debugging
- **Examples**: Dataset validation, CUDA version checking, category analysis, YOLO label matching

### Fix Verification Tests (`fixes/`)
- **Purpose**: Verify that specific fixes work correctly
- **Scope**: Targeted testing of bug fixes and improvements
- **Examples**: ONNX export fixes, configuration fixes, normalization fixes

### Dataset Tools (`dataset_tools/`)
- **Purpose**: Diagnose and fix dataset issues
- **Scope**: COCO to YOLO conversion, class configuration, validation, integration testing, coordinate fixing, dataset extraction and verification
- **Examples**: Dataset diagnosis, class normalization, format conversion, coordinate normalization, dataset extraction, final verification, dataset fix integration tests

### Utility Tests (`utils/`)
- **Purpose**: Environment and compatibility verification
- **Scope**: System requirements and dependencies
- **Examples**: CUDA setup verification, Python version compatibility

### Training Tests (`training/`)
- **Purpose**: Test enhanced training systems with checkpoint functionality
- **Scope**: Training workflows, checkpoint management, local and Colab execution
- **Examples**: Enhanced training system, simple training simulation, checkpoint operations

### Existing Results Tests (`existing_results/`)
- **Purpose**: Test using existing training results without retraining
- **Scope**: Finding existing runs, analysis, inference, and RKNN conversion
- **Examples**: Safe analysis, inference visualization, model conversion

## 🧪 Dynamic Directory Fix Tests

### Overview
The dynamic directory fix tests verify that the system can correctly handle Ultralytics' automatic directory naming with suffixes (e.g., `segment_train_v8n2` instead of `segment_train_v8n`).

### Test Files
- **`test_dynamic_directory_fix.py`**: Comprehensive test suite for the ModelProcessor's dynamic directory finding functionality
- **`test_dynamic_directory_logic.py`**: Focused test for directory finding logic without ultralytics dependencies

### Key Test Cases
1. **No suffix**: `segment_train_v8n` - Found correctly
2. **With suffix**: `segment_train_v8n2` - Found correctly  
3. **Large suffix**: `segment_train_v8n15` - Found correctly
4. **Multiple versions**: `v8n`, `v10n`, `v11n` - All found correctly
5. **No directory**: Returns `None` when no directory exists
6. **Pattern matching**: Finds directories with correct prefixes
7. **Multiple directories**: Handles multiple directories with same prefix
8. **Case sensitivity**: Correctly handles case differences
9. **Partial matches**: Finds directories that start with the pattern

### Running Dynamic Directory Tests
```bash
# Run both dynamic directory tests
python tests/unit/test_dynamic_directory_logic.py
python tests/unit/test_dynamic_directory_fix.py

# Run with unittest
python -m unittest tests.unit.test_dynamic_directory_logic
python -m unittest tests.unit.test_dynamic_directory_fix
```

### Problem Solved
- **Original Issue**: Hardcoded paths expected `segment_train_v8n` but Ultralytics created `segment_train_v8n2`
- **Solution**: Dynamic directory finding that searches for directories starting with `segment_train_{model_version}`
- **Benefits**: Compatible with any suffix Ultralytics adds, future-proof, robust error handling

## 🚀 Running Tests
```bash
python run_tests.py
```

### Run Specific Categories
```bash
# Unit tests only
python run_tests.py --category unit

# Integration tests only
python run_tests.py --category integration

# Diagnostic tests only
python run_tests.py --category diagnostic

# Fix verification tests only
python run_tests.py --category fixes

# Dataset tools only
python run_tests.py --category dataset_tools

# Utility tests only
python run_tests.py --category utils

# Training tests only
python run_tests.py --category training

# Existing results tests only
python run_tests.py --category existing_results
```

### Run Specific Tests
```bash
# Run a specific test file
python run_tests.py --test config_manager

# Run dataset fix integration test
python tests/dataset_tools/test_dataset_fix_integration.py

# Run with verbose output
python run_tests.py --category unit --verbose

# Run with quiet output
python run_tests.py --category integration --quiet
```

### Legacy Categories (Still Supported)
```bash
# Security tests
python run_tests.py --category security

# Configuration tests
python run_tests.py --category config

# Fuzzy logic tests
python run_tests.py --category fuzzy

# Exception tests
python run_tests.py --category exceptions
```

## 📋 Test Guidelines

### Writing New Tests

1. **Choose the right category**:
   - `unit/` for individual module tests
   - `integration/` for system-wide tests
   - `diagnostic/` for troubleshooting scripts
   - `fixes/` for bug fix verification
   - `dataset_tools/` for dataset diagnosis and fixes
   - `utils/` for environment checks

2. **Follow naming conventions**:
   - Unit/Integration tests: `test_*.py`
   - Diagnostic scripts: `check_*.py`
   - Fix verification: `test_*_fix.py`
   - Dataset tools: `diagnose_*.py`, `fix_*.py`, `*_validator.py`
   - Utility scripts: `test_*.py` or `verify_*.py`

3. **Include proper documentation**:
   - Clear docstrings explaining test purpose
   - Comments for complex test logic
   - Usage examples in docstrings

### Test File Structure
```python
#!/usr/bin/env python3
"""
Brief description of what this test does.
"""

import unittest
import logging
import tempfile
import shutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestClassName(unittest.TestCase):
    """Test class description."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Set up test environment
    
    def test_specific_functionality(self):
        """Test specific functionality."""
        # Test implementation
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
```

## 🔧 Maintenance

### Adding New Tests
1. Create the test file in the appropriate category directory
2. Add `__init__.py` if creating a new category
3. Update this README if adding new categories
4. Test the new test file individually before committing

### Updating Test Runner
- Modify `run_tests.py` to support new categories
- Update argument parser choices
- Add category-specific test discovery logic

### Cleaning Up
- Remove obsolete test files
- Update imports when moving files
- Keep diagnostic scripts focused and up-to-date

## 📊 Test Results

Tests provide detailed output including:
- ✅ Passed tests
- ❌ Failed tests with error details
- ⏱️ Execution time
- 📈 Success rates
- 🔍 Detailed error traces

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Ensure project root is in Python path
2. **Missing dependencies**: Install required packages
3. **Path issues**: Check file paths are correct for your OS
4. **Permission errors**: Ensure write access to test directories

### Getting Help
- Check test output for specific error messages
- Run individual test files for detailed debugging
- Use `--verbose` flag for more detailed output
- Review test logs for additional context 