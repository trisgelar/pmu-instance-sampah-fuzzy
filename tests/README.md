# Comprehensive Testing System

This directory contains a comprehensive test suite for the Waste Detection System, organized into logical categories for better maintainability, clarity, and systematic validation.

## 🎯 **Testing Philosophy**

Our testing system follows these principles:
- **Comprehensive Coverage**: Every component is tested
- **Systematic Organization**: Logical categorization for easy navigation
- **Graceful Error Handling**: Robust import and error management
- **Extensible Design**: Easy to add new test categories
- **Documentation**: Clear documentation for all test components

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
├── dataset_validation/      # Dataset validation tools
│   ├── dataset_validator.py
│   └── final_verification.py
├── diagnostic/              # Diagnostic checks
│   ├── check_basic_structure.py
│   ├── check_categories.py
│   ├── check_cuda_versions.py
│   ├── check_dataset_annotations.py
│   ├── check_polygon_segmentation.py
│   ├── check_yolo_label_matching.py
│   ├── diagnose_dataset.py
│   ├── check_segmentation_format.py
│   └── extract_and_check_dataset.py
├── bug_fixing/              # Bug fixing tools
│   ├── bug_detector.py
│   ├── bug_fixer.py
│   ├── bug_validator.py
│   ├── fix_dataset_classes.py
│   ├── fix_dataset_ultralytics.py
│   ├── fix_segmentation_labels.py
│   └── fix_yolo_coordinates.py
├── integration/             # Integration tests
│   ├── test_dataset_fix_integration.py
│   ├── test_segmentation_integration.py
│   ├── test_config_fix.py
│   ├── test_onnx_export_fix.py
│   ├── test_sampah_normalization.py
│   ├── test_sampah_only.py
│   └── run_dataset_fix_test.py
├── training_tools/          # Training tools and tests
│   ├── enhanced_training_system.py
│   ├── simple_enhanced_training.py
│   └── test_training_imports.py
├── utils/                   # Utility tools
│   ├── example_usage.py
│   ├── test_cuda.py
│   └── verify_python311_compatibility.py
└── existing_results/        # Existing results usage tests
    ├── use_existing_training.py
    ├── use_existing_results.py
    ├── use_existing_results_safe.py
    └── README.md
└── onnx_testing/           # Structured ONNX testing
    ├── check_onnx_environment.py
    ├── check_onnx_models.py
    ├── check_onnx_conversion.py
    └── check_onnx_inference.py
└── type_checking/          # Type validation and checking
    ├── type_validator.py
    ├── type_fixer.py
    └── type_checker.py
└── bug_fixing/             # Bug detection and fixing
    ├── bug_detector.py
    ├── bug_fixer.py
    └── bug_validator.py
└── validation/             # General validation tools
    ├── model_validator.py
    ├── data_validator.py
    └── config_validator.py
```

## 🧪 Test Categories

### **🔧 ONNX Testing (`onnx_testing/`)**
- **Purpose**: Structured ONNX testing and validation
- **Scope**: Environment setup, model validation, conversion testing, inference testing
- **Key Files**:
  - `check_onnx_environment.py` - Environment setup validation (check0)
  - `check_onnx_models.py` - Model file validation (check1)
  - `check_onnx_conversion.py` - Conversion process testing (check2)
  - `check_onnx_inference.py` - Inference testing (check3)
  - `check_onnx_rknn_environment.py` - ONNX/RKNN environment checker
  - `check0_base_optimize.onnx` - Base optimized ONNX model
  - `check1_fold_constant.onnx` - Fold constant optimized ONNX model
  - `check2_correct_ops.onnx` - Correct ops optimized ONNX model
  - `check3_fuse_ops.onnx` - Fuse ops optimized ONNX model
- **Usage**: `python run_tests.py --category onnx_testing`

### **🔍 Type Checking (`type_checking/`)**
- **Purpose**: Type validation, checking, and fixing
- **Scope**: Data type validation, type compatibility checks, type fixing utilities
- **Key Files**:
  - `type_validator.py` - Validates data types and structures
  - `type_fixer.py` - Fixes type-related issues
  - `type_checker.py` - Checks for type compatibility
- **Usage**: `python run_tests.py --category type_checking`

### **🐛 Bug Fixing (`bug_fixing/`)**
- **Purpose**: Bug detection, fixing, and validation
- **Scope**: Common bug detection, automatic fixing, fix validation
- **Key Files**:
  - `bug_detector.py` - Detects common bugs and issues
  - `bug_fixer.py` - Fixes detected bugs
  - `bug_validator.py` - Validates fixes
- **Usage**: `python run_tests.py --category bug_fixing`

### **✅ Validation (`validation/`)**
- **Purpose**: General validation tools
- **Scope**: Model validation, data validation, configuration validation
- **Key Files**:
  - `model_validator.py` - Validates model files and structures
  - `data_validator.py` - Validates dataset and data formats
  - `config_validator.py` - Validates configuration files
- **Usage**: `python run_tests.py --category validation`

### **📊 Unit Tests (`unit/`)**
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

### **📊 Dataset Validation (`dataset_validation/`)**
- **Purpose**: Dataset validation and verification tools
- **Scope**: Dataset structure validation, final verification, data integrity checks
- **Examples**: Dataset validation, final verification, data integrity checks

### **🔍 Diagnostic (`diagnostic/`)**
- **Purpose**: Diagnose and identify issues
- **Scope**: Dataset diagnosis, format checking, structure validation, annotation verification
- **Examples**: Dataset diagnosis, segmentation format checking, dataset extraction checking, annotation verification

### **🐛 Bug Fixing (`bug_fixing/`)**
- **Purpose**: Fix specific dataset and code issues
- **Scope**: Dataset class fixing, ultralytics fixing, segmentation fixing, coordinate fixing
- **Examples**: Dataset class normalization, ultralytics dataset fixing, segmentation label fixing, YOLO coordinate fixing

### **🔗 Integration (`integration/`)**
- **Purpose**: Test integration and fixes
- **Scope**: Dataset fix integration, segmentation integration, configuration fixes
- **Examples**: Dataset fix integration tests, segmentation integration tests, configuration fix tests

### **🏋️ Training Tools (`training_tools/`)**
- **Purpose**: Training systems and tools
- **Scope**: Enhanced training systems, simple training, training import tests
- **Examples**: Enhanced training system, simple training system, training import validation

### **🛠️ Utility Tests (`utils/`)**
- **Purpose**: Environment and compatibility verification
- **Scope**: System requirements, dependencies, usage examples
- **Examples**: CUDA setup verification, Python version compatibility, usage examples, environment testing

### Existing Results Tests (`existing_results/`)
- **Purpose**: Test using existing training results without retraining
- **Scope**: Finding existing runs, analysis, inference, and RKNN conversion
- **Examples**: Safe analysis, inference visualization, model conversion

### ONNX Testing (`onnx_testing/`)
- **Purpose**: Structured ONNX testing and validation
- **Scope**: Environment setup, model validation, conversion testing, inference testing
- **Examples**: Environment checks (check0), model validation (check1), conversion testing (check2), inference testing (check3)

### Type Checking (`type_checking/`)
- **Purpose**: Type validation, checking, and fixing
- **Scope**: Data type validation, type compatibility checks, type fixing utilities
- **Examples**: Configuration type validation, model type checking, data type validation, path type validation

### Bug Fixing (`bug_fixing/`)
- **Purpose**: Bug detection, fixing, and validation
- **Scope**: Common bug detection, automatic fixing, fix validation
- **Examples**: Import bug detection, syntax bug detection, path bug detection, type bug detection, logic bug detection

### Validation (`validation/`)
- **Purpose**: General validation tools
- **Scope**: Model validation, data validation, configuration validation
- **Examples**: Model file validation, dataset format validation, configuration file validation

## 🔧 **ONNX Testing System (check0, check1, check2, check3)**

### **Overview**
The ONNX testing system provides comprehensive validation for ONNX model lifecycle, from environment setup to inference testing. It's organized into four systematic checks:

### **check0 - Environment Setup (`check_onnx_environment.py`)**
Validates the complete environment for ONNX operations:

**Checks Performed:**
- ✅ Python version compatibility (3.8+)
- ✅ PyTorch installation and CUDA support
- ✅ Ultralytics installation
- ✅ ONNX and ONNX Runtime installation
- ✅ OpenCV and NumPy installation
- ✅ Memory and GPU memory availability
- ✅ ONNX export capability testing
- ✅ Project structure validation

**Usage:**
```bash
python tests/onnx_testing/check_onnx_environment.py
```

### **check1 - Model Validation (`check_onnx_models.py`)**
Validates ONNX model files and their properties:

**Checks Performed:**
- ✅ ONNX model file existence
- ✅ ONNX file validity (load and validate)
- ✅ Model size validation (reasonable sizes)
- ✅ Model metadata validation (IR version, opset, producer)
- ✅ Model input specifications (shape, type, dimensions)
- ✅ Model output specifications (shape, type, dimensions)
- ✅ Model opset compatibility (version 11+)

**Usage:**
```bash
python tests/onnx_testing/check_onnx_models.py
```

### **check2 - Conversion Testing (`check_onnx_conversion.py`)**
Tests the ONNX conversion process and validates conversion results:

**Checks Performed:**
- ✅ PyTorch model existence
- ✅ Conversion environment validation
- ✅ Conversion process testing (actual export)
- ✅ Conversion output validation
- ✅ Conversion result validation (loadable models)
- ✅ Conversion performance testing (timing)

**Usage:**
```bash
python tests/onnx_testing/check_onnx_conversion.py
```

### **check3 - Inference Testing (`check_onnx_inference.py`)**
Tests ONNX model inference and performance:

**Checks Performed:**
- ✅ ONNX model existence
- ✅ Inference environment validation
- ✅ Inference process testing (actual inference)
- ✅ Inference accuracy testing (finite values, valid outputs)
- ✅ Inference performance testing (timing, statistics)
- ✅ Inference memory usage testing (memory consumption)

**Usage:**
```bash
python tests/onnx_testing/check_onnx_inference.py
```

### **Complete ONNX Testing Workflow:**
```bash
# Run all ONNX tests
python run_tests.py --category onnx_testing

# Run individual checks
python tests/onnx_testing/check_onnx_environment.py
python tests/onnx_testing/check_onnx_models.py
python tests/onnx_testing/check_onnx_conversion.py
python tests/onnx_testing/check_onnx_inference.py
```

## 🐛 **Bug Detection System**

### **Overview**
The bug detection system identifies common issues in the codebase using static analysis and pattern matching.

### **Bug Types Detected:**

#### **Import Bugs:**
- Unused imports
- Circular imports
- Missing imports (yaml, torch, etc.)

#### **Syntax Bugs:**
- Missing colons after control structures
- Unmatched parentheses
- Indentation issues

#### **Path Bugs:**
- Hardcoded paths
- Relative path issues
- Missing path joins

#### **Type Bugs:**
- Type mismatches in concatenation
- Potential division by zero
- Index out of bounds access

#### **Logic Bugs:**
- Unreachable code after return
- Potential infinite loops
- Empty except blocks
- File operations without context managers

### **Usage:**
```bash
# Run bug detection
python run_tests.py --category bug_fixing

# Run specific bug detector
python tests/bug_fixing/bug_detector.py
```

## 🔍 **Type Checking System**

### **Overview**
The type checking system validates data types and structures throughout the project.

### **Validation Areas:**

#### **Configuration Types:**
- YAML configuration file validation
- Model configuration type checking
- Dataset configuration validation
- Training configuration verification

#### **Model Types:**
- ModelProcessor attribute validation
- Path attribute type checking
- Configuration attribute validation

#### **Data Types:**
- Dataset structure validation
- Required file existence
- Data.yaml structure validation

#### **Path Types:**
- Required directory validation
- File permission checking
- Path accessibility testing

### **Usage:**
```bash
# Run type checking
python run_tests.py --category type_checking

# Run specific type validator
python tests/type_checking/type_validator.py
```

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

## 🚀 **Quick Reference**

### **Test Categories:**
```bash
# Core testing categories
python run_tests.py --category onnx_testing      # ONNX testing (check0-3)
python run_tests.py --category type_checking     # Type validation
python run_tests.py --category bug_fixing        # Bug detection
python run_tests.py --category validation        # General validation

# Traditional categories
python run_tests.py --category unit              # Unit tests
python run_tests.py --category integration       # Integration tests
python run_tests.py --category diagnostic        # Diagnostic scripts
python run_tests.py --category dataset_validation # Dataset validation
python run_tests.py --category training_tools    # Training tools
python run_tests.py --category existing_results  # Existing results tests
python run_tests.py --category utils             # Utility tests
```

### **Individual ONNX Checks:**
```bash
python tests/onnx_testing/check_onnx_environment.py  # Environment (check0)
python tests/onnx_testing/check_onnx_models.py       # Models (check1)
python tests/onnx_testing/check_onnx_conversion.py   # Conversion (check2)
python tests/onnx_testing/check_onnx_inference.py    # Inference (check3)
```

### **Individual Test Modules:**
```bash
python tests/bug_fixing/bug_detector.py              # Bug detection
python tests/type_checking/type_validator.py         # Type validation
python tests/validation/model_validator.py           # Model validation
```

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

# ONNX testing only
python run_tests.py --category onnx_testing

# Type checking only
python run_tests.py --category type_checking

# Bug fixing only
python run_tests.py --category bug_fixing

# Validation only
python run_tests.py --category validation
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