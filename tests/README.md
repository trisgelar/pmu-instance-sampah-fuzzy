# Test Structure

This directory contains all tests for the Waste Detection System, organized into logical categories for better maintainability and clarity.

## 📁 Directory Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_config_manager.py
│   ├── test_exceptions.py
│   ├── test_fuzzy_area_classifier.py
│   └── test_secrets_validation.py
├── integration/             # Integration tests for system-wide functionality
│   ├── test_main_colab.py
│   └── test_dataset_fix.py
├── diagnostic/              # Diagnostic scripts for troubleshooting
│   ├── check_categories.py
│   ├── check_dataset_annotations.py
│   ├── check_polygon_segmentation.py
│   └── check_cuda_versions.py
├── fixes/                   # Fix verification scripts
│   ├── test_config_fix.py
│   ├── test_onnx_export_fix.py
│   ├── test_sampah_normalization.py
│   └── test_sampah_only.py
└── utils/                   # Utility test scripts
    ├── test_cuda.py
    └── verify_python311_compatibility.py
```

## 🧪 Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual modules and components in isolation
- **Scope**: Single functions, classes, or small groups of related functionality
- **Examples**: Configuration management, exception handling, fuzzy logic

### Integration Tests (`integration/`)
- **Purpose**: Test how different modules work together
- **Scope**: End-to-end workflows and system-wide functionality
- **Examples**: Main system initialization, dataset processing workflows

### Diagnostic Tests (`diagnostic/`)
- **Purpose**: Troubleshoot specific issues and verify system state
- **Scope**: Problem identification and debugging
- **Examples**: Dataset validation, CUDA version checking, category analysis

### Fix Verification Tests (`fixes/`)
- **Purpose**: Verify that specific fixes work correctly
- **Scope**: Targeted testing of bug fixes and improvements
- **Examples**: ONNX export fixes, configuration fixes, normalization fixes

### Utility Tests (`utils/`)
- **Purpose**: Environment and compatibility verification
- **Scope**: System requirements and dependencies
- **Examples**: CUDA setup verification, Python version compatibility

## 🚀 Running Tests

### Run All Tests
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

# Utility tests only
python run_tests.py --category utils
```

### Run Specific Tests
```bash
# Run a specific test file
python run_tests.py --test config_manager

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
   - `utils/` for environment checks

2. **Follow naming conventions**:
   - Unit/Integration tests: `test_*.py`
   - Diagnostic scripts: `check_*.py`
   - Fix verification: `test_*_fix.py`
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestClassName(unittest.TestCase):
    """Test class description."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_specific_functionality(self):
        """Test specific functionality."""
        # Test implementation
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass

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