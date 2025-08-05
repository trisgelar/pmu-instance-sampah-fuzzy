# Dataset Tools Module

This module contains comprehensive tools for diagnosing and fixing dataset issues, particularly for YOLO training with COCO annotations.

## 🎯 Overview

The dataset tools help you:
- **Diagnose** dataset configuration issues
- **Fix** class configuration problems
- **Validate** dataset structure and format
- **Convert** COCO annotations to proper YOLO format

## 📁 Structure

```
tests/dataset_tools/
├── __init__.py              # Module initialization
├── README.md               # This file
├── diagnose_dataset.py      # Dataset diagnostic tool
├── fix_dataset_classes.py   # Manual fix for class issues
├── fix_dataset_ultralytics.py # Fix using Ultralytics tools
├── dataset_validator.py    # Comprehensive validation
├── extract_and_check_dataset.py # Extract and verify dataset structure
├── final_verification.py   # Final dataset verification for YOLO training
├── fix_yolo_coordinates.py # Fix coordinate normalization issues
├── check_segmentation_format.py # Check dataset format for segmentation
├── fix_segmentation_labels.py # Convert labels to segmentation format
├── test_dataset_fix_integration.py # Integration test for DatasetManager
└── run_dataset_fix_test.py # Test runner script
```

## 🛠️ Tools

### 1. Dataset Diagnostic (`diagnose_dataset.py`)

**Purpose**: Identify issues with your dataset configuration

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.diagnose_dataset

# Python import
from tests.dataset_tools import diagnose_dataset
results = diagnose_dataset()
```

**What it checks**:
- ✅ `data.yaml` configuration
- ✅ COCO annotation files
- ✅ Roboflow project settings
- ✅ Class configuration issues

### 2. Dataset Validator (`dataset_validator.py`)

**Purpose**: Comprehensive validation of dataset structure and format

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.dataset_validator

# Python import
from tests.dataset_tools import validate_dataset
results = validate_dataset()
```

**What it validates**:
- 📁 Directory structure
- 📄 `data.yaml` format and content
- 📋 COCO annotation format
- 🖼️ Image file integrity
- 🏷️ Class configuration

### 3. Manual Fix (`fix_dataset_classes.py`)

**Purpose**: Fix class issues by modifying existing COCO files

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.fix_dataset_classes

# Python import
from tests.dataset_tools import fix_dataset_classes
results = fix_dataset_classes()
```

**What it does**:
- 🔧 Normalizes COCO annotations to use only 'sampah' category
- 📄 Updates `data.yaml` to use only 'sampah' class
- ✅ Verifies the fix was successful

### 4. Ultralytics Fix (`fix_dataset_ultralytics.py`) ⭐ **RECOMMENDED**

**Purpose**: Convert COCO to proper YOLO format using official Ultralytics tools

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.fix_dataset_ultralytics

# Python import
from tests.dataset_tools import fix_dataset_ultralytics
results = fix_dataset_ultralytics()
```

**What it does**:
- 🔄 Converts COCO annotations to YOLO format
- 📁 Creates proper directory structure (`images/` and `labels/`)
- 📄 Creates correct `data.yaml` configuration
- 💾 Creates backup of original dataset
- ✅ Verifies the conversion

### 5. Dataset Extraction and Check (`extract_and_check_dataset.py`)

**Purpose**: Extract dataset zip file and verify its structure

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.extract_and_check_dataset

# Python import
from tests.dataset_tools import extract_and_check_dataset
results = extract_and_check_dataset()
```

**What it does**:
- 📦 Extracts `datasets.zip` file
- 🔍 Checks extracted dataset structure
- 📊 Reports image and label counts
- ✅ Verifies COCO JSON files exist
- 📄 Validates `data.yaml` configuration

### 6. Final Verification (`final_verification.py`)

**Purpose**: Comprehensive final verification before YOLO training

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.final_verification

# Python import
from tests.dataset_tools import final_verification
results = final_verification()
```

**What it does**:
- 🔍 Checks complete dataset structure
- 📸 Validates image-label matching
- 📄 Verifies `data.yaml` paths and classes
- 🎯 Ensures coordinates are within bounds
- ✅ Confirms readiness for YOLO training

### 7. Coordinate Fix (`fix_yolo_coordinates.py`)

**Purpose**: Fix "non-normalized or out of bounds coordinates" errors

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.fix_yolo_coordinates

# Python import
from tests.dataset_tools import fix_yolo_coordinates
results = fix_yolo_coordinates()
```

**What it does**:
- 🔧 Re-generates YOLO labels with precise coordinate normalization
- 📐 Uses actual image dimensions from COCO JSON
- ✅ Ensures all coordinates are within 0-1 range
- 🎯 Fixes "out of bounds coordinates" training errors
- 📊 Reports coordinate statistics

### 8. Segmentation Format Check (`check_segmentation_format.py`)

**Purpose**: Check if dataset is properly formatted for YOLO segmentation training

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.check_segmentation_format

# Python import
from tests.dataset_tools import check_segmentation_format
results = check_segmentation_format()
```

**What it does**:
- 🔍 Analyzes dataset structure for segmentation compatibility
- 📋 Checks COCO annotations for segmentation data
- 📄 Validates data.yaml configuration
- 🏷️ Verifies YOLO label format
- ✅ Confirms readiness for segmentation training

### 9. Segmentation Labels Fix (`fix_segmentation_labels.py`)

**Purpose**: Convert YOLO detection labels to segmentation labels

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.fix_segmentation_labels

# Python import
from tests.dataset_tools import fix_segmentation_labels
results = fix_segmentation_labels()
```

**What it does**:
- 🔄 Converts detection labels to segmentation format
- 📐 Uses polygon coordinates from COCO annotations
- 🎯 Normalizes coordinates to 0-1 range
- ✅ Creates proper YOLO segmentation labels
- 📊 Reports conversion statistics

### 10. Segmentation Integration Test (`test_segmentation_integration.py`)

**Purpose**: Test the integration of segmentation fixing into DatasetManager class

**Usage**:
```bash
# Command line
python -m tests.dataset_tools.test_segmentation_integration

# Python import
from tests.dataset_tools import test_segmentation_integration
success = test_segmentation_integration()
```

**What it tests**:
- ✅ Polygon coordinate normalization
- ✅ Invalid polygon handling
- ✅ Out of bounds coordinate clamping
- ✅ Integration with DatasetManager class
- ✅ Method availability and functionality

## 🚀 Quick Start

### Step 1: Diagnose Your Dataset
```bash
python -m tests.dataset_tools.diagnose_dataset
```

### Step 2: Fix Issues (Choose One)

**Option A: Use Ultralytics Tools (Recommended)**
```bash
python -m tests.dataset_tools.fix_dataset_ultralytics
```

**Option B: Manual Fix**
```bash
python -m tests.dataset_tools.fix_dataset_classes
```

### Step 3: Validate the Fix
```bash
python -m tests.dataset_tools.dataset_validator
```

## 📋 Common Issues and Solutions

### Issue: "Multiple classes found in training results"
**Solution**: Use `fix_dataset_ultralytics.py` to convert to proper YOLO format

### Issue: "No label found in segment set"
**Solution**: Use `fix_dataset_ultralytics.py` to convert COCO to YOLO format

### Issue: "data.yaml has wrong classes"
**Solution**: Use `fix_dataset_classes.py` or `fix_dataset_ultralytics.py`

### Issue: "COCO file not found"
**Solution**: Check dataset path and run `diagnose_dataset.py` to identify issues

## 🔧 Advanced Usage

### Programmatic Usage
```python
from tests.dataset_tools import (
    diagnose_dataset,
    validate_dataset,
    fix_dataset_ultralytics,
    fix_dataset_classes
)

# Diagnose issues
diagnosis = diagnose_dataset(verbose=False)
if diagnosis.get('has_issues'):
    print(f"Found {len(diagnosis['all_issues'])} issues")

# Fix using Ultralytics tools
results = fix_dataset_ultralytics(verbose=False)
if results.get('success'):
    print("Dataset fixed successfully!")

# Validate the fix
validation = validate_dataset(verbose=False)
if validation['valid']:
    print("Dataset is ready for training!")
```

### Custom Dataset Path
```python
# Specify custom dataset path
results = diagnose_dataset(dataset_path="path/to/your/dataset")
fix_results = fix_dataset_ultralytics(dataset_path="path/to/your/dataset")
```

## 📊 Expected Results

### After Successful Fix
- ✅ Only "sampah" class in confusion matrix
- ✅ Proper YOLO directory structure
- ✅ Clean training results
- ✅ Backup preserved

### Directory Structure (After Ultralytics Fix)
```
datasets/your_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🎯 Which Tool to Use?

| Scenario | Recommended Tool |
|----------|------------------|
| **First time setup** | `extract_and_check_dataset.py` → `fix_dataset_ultralytics.py` |
| **Quick fix** | `fix_dataset_ultralytics.py` |
| **Manual fix** | `fix_dataset_classes.py` |
| **Coordinate issues** | `fix_yolo_coordinates.py` |
| **Segmentation format** | `check_segmentation_format.py` |
| **Segmentation labels** | `fix_segmentation_labels.py` |
| **Final verification** | `final_verification.py` |
| **Validation** | `dataset_validator.py` |
| **Troubleshooting** | `diagnose_dataset.py` |

## 🧪 Integration Tests

### 5. Dataset Fix Integration Test (`test_dataset_fix_integration.py`)

**Purpose**: Test the integrated dataset fixing functionality in DatasetManager

**Usage**:
```bash
# Direct execution
python tests/dataset_tools/test_dataset_fix_integration.py

# Using runner script
python tests/dataset_tools/run_dataset_fix_test.py
```

**What it tests**:
- ✅ Dataset validation functionality
- ✅ Dataset fixing with Ultralytics tools
- ✅ Dataset preparation workflows
- ✅ Individual method testing
- ✅ Error handling and recovery

### 6. Test Runner (`run_dataset_fix_test.py`)

**Purpose**: Convenient runner script for integration tests

**Usage**:
```bash
# Run integration test with proper setup
python tests/dataset_tools/run_dataset_fix_test.py
```

**Features**:
- 🔧 Automatic path setup
- 📁 Project root detection
- ⚡ Error handling
- 📊 Test result reporting

## 🔍 Troubleshooting

### Error: "Ultralytics not installed"
```bash
pip install ultralytics
```

### Error: "Dataset path not found"
- Run from project root directory
- Check if dataset exists in expected location
- Use `diagnose_dataset.py` to find issues

### Error: "Backup failed"
- Check disk space
- Ensure write permissions
- Try manual backup first

## 📚 References

- [Ultralytics JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)
- [Ultralytics Segmentation Documentation](https://docs.ultralytics.com/datasets/segment/)
- [COCO Format Specification](https://cocodataset.org/#format-data)

## 🤝 Contributing

To add new tools or improve existing ones:

1. Add your script to `tests/dataset_tools/`
2. Update `__init__.py` with imports
3. Add documentation to this README
4. Test with your dataset

## 📝 License

This module is part of the main project and follows the same license terms. 