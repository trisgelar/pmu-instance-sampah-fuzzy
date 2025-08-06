# Existing Results Test Module

This module contains scripts for using existing training results without retraining.

## Files

### `use_existing_training.py`
Full-featured script with command-line arguments for using existing training results.

**Features:**
- Command-line argument parsing
- Configurable model version and number of inference images
- Complete pipeline: analysis, inference, and RKNN conversion

**Usage:**
```bash
# Use v8n with 6 images (default)
./run_with_venv.sh python tests/existing_results/use_existing_training.py

# Use v10n with 3 images
./run_with_venv.sh python tests/existing_results/use_existing_training.py --model v10n --images 3

# Use v11n with 10 images
./run_with_venv.sh python tests/existing_results/use_existing_training.py --model v11n --images 10
```

### `use_existing_results.py`
Simple script that uses the most recent training run without prompts.

**Features:**
- No command-line arguments needed
- Uses v8n by default
- Runs complete pipeline

**Usage:**
```bash
./run_with_venv.sh python tests/existing_results/use_existing_results.py
```

### `use_existing_results_safe.py`
Safe version that skips problematic inference visualization.

**Features:**
- Skips RGBA visualization issues
- Focuses on analysis and RKNN conversion
- Most reliable option

**Usage:**
```bash
./run_with_venv.sh python tests/existing_results/use_existing_results_safe.py
```

## Integration

The functions can be imported and used in other modules:

```python
from tests.existing_results import use_existing_training, use_existing_results, use_existing_results_safe

# Use with arguments
success = use_existing_training("v8n", 6)

# Use simple version
success = use_existing_results("v8n")

# Use safe version
success = use_existing_results_safe("v8n")
```

## Output

All scripts will:
1. Find existing training runs
2. Generate training metrics plots
3. Display performance metrics
4. Convert models to RKNN format (when possible)
5. Save results to appropriate directories

## Error Handling

- Gracefully handles missing training runs
- Provides clear error messages
- Safe version skips problematic visualization 