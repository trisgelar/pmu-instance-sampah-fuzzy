# Training Test Module

This module contains enhanced training systems for testing purposes, providing checkpoint functionality and separate execution modes for local and Colab environments.

## Files

### `enhanced_training_system.py`
Full-featured training system with checkpoint support that integrates with the main project components.

**Features:**
- Checkpoint saving/loading for long training sessions
- Separate execution modes for local and Colab environments
- Integration with main project modules
- Dataset preparation and validation
- Comprehensive logging and status monitoring

**Usage:**
```bash
# Local training
python tests/training/enhanced_training_system.py --mode local --model v8n --epochs 200

# Colab training
python tests/training/enhanced_training_system.py --mode colab --model v8n --epochs 200

# Dataset preparation only
python tests/training/enhanced_training_system.py --mode dataset
```

### `simple_enhanced_training.py`
Simplified training system for testing checkpoint functionality without complex dependencies.

**Features:**
- Simulated training with checkpoint support
- Lightweight implementation for testing
- No external dependencies beyond standard library
- Demonstrates checkpoint workflow

**Usage:**
```bash
# Local simulated training
python tests/training/simple_enhanced_training.py --mode local --model v8n --epochs 50

# Colab simulated training
python tests/training/simple_enhanced_training.py --mode colab --model v8n --epochs 50
```

## Checkpoint System

Both training systems implement a checkpoint system that:

1. **Saves training state** at regular intervals
2. **Resumes training** from the last checkpoint if interrupted
3. **Tracks progress** with timestamps and status information
4. **Handles errors** gracefully with error checkpoints

### Checkpoint Data Structure
```json
{
  "model_version": "v8n",
  "epoch": 150,
  "status": "in_progress",
  "timestamp": "2024-01-01T12:00:00",
  "additional_info": {
    "current_epoch": 150,
    "total_epochs": 200
  }
}
```

## Integration

The training systems can be imported and used in other test modules:

```python
from tests.training import EnhancedTrainingSystem, SimpleEnhancedTrainingSystem

# Use enhanced system
enhanced = EnhancedTrainingSystem()
success = enhanced.execute_local_training(model_version="v8n", epochs=200)

# Use simple system for testing
simple = SimpleEnhancedTrainingSystem()
success = simple.execute_local_training(model_version="v8n", epochs=50)
```

## Logging

Both systems provide comprehensive logging:
- Training progress updates
- Checkpoint operations
- Error handling and recovery
- System status information

Log files are created in the project root:
- `enhanced_training.log` for the full system
- `simple_enhanced_training.log` for the simple system 