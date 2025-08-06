# Enhanced Training System Guide

This guide explains how to use the enhanced training system with checkpoint support for stable long-term training sessions.

## 🚀 Overview

The enhanced training system provides:
- **Separate execution modes** for local and Colab environments
- **Checkpoint support** for resuming interrupted training
- **Enhanced monitoring** and logging
- **Stable long-term training** for high epoch counts (200+ epochs)

## 📁 File Structure

```
enhanced_training_system.py    # Main enhanced training system
checkpoints/                   # Checkpoint directory (auto-created)
├── training_state.json       # Training state checkpoint
enhanced_training.log         # Enhanced training logs
```

## 🎯 Execution Modes

### 1. Local Training Mode
```bash
# Basic local training
python enhanced_training_system.py --mode local

# Custom parameters
python enhanced_training_system.py --mode local --model v8n --epochs 200 --batch-size 16
```

### 2. Google Colab Training Mode
```bash
# Basic Colab training
python enhanced_training_system.py --mode colab

# Custom parameters for Colab Pro with 24GB GPU
python enhanced_training_system.py --mode colab --model v8n --epochs 300 --batch-size 32
```

### 3. Dataset Preparation Mode
```bash
# Dataset preparation only
python enhanced_training_system.py --mode dataset
```

## 🔧 Checkpoint System

### How Checkpoints Work
1. **Automatic Saving**: Checkpoints are saved at key points during training
2. **Resume Capability**: Training can resume from the last checkpoint
3. **State Tracking**: Tracks model version, epoch, status, and timestamp
4. **Error Recovery**: Saves state even when errors occur

### Checkpoint States
- `started`: Training has begun
- `completed`: Training finished successfully
- `failed`: Training failed
- `error`: Training encountered an error

### Manual Checkpoint Management
```python
from enhanced_training_system import EnhancedTrainingSystem

# Initialize system
enhanced_system = EnhancedTrainingSystem()

# Check current status
status = enhanced_system.get_training_status()
print(f"Checkpoint exists: {status['checkpoint_exists']}")
if status['checkpoint_exists']:
    print(f"Model: {status['checkpoint_model']}")
    print(f"Epoch: {status['checkpoint_epoch']}")
    print(f"Status: {status['checkpoint_status']}")

# Clear checkpoint to start fresh
enhanced_system.clear_checkpoint()
```

## 🎮 Usage Examples

### Example 1: Local Training with 200 Epochs
```python
from enhanced_training_system import main_local

# Run local training
main_local()
```

### Example 2: Colab Training with Custom Parameters
```python
from enhanced_training_system import EnhancedTrainingSystem

# Initialize system
enhanced_system = EnhancedTrainingSystem()

# Execute Colab training with custom parameters
success = enhanced_system.execute_colab_training(
    model_version="v8n",
    epochs=300,  # High epoch count for Colab Pro
    batch_size=32  # Larger batch size for 24GB GPU
)

if success:
    print("🎉 Training completed successfully!")
else:
    print("💥 Training failed!")
```

### Example 3: Dataset Preparation Only
```python
from enhanced_training_system import main_dataset_prep

# Run dataset preparation
main_dataset_prep()
```

## 📊 Monitoring and Logging

### Log Files
- `enhanced_training.log`: Detailed training logs
- `checkpoints/training_state.json`: Checkpoint state

### Log Levels
- **INFO**: General progress information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and failures

### Real-time Monitoring
```python
# Monitor training status
status = enhanced_system.get_training_status()
print(f"System Status: {status['system_status']}")
print(f"CUDA Status: {status['cuda_status']}")
```

## 🔄 Resuming Training

### Automatic Resume
The system automatically detects and resumes from checkpoints:
```python
# Training will automatically resume from checkpoint if available
enhanced_system.execute_local_training(
    model_version="v8n",
    epochs=200,
    resume_from_checkpoint=True  # Default: True
)
```

### Manual Resume Control
```python
# Force fresh start (ignore checkpoints)
enhanced_system.clear_checkpoint()
enhanced_system.execute_local_training(
    model_version="v8n",
    epochs=200,
    resume_from_checkpoint=False
)
```

## 🎯 High-Epoch Training (200+ Epochs)

### For Google Colab Pro (24GB GPU)
```bash
# Optimal settings for Colab Pro
python enhanced_training_system.py \
    --mode colab \
    --model v8n \
    --epochs 300 \
    --batch-size 32
```

### For Local Environment
```bash
# Conservative settings for local training
python enhanced_training_system.py \
    --mode local \
    --model v8n \
    --epochs 200 \
    --batch-size 16
```

### Training Stability Tips
1. **Use checkpoints**: Always enable checkpoint resuming
2. **Monitor logs**: Check `enhanced_training.log` regularly
3. **Clear checkpoints**: Start fresh when changing model versions
4. **Batch size**: Adjust based on GPU memory
5. **Epochs**: Start with lower counts, increase gradually

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python enhanced_training_system.py --mode colab --batch-size 16
```

#### 2. Training Interrupted
```bash
# Resume from checkpoint (automatic)
python enhanced_training_system.py --mode colab
```

#### 3. Checkpoint Corruption
```bash
# Clear checkpoint and start fresh
python -c "from enhanced_training_system import EnhancedTrainingSystem; EnhancedTrainingSystem().clear_checkpoint()"
```

#### 4. Model Loading Issues
```bash
# Check model files exist
ls -la yolov8n-seg.pt
```

### Error Recovery
```python
# Check training status
status = enhanced_system.get_training_status()
if status['checkpoint_exists']:
    print(f"Last training: {status['checkpoint_model']} at epoch {status['checkpoint_epoch']}")
    print(f"Status: {status['checkpoint_status']}")
```

## 📈 Performance Optimization

### For Colab Pro (24GB GPU)
- **Batch Size**: 32-64 (depending on model size)
- **Epochs**: 200-500 (for thorough training)
- **Image Size**: 640x640 (default, can increase to 832x832)

### For Local Environment
- **Batch Size**: 8-16 (depending on GPU memory)
- **Epochs**: 100-200 (for reasonable training time)
- **Image Size**: 640x640 (default)

### Memory Management
```python
# Clear CUDA cache before training
enhanced_system.system.clear_cuda_cache()

# Monitor GPU memory
cuda_status = enhanced_system.system.get_cuda_status()
print(f"GPU Memory: {cuda_status}")
```

## 🔍 Advanced Usage

### Custom Training Configuration
```python
from enhanced_training_system import EnhancedTrainingSystem

# Initialize with custom config
enhanced_system = EnhancedTrainingSystem("custom_config.yaml")

# Custom training with all parameters
success = enhanced_system.train_with_checkpoints(
    model_version="v10n",
    epochs=500,
    batch_size=64,
    resume_from_checkpoint=True
)
```

### Integration with Existing Code
```python
# Use enhanced system with existing main_colab.py
from enhanced_training_system import EnhancedTrainingSystem
from main_colab import WasteDetectionSystemColab

# Initialize both systems
enhanced_system = EnhancedTrainingSystem()
original_system = WasteDetectionSystemColab()

# Use enhanced training
run_dir = enhanced_system.train_with_checkpoints("v8n", epochs=200)

# Use original system for analysis
if run_dir:
    original_system.analyze_training_run(run_dir, "v8n")
```

## 📋 Best Practices

1. **Always use checkpoints** for long training sessions
2. **Monitor logs** regularly during training
3. **Start with lower epochs** and increase gradually
4. **Clear checkpoints** when changing model versions
5. **Use appropriate batch sizes** for your GPU memory
6. **Backup important checkpoints** before clearing them
7. **Test with small epochs** before long training sessions

## 🎉 Success Indicators

- ✅ Checkpoint saved successfully
- ✅ Training completed without errors
- ✅ Model files generated in `results/runs/segment/`
- ✅ ONNX export completed
- ✅ Logs show successful completion

## 🚨 Error Handling

The enhanced system provides robust error handling:
- **Automatic checkpoint saving** on errors
- **Detailed error logging** with timestamps
- **Graceful failure recovery** with status tracking
- **Clear error messages** for troubleshooting

This enhanced training system ensures stable, long-term training sessions with automatic recovery and comprehensive monitoring! 🚀 