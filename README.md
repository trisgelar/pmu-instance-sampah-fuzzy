# Waste Detection System with Fuzzy Classification

A comprehensive instance segmentation system for street litter detection using YOLOv8-YOLOv11 models with fuzzy logic classification for litter quantity estimation.

## 🎯 Project Overview

This system combines state-of-the-art computer vision techniques with fuzzy logic to detect and classify street litter. It uses YOLO models for instance segmentation and fuzzy logic to categorize litter quantity into three levels: "sedikit" (little), "sedang" (medium), and "banyak" (much).

## ✨ Key Features

- **Instance Segmentation**: YOLOv8-YOLOv11 models for precise litter detection
- **Fuzzy Classification**: Intelligent categorization of litter quantity
- **Robust Error Handling**: Comprehensive exception management
- **Configuration Management**: Centralized parameter management
- **Security**: Secure secrets management and validation
- **Testing**: Comprehensive unit test suite
- **Google Colab Integration**: Optimized for Colab environment
- **Model Export**: ONNX and RKNN model conversion (optional)

## 🏗️ System Architecture

```
pmu-instance-sampah-fuzzy/
├── main_colab.py              # Main orchestrator (reorganized with functions)
├── config.yaml               # Configuration file
├── secrets.yaml              # Secrets (not in repo)
├── requirements.txt           # Dependencies
├── modules/                  # Core modules
│   ├── config_manager.py     # Configuration management
│   ├── dataset_manager.py    # Dataset operations
│   ├── model_processor.py    # Model training & export
│   ├── fuzzy_area_classifier.py # Fuzzy classification
│   ├── inference_visualizer.py  # Inference & visualization
│   ├── metrics_analyzer.py   # Training metrics
│   ├── rknn_converter.py     # RKNN model conversion
│   ├── drive_manager.py      # Google Drive integration
│   └── exceptions.py         # Custom exceptions
├── setup/                    # Setup utilities
│   └── cuda/                 # CUDA setup and installation
│       ├── install_cuda.py   # Automated CUDA installation
│       ├── setup_cuda_environment.py  # Environment configuration
│       ├── setup_cuda_env.bat # Windows batch setup
│       └── requirements-*.txt # CUDA dependencies
├── tests/                    # Comprehensive test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── diagnostic/           # Diagnostic scripts
│   ├── fixes/                # Fix verification
│   ├── dataset_tools/        # Dataset validation and fixing
│   ├── training/             # Training system tests
│   ├── existing_results/     # Existing results usage tests
│   ├── onnx_testing/         # Structured ONNX testing (check0-3)
│   ├── type_checking/        # Type validation and checking
│   ├── bug_fixing/           # Bug detection and fixing
│   ├── validation/           # General validation tools
│   └── utils/                # Utility tests
├── run_tests.py              # Test runner
├── validate_secrets.py       # Secrets validation
└── docs/                     # Documentation
    ├── API_REFERENCE.md
    ├── DEPLOYMENT_GUIDE.md
    ├── TROUBLESHOOTING.md
    └── CONTRIBUTING.md
```

## 🧪 Testing System

### **Comprehensive Test Suite**

The project includes a comprehensive testing system organized into logical categories:

#### **🔧 ONNX Testing (`tests/onnx_testing/`)**
- **check0** - Environment setup validation
- **check1** - Model file validation  
- **check2** - Conversion process testing
- **check3** - Inference testing

#### **🔍 Type Checking (`tests/type_checking/`)**
- Type validation and compatibility checking
- Data structure validation
- Configuration type verification

#### **🐛 Bug Fixing (`tests/bug_fixing/`)**
- Common bug detection (imports, syntax, paths, types, logic)
- Automatic bug fixing capabilities
- Fix validation and verification

#### **✅ Validation (`tests/validation/`)**
- Model validation
- Data format validation
- Configuration validation

#### **📊 Other Test Categories**
- **Unit Tests** (`unit/`) - Individual module testing
- **Integration Tests** (`integration/`) - System-wide testing
- **Diagnostic Tests** (`diagnostic/`) - Troubleshooting scripts
- **Dataset Tools** (`dataset_tools/`) - Dataset validation and fixing
- **Training Tests** (`training/`) - Training system validation
- **Existing Results** (`existing_results/`) - Using existing training results

### **Running Tests**

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --category onnx_testing
python run_tests.py --category type_checking
python run_tests.py --category bug_fixing
python run_tests.py --category validation

# Run individual ONNX checks
python tests/onnx_testing/check_onnx_environment.py
python tests/onnx_testing/check_onnx_models.py
python tests/onnx_testing/check_onnx_conversion.py
python tests/onnx_testing/check_onnx_inference.py

# Run specific test
python run_tests.py --test config_manager

# Verbose output
python run_tests.py --category unit --verbose
```

### **Test Documentation**
- **[Tests README](tests/README.md)** - Comprehensive test documentation
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Complete testing guide and reference
- **[Code Structure Memory](docs/CODE_STRUCTURE_MEMORY.md)** - Testing organization memory
- **[Code Restructure Summary](docs/CODE_RESTRUCTURE_SUMMARY.md)** - Implementation details

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Google Colab (recommended) or local GPU setup
- Roboflow API key
- NVIDIA GPU with CUDA support (optional, for local training)
- RKNN-Toolkit2 (optional, for RKNN model conversion)

### 2. Installation

#### Option A: Google Colab (Recommended)
```python
# Quick setup in Google Colab
!git clone https://github.com/trisgelar/pmu-instance-sampah-fuzzy.git
%cd pmu-instance-sampah-fuzzy
!pip install -r requirements.txt

# Create secrets.yaml with your API key
import yaml
with open('secrets.yaml', 'w') as f:
    yaml.dump({'roboflow_api_key': 'YOUR_API_KEY'}, f)

# Initialize and run
from main_colab import WasteDetectionSystemColab
system = WasteDetectionSystemColab()
system.train_and_export_model("v8n", epochs=50, batch_size=16)
```

**📖 For detailed Colab instructions, see [COLAB_EXECUTION_GUIDE.md](docs/COLAB_EXECUTION_GUIDE.md)**

#### Option B: Local Setup (CPU/GPU)
```bash
# Clone the repository
git clone https://github.com/trisgelar/pmu-instance-sampah-fuzzy.git
cd pmu-instance-sampah-fuzzy

# Install dependencies
pip install -r requirements.txt
```

#### Option C: CUDA Setup (Local GPU Training)
```bash
# Clone the repository
git clone https://github.com/trisgelar/pmu-instance-sampah-fuzzy.git
cd pmu-instance-sampah-fuzzy

# Automated CUDA installation
python setup/cuda/install_cuda.py

# Or setup existing CUDA environment
python setup/cuda/setup_cuda_environment.py

# Or use Windows batch file
setup/cuda/setup_cuda_env.bat

# Or manually install CUDA-enabled PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify CUDA Installation
```bash
# Test CUDA setup
python tests/utils/test_cuda.py

# Check CUDA status in your system
python -c "from main_colab import WasteDetectionSystemColab; system = WasteDetectionSystemColab(); print(system.get_cuda_status())"
```

### 3. Configuration

1. **Set up secrets**:
   ```bash
   # Copy and edit secrets template
   cp secrets.yaml.example secrets.yaml
   ```

2. **Validate configuration**:
   ```bash
   python validate_secrets.py
   ```

3. **Run tests**:
   ```bash
   python run_tests.py
   ```

4. **Start the system**:
   ```bash
   python main_colab.py
   ```

## 🔧 Troubleshooting

### Common Issues

**Error: "img_size must be a tuple of two integers"**
- This has been fixed in the latest version
- The configuration now automatically converts lists to tuples
- If you encounter this error, run: `python test_config_fix.py` to verify the fix

**CUDA Issues**
- Run `python check_cuda_versions.py` to diagnose CUDA problems
- Use `python setup_cuda_environment.py` for local CUDA setup
- For Google Colab, CUDA is typically pre-configured

**Configuration Issues**
- Check `config.yaml` for valid parameters
- Run `python validate_secrets.py` to verify secrets
- Use `python test_config_fix.py` to test configuration loading

**RKNN-Toolkit2 Issues**
- RKNN-Toolkit2 is optional and only needed for RKNN model conversion
- If you get RKNN import errors, the system will skip RKNN conversion
- To install RKNN-Toolkit2:
  ```bash
  # Download RKNN-Toolkit2 repository
  git clone -b v2.3.0 https://github.com/airockchip/rknn-toolkit2.git
  # Download RKNN Model Zoo repository
  git clone -b v2.3.0 https://github.com/airockchip/rknn_model_zoo.git

  # Enter rknn-toolkit2 directory
  cd rknn-toolkit2/rknn-toolkit2/packages/x86_64/
  # Please select the appropriate requirements file based on your Python version; here it is for python3.11
  pip3 install -r requirements_cp11-2.3.0.txt
  # Please select the appropriate wheel installation package based on your Python version and processor architecture:
  pip3 install ./rknn_toolkit2-2.3.0-cp11-cp11-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  ```

**PyTorch 2.6+ Model Loading Issues**
- If you get "Weights only load failed" or "Unsupported global" errors, this is due to PyTorch 2.6+ security changes
- The system now automatically handles this with safe globals and fallback methods
- Test the fix: `python test_pytorch_compatibility.py`
- If issues persist, try: `pip install --upgrade ultralytics`

### 2. Installation

#### Option A: Quick Setup (CPU/Google Colab)
```bash
# Clone the repository
git clone <repository-url>
cd pmu-instance-sampah-fuzzy

# Install dependencies
pip install -r requirements.txt
```

#### Option B: CUDA Setup (Local GPU Training)
```bash
# Clone the repository
git clone <repository-url>
cd pmu-instance-sampah-fuzzy

# Install PyTorch with CUDA support
python install_cuda.py

# Or manually install CUDA-enabled PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify CUDA Installation
```bash
# Test CUDA setup
python test_cuda.py

# Check CUDA status in your system
python -c "from main_colab import WasteDetectionSystemColab; system = WasteDetectionSystemColab(); print(system.get_cuda_status())"
```

### 3. Configuration

1. **Set up secrets**:
   ```bash
   # Copy and edit secrets template
   cp secrets.yaml.example secrets.yaml
   # Edit secrets.yaml with your API keys
   ```

2. **Configure settings**:
   ```bash
   # Edit config.yaml for your environment
   # Adjust model parameters, dataset settings, etc.
   ```

### 4. Run Tests

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --category security
python run_tests.py --category config
python run_tests.py --category fuzzy

# Run specific tests
python run_tests.py --test config_manager
```

### 5. Usage in Google Colab

```python
# Import the main system
from main_colab import WasteDetectionSystemColab

# Initialize the system
system = WasteDetectionSystemColab()

# Train and export model
result = system.train_and_export_model("v8n", epochs=50, batch_size=16)

# Run inference and visualization
inference_results = system.run_inference_and_visualization(result, "v8n")

# Analyze training metrics
metrics = system.analyze_training_run(result, "v8n")
```

### 6. Command-Line Usage

The system now supports flexible command-line execution with various options:

#### Basic Usage
```bash
# Run with default settings (YOLOv8n)
python main_colab.py

# Run specific model
python main_colab.py --models v8n

# Run multiple models
python main_colab.py --models v8n v10n v11n
```

#### Training Options
```bash
# Force retraining (ignore existing models)
python main_colab.py --force-retrain

# Custom training parameters
python main_colab.py --epochs 100 --batch-size 32

# Combine options
python main_colab.py --models v8n v10n --force-retrain --epochs 200 --batch-size 16
```

#### Dataset Management
```bash
# Prepare datasets before training
python main_colab.py --prepare-datasets

# Fix existing datasets if preparation fails
python main_colab.py --fix-datasets

# Both dataset operations
python main_colab.py --prepare-datasets --fix-datasets
```

#### Google Drive Integration
```bash
# Save results to Google Drive
python main_colab.py --save-to-drive

# Complete pipeline with drive save
python main_colab.py --models v8n --force-retrain --save-to-drive
```

#### Setup and Help
```bash
# Show Google Colab setup instructions
python main_colab.py --show-setup

# Show help
python main_colab.py --help
```

#### Complete Examples

**Quick Start (Use existing models):**
```bash
python main_colab.py --models v8n
```

**Full Training Pipeline:**
```bash
python main_colab.py --models v8n v10n v11n --force-retrain --epochs 200 --batch-size 16 --prepare-datasets --save-to-drive
```

**Development/Testing:**
```bash
python main_colab.py --models v8n --epochs 50 --batch-size 8
```

**Production Deployment:**
```bash
python main_colab.py --models v8n v10n --force-retrain --epochs 300 --batch-size 32 --save-to-drive
```

#### Command-Line Arguments Reference

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--models` | Model versions to run | `["v8n"]` | `--models v8n v10n v11n` |
| `--force-retrain` | Force retraining | `False` | `--force-retrain` |
| `--epochs` | Training epochs | `200` | `--epochs 100` |
| `--batch-size` | Batch size | `16` | `--batch-size 32` |
| `--prepare-datasets` | Prepare datasets | `False` | `--prepare-datasets` |
| `--fix-datasets` | Fix existing datasets | `False` | `--fix-datasets` |
| `--save-to-drive` | Save to Google Drive | `False` | `--save-to-drive` |
| `--show-setup` | Show Colab setup | `False` | `--show-setup` |
| `--help` | Show help | - | `--help` |

## 📚 Documentation

### Core Documentation
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Production deployment instructions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[ONNX/RKNN Troubleshooting](docs/ONNX_RKNN_TROUBLESHOOTING.md)**: ONNX and RKNN model issues and solutions
- **[Pipeline Differences](docs/PIPELINE_DIFFERENCES_GUIDE.md)**: Understanding pipeline types and folder issues
- **[Pipeline Quick Reference](docs/PIPELINE_QUICK_REFERENCE.md)**: Quick comparison table and commands
- **[Contributing](docs/CONTRIBUTING.md)**: Development guidelines

### System Improvements
- **[Error Handling](ERROR_HANDLING_IMPROVEMENTS.md)**: Comprehensive error management
- **[Configuration Management](CONFIGURATION_MANAGEMENT_IMPROVEMENTS.md)**: Centralized configuration system
- **[Secrets Management](SECRETS_MANAGEMENT.md)**: Security and secrets handling
- **[Unit Testing](UNIT_TESTING_IMPROVEMENTS.md)**: Testing framework and coverage

## 🔧 Configuration

### Main Configuration (`config.yaml`)

```yaml
model:
  supported_versions: ["v8n", "v10n", "v11n"]
  default_epochs: 50
  default_batch_size: 16
  default_img_size: [640, 640]
  default_conf_threshold: 0.25

dataset:
  roboflow_project: "your_project_name"
  roboflow_version: "1"
  default_dataset_dir: "datasets"
  default_model_dir: "runs"

fuzzy:
  area_percent_ranges:
    sedikit: [0, 0, 5]
    sedang: [3, 10, 20]
    banyak: [15, 100, 100]
  sedikit_threshold: 33.0
  sedang_threshold: 66.0
```

### Secrets Configuration (`secrets.yaml`)

```yaml
# Roboflow API Configuration
roboflow_api_key: "YOUR_ROBOFLOW_API_KEY_HERE"

# Optional: Additional services
# google_cloud_api_key: "YOUR_GOOGLE_CLOUD_API_KEY"
# aws_access_key_id: "YOUR_AWS_ACCESS_KEY_ID"
```

## 🏗️ System Architecture (Updated)

### Main Orchestrator (`main_colab.py`)

The main system has been reorganized into focused, reusable functions:

#### **System Functions**
- `initialize_system()`: Clean system initialization with status display
- `setup_colab_environment()`: Colab setup instructions
- `display_completion_message()`: User-friendly completion messages

#### **Dataset Functions**
- `prepare_datasets()`: Dataset preparation with Ultralytics normalization
- `fix_existing_dataset()`: Dataset fixing functionality

#### **Pipeline Functions**
- `run_single_model_pipeline()`: Single model execution with error handling
- `run_all_model_pipelines()`: Multi-model execution with results tracking
- `print_pipeline_summary()`: Execution summary display

#### **Utility Functions**
- `parse_arguments()`: Command-line argument parsing
- `save_results_to_drive()`: Google Drive integration

#### **Benefits of New Organization**
- **🎯 Readability**: Functions are focused and well-documented
- **🔧 Flexibility**: Command-line arguments for different use cases
- **🔄 Maintainability**: Easy to modify individual components
- **📋 Documentation**: Clear docstrings and help messages
- **📊 Progress Tracking**: Better execution monitoring and summaries

#### **Function Usage Examples**

```python
# Initialize system
from main_colab import initialize_system, run_all_model_pipelines
system = initialize_system()

# Run multiple models with custom parameters
results = run_all_model_pipelines(
    system, 
    models=["v8n", "v10n"], 
    force_retrain=False,
    epochs=200,
    batch_size=16
)

# Check results
for model, success in results.items():
    print(f"YOLO{model}: {'✅' if success else '❌'}")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run by category
python run_tests.py --category security
python run_tests.py --category config
python run_tests.py --category fuzzy
python run_tests.py --category exceptions
python run_tests.py --category integration

# Run specific test
python run_tests.py --test config_manager

# Run dataset fix integration test
python tests/dataset_tools/test_dataset_fix_integration.py
# or
python tests/dataset_tools/run_dataset_fix_test.py

# Run dataset tools individually
python tests/dataset_tools/extract_and_check_dataset.py
python tests/dataset_tools/fix_yolo_coordinates.py
python tests/dataset_tools/final_verification.py

# Verbose output
python run_tests.py --verbose

# Quiet output
python run_tests.py --quiet
```

### Test Categories

- **Security Tests**: Secrets validation, file permissions, git tracking
- **Configuration Tests**: Config management, validation, updates
- **Fuzzy Logic Tests**: Classification logic, input validation, performance
- **Exception Tests**: Custom exception hierarchy, error handling
- **Integration Tests**: System-wide functionality, workflows
- **Dataset Fix Integration Test**: Ultralytics-based dataset normalization and validation
- **Dataset Tools**: Extract, verify, fix coordinates, and validate datasets for YOLO training

## 🔒 Security

### Secrets Management

1. **Never commit secrets**: `secrets.yaml` is in `.gitignore`
2. **Validate setup**: Run `python validate_secrets.py`
3. **Environment variables**: Alternative to file-based secrets
4. **Key rotation**: Regular API key updates

### Security Checklist

- [ ] `secrets.yaml` is created with actual API keys
- [ ] `secrets.yaml` is in `.gitignore`
- [ ] File permissions are set to 600
- [ ] No API keys are hardcoded in source code
- [ ] Different keys for different environments

## 🚀 CUDA/GPU Support

### Automatic CUDA Detection
The system automatically detects and uses CUDA if available:
- **GPU Training**: Automatically uses GPU for faster training
- **Memory Management**: Automatic memory optimization and cache clearing
- **Fallback**: Gracefully falls back to CPU if CUDA unavailable

### CUDA Features
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Batch Size Optimization**: Automatic optimal batch size calculation
- **Training Optimization**: Recommendations for best performance
- **Cache Management**: Automatic memory cleanup

### Usage Examples

```python
from main_colab import WasteDetectionSystemColab

# Initialize system
system = WasteDetectionSystemColab()

# Check CUDA status
cuda_status = system.get_cuda_status()
print(f"GPU Memory: {cuda_status['memory_info']}")
print(f"Optimal Batch Size: {cuda_status['optimal_batch_size']}")

# Clear GPU cache
system.clear_cuda_cache()

# Train with GPU optimization
result = system.train_and_export_model("v8n", epochs=50, batch_size=16)
```

### Performance Tips
- **Batch Size**: Use `system.get_cuda_status()['optimal_batch_size']` for best performance
- **Memory**: Call `system.clear_cuda_cache()` between training runs
- **Monitoring**: Check `system.get_cuda_status()['memory_info']` for memory usage

## 🚀 Deployment

### Google Colab Deployment

1. **Upload files to Colab**
2. **Install dependencies**
3. **Configure secrets**
4. **Run training pipeline**

### Production Deployment

1. **Environment setup**
2. **Model optimization**
3. **Performance monitoring**
4. **Error handling**

See [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) for detailed instructions.

## ⚡ Quick Reference

### Command-Line Quick Start

```bash
# Basic usage
python main_colab.py

# Run specific model
python main_colab.py --models v8n

# Force retraining
python main_colab.py --force-retrain

# Custom parameters
python main_colab.py --epochs 100 --batch-size 32

# Complete pipeline
python main_colab.py --models v8n v10n --force-retrain --epochs 200 --save-to-drive

# Show help
python main_colab.py --help
```

### Function Quick Reference

```python
# Initialize system
from main_colab import initialize_system, run_all_model_pipelines
system = initialize_system()

# Run pipelines
results = run_all_model_pipelines(system, models=["v8n", "v10n"])

# Check results
for model, success in results.items():
    print(f"YOLO{model}: {'✅' if success else '❌'}")
```

### Test Quick Reference

```bash
# Run all tests
python run_tests.py

# Run specific category
python run_tests.py --category security

# Run specific test
python run_tests.py --test config_manager
```

### Environment Diagnostic

```bash
# Check ONNX and RKNN environment
python tests/onnx_testing/check_onnx_rknn_environment.py

# This will check:
# - Python version compatibility
# - PyTorch and CUDA installation
# - ONNX and RKNN dependencies
# - GPU vs CPU environment
# - Available memory
# - Existing model files
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
3. **Make changes**
4. **Run tests**
5. **Submit pull request**

### Code Standards

- **Error handling**: Use custom exceptions
- **Configuration**: Use centralized config management
- **Testing**: Maintain high test coverage
- **Documentation**: Update docs with changes

See [Contributing Guide](docs/CONTRIBUTING.md) for detailed guidelines.

## 📊 Performance

### Model Performance

- **Training time**: ~2-4 hours (50 epochs)
- **Inference speed**: ~30-50ms per image
- **Memory usage**: ~4-8GB GPU memory
- **Accuracy**: 85%+ mAP on test set

### System Performance

- **Test execution**: ~12 seconds for full suite
- **Configuration loading**: <100ms
- **Error recovery**: Graceful degradation
- **Memory efficiency**: Optimized for Colab

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Check `secrets.yaml` configuration
2. **Model Training Failures**: Verify dataset availability
3. **Memory Issues**: Reduce batch size or image size
4. **Test Failures**: Check dependencies and configuration

### Getting Help

1. **Check documentation**: Review relevant guides
2. **Run tests**: Verify system integrity
3. **Check logs**: Review error messages
4. **Validate setup**: Run validation scripts

See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for detailed solutions.

## 📈 Roadmap

### Planned Features

- **Real-time Processing**: Live video stream analysis
- **Mobile Deployment**: Edge device optimization
- **Advanced Analytics**: Detailed performance metrics
- **Multi-language Support**: Internationalization
- **Cloud Integration**: AWS/GCP deployment options

### Future Enhancements

- **Model Compression**: Quantization and pruning
- **Active Learning**: Continuous model improvement
- **API Service**: RESTful API endpoints
- **Dashboard**: Web-based monitoring interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO Models**: Ultralytics for state-of-the-art detection
- **Roboflow**: Dataset management and annotation
- **Fuzzy Logic**: Scikit-fuzzy for intelligent classification
- **Google Colab**: Cloud-based development environment

## 📞 Support

### Getting Help

- **Documentation**: Check the docs folder
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact maintainers for urgent issues

### Community

- **Contributions**: Welcome and appreciated
- **Feedback**: Share your experience
- **Suggestions**: Propose new features
- **Testing**: Help improve test coverage

---

**Made with ❤️ for environmental monitoring and waste management** 