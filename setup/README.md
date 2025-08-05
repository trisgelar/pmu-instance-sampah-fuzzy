# Setup Utilities

This directory contains setup utilities and installation scripts for the Waste Detection System.

## 📁 Directory Structure

```
setup/
├── cuda/                     # CUDA setup and installation
│   ├── install_cuda.py       # Automated CUDA installation script
│   ├── setup_cuda_environment.py  # CUDA environment configuration
│   ├── setup_cuda_env.bat    # Windows batch file for CUDA setup
│   ├── requirements-dev-cuda.txt  # CUDA development dependencies
│   └── requirements-dev-cuda-rknn.txt  # CUDA + RKNN dependencies
└── README.md                 # This file
```

## 🚀 Setup Categories

### CUDA Setup (`cuda/`)
- **Purpose**: CUDA installation and environment configuration
- **Scope**: GPU acceleration setup for deep learning
- **Files**:
  - `install_cuda.py`: Automated PyTorch CUDA installation
  - `setup_cuda_environment.py`: Environment configuration for existing CUDA
  - `setup_cuda_env.bat`: Windows batch file for quick setup
  - `requirements-dev-cuda.txt`: CUDA development dependencies
  - `requirements-dev-cuda-rknn.txt`: CUDA + RKNN dependencies

## 🔧 Usage

### CUDA Installation
```bash
# Automated CUDA installation
python setup/cuda/install_cuda.py

# Setup existing CUDA environment
python setup/cuda/setup_cuda_environment.py

# Windows batch setup
setup/cuda/setup_cuda_env.bat
```

### Install Dependencies
```bash
# Install CUDA development dependencies
pip install -r setup/cuda/requirements-dev-cuda.txt

# Install CUDA + RKNN dependencies
pip install -r setup/cuda/requirements-dev-cuda-rknn.txt
```

## 📋 Setup Guidelines

### Adding New Setup Scripts

1. **Choose the right category**:
   - `cuda/` for GPU-related setup
   - Create new directories for other setup types (e.g., `database/`, `web/`)

2. **Follow naming conventions**:
   - Installation scripts: `install_*.py`
   - Setup scripts: `setup_*.py`
   - Batch files: `setup_*.bat`
   - Requirements: `requirements-*.txt`

3. **Include proper documentation**:
   - Clear docstrings explaining script purpose
   - Usage examples in docstrings
   - Update this README when adding new categories

### Script Structure
```python
#!/usr/bin/env python3
"""
Brief description of what this setup script does.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main setup function."""
    try:
        # Setup implementation
        pass
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 🔧 Maintenance

### Adding New Setup Categories
1. Create new directory under `setup/`
2. Add `__init__.py` file
3. Update this README
4. Test the setup scripts

### Updating Dependencies
- Keep requirements files up-to-date
- Test installation on different platforms
- Document version compatibility

## 🐛 Troubleshooting

### Common Issues
1. **Permission errors**: Run as administrator on Windows
2. **Path issues**: Check environment variables
3. **Version conflicts**: Use virtual environments
4. **Missing dependencies**: Install base requirements first

### Getting Help
- Check script output for specific error messages
- Review setup logs for detailed information
- Test setup on clean environment
- Consult platform-specific documentation 