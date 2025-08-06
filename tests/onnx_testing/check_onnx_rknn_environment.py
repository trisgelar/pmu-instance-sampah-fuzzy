#!/usr/bin/env python3
"""
ONNX and RKNN Environment Diagnostic Script

This script checks your environment for ONNX and RKNN dependencies
and identifies potential issues with GPU vs CPU setups.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section."""
    print(f"\n📋 {title}")
    print("-" * 40)

def check_python_version():
    """Check Python version."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
    else:
        print("❌ Python version should be 3.8+")

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_section("PyTorch Installation")
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("✅ GPU environment detected")
        else:
            print("ℹ️ CPU-only environment detected")
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("Install with: pip install torch torchvision torchaudio")

def check_ultralytics():
    """Check Ultralytics installation."""
    print_section("Ultralytics Installation")
    
    try:
        import ultralytics
        print(f"Ultralytics Version: {ultralytics.__version__}")
        print("✅ Ultralytics is installed")
    except ImportError:
        print("❌ Ultralytics not installed")
        print("Install with: pip install ultralytics")

def check_onnx():
    """Check ONNX installation."""
    print_section("ONNX Installation")
    
    try:
        import onnx
        print(f"ONNX Version: {onnx.__version__}")
        print("✅ ONNX is installed")
    except ImportError:
        print("❌ ONNX not installed")
        print("Install with: pip install onnx")
    
    try:
        import onnxruntime
        print(f"ONNX Runtime Version: {onnxruntime.__version__}")
        print("✅ ONNX Runtime is installed")
    except ImportError:
        print("❌ ONNX Runtime not installed")
        print("Install with: pip install onnxruntime")

def check_rknn():
    """Check RKNN-Toolkit2 installation."""
    print_section("RKNN-Toolkit2 Installation")
    
    try:
        import rknn_toolkit2
        print("✅ RKNN-Toolkit2 is installed")
    except ImportError:
        print("❌ RKNN-Toolkit2 not installed")
        print("Install with: pip install rknn-toolkit2")
        print("Or follow the official installation guide:")
        print("git clone -b v2.3.0 https://github.com/airockchip/rknn-toolkit2.git")

def check_cuda():
    """Check CUDA installation."""
    print_section("CUDA Installation")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract driver version
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"Driver Version: {line.split('Driver Version:')[1].split()[0]}")
                    break
        else:
            print("❌ NVIDIA GPU not detected or nvidia-smi not available")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA compiler (nvcc) available")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    print(f"CUDA Version: {cuda_version}")
                    break
        else:
            print("❌ CUDA compiler (nvcc) not available")
    except FileNotFoundError:
        print("❌ nvcc not found")

def check_model_files():
    """Check for existing model files."""
    print_section("Model Files Check")
    
    # Check PyTorch models
    pytorch_dir = Path("results/runs")
    if pytorch_dir.exists():
        pytorch_models = list(pytorch_dir.glob("*/weights/best.pt"))
        if pytorch_models:
            print(f"✅ Found {len(pytorch_models)} PyTorch models:")
            for model in pytorch_models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   - {model}: {size_mb:.1f} MB")
        else:
            print("❌ No PyTorch models found")
    else:
        print("❌ results/runs directory not found")
    
    # Check ONNX models
    onnx_dir = Path("results/onnx_models")
    if onnx_dir.exists():
        onnx_models = list(onnx_dir.glob("*.onnx"))
        if onnx_models:
            print(f"✅ Found {len(onnx_models)} ONNX models:")
            for model in onnx_models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   - {model}: {size_mb:.1f} MB")
        else:
            print("❌ No ONNX models found")
    else:
        print("❌ results/onnx_models directory not found")
    
    # Check RKNN models
    rknn_dir = Path("results/rknn_models")
    if rknn_dir.exists():
        rknn_models = list(rknn_dir.glob("*.rknn"))
        if rknn_models:
            print(f"✅ Found {len(rknn_models)} RKNN models:")
            for model in rknn_models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   - {model}: {size_mb:.1f} MB")
        else:
            print("❌ No RKNN models found")
    else:
        print("❌ results/rknn_models directory not found")

def check_memory():
    """Check available memory."""
    print_section("Memory Check")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"RAM Usage: {memory.percent}%")
        
        if memory.available / (1024**3) > 4:
            print("✅ Sufficient RAM available")
        else:
            print("⚠️ Low RAM available - may cause issues")
            
    except ImportError:
        print("ℹ️ Install psutil for memory info: pip install psutil")
    
    # Check GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"GPU Allocated: {allocated:.1f} GB")
            print(f"GPU Cached: {cached:.1f} GB")
            
            if gpu_memory > 4:
                print("✅ Sufficient GPU memory")
            else:
                print("⚠️ Limited GPU memory - may cause issues")
                
    except Exception as e:
        print(f"ℹ️ GPU memory check failed: {e}")

def check_dependencies():
    """Check all required dependencies."""
    print_section("Dependencies Check")
    
    required_packages = [
        'torch', 'torchvision', 'torchaudio',
        'ultralytics', 'onnx', 'onnxruntime',
        'opencv-python', 'matplotlib', 'seaborn',
        'pandas', 'numpy', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ {package} (cv2)")
            elif package == 'matplotlib':
                import matplotlib
                print(f"✅ {package} ({matplotlib.__version__})")
            elif package == 'seaborn':
                import seaborn
                print(f"✅ {package}")
            elif package == 'pandas':
                import pandas
                print(f"✅ {package} ({pandas.__version__})")
            elif package == 'numpy':
                import numpy
                print(f"✅ {package} ({numpy.__version__})")
            elif package == 'scipy':
                import scipy
                print(f"✅ {package} ({scipy.__version__})")
            else:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} ({version})")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required dependencies are installed")

def main():
    """Run all diagnostic checks."""
    print_header("ONNX and RKNN Environment Diagnostic")
    
    check_python_version()
    check_pytorch()
    check_ultralytics()
    check_onnx()
    check_rknn()
    check_cuda()
    check_memory()
    check_dependencies()
    check_model_files()
    
    print_header("Diagnostic Complete")
    print("💡 If you see issues above, refer to docs/ONNX_RKNN_TROUBLESHOOTING.md")
    print("🚀 For ONNX export, use: python main_colab.py --models v8n --onnx-export")

if __name__ == "__main__":
    main() 