#!/usr/bin/env python3
"""
CUDA Installation Helper Script
Helps users install PyTorch with the correct CUDA version for their system.
Supports CUDA 12.4 and other versions.
"""

import subprocess
import sys
import platform
import os

def check_cuda_version():
    """Check if CUDA is installed and get version."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        # Extract version from output like "Cuda compilation tools, release 11.8"
        output = result.stdout
        if 'release' in output:
            version = output.split('release ')[1].split(',')[0]
            return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None

def check_nvidia_smi():
    """Check NVIDIA driver version using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        # Look for CUDA Version in nvidia-smi output
        for line in output.split('\n'):
            if 'CUDA Version:' in line:
                version = line.split('CUDA Version:')[1].strip()
                return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None

def get_pytorch_cuda_url(cuda_version):
    """Get the appropriate PyTorch installation URL based on CUDA version."""
    cuda_mapping = {
        '12.4': 'cu124',
        '12.1': 'cu121',
        '11.8': 'cu118',
        '11.7': 'cu117', 
        '11.6': 'cu116',
        '11.5': 'cu115',
        '11.4': 'cu114',
        '11.3': 'cu113',
        '11.2': 'cu112',
        '11.1': 'cu111',
        '11.0': 'cu110',
        '10.2': 'cu102',
        '10.1': 'cu101',
        '10.0': 'cu100',
    }
    
    return cuda_mapping.get(cuda_version, 'cpu')

def install_pytorch_cuda():
    """Install PyTorch with appropriate CUDA support."""
    print("🔍 Checking CUDA installation...")
    
    # Check both nvcc and nvidia-smi
    nvcc_version = check_cuda_version()
    nvidia_smi_version = check_nvidia_smi()
    
    print(f"📋 CUDA Toolkit (nvcc): {nvcc_version or 'Not found'}")
    print(f"📋 NVIDIA Driver (nvidia-smi): {nvidia_smi_version or 'Not found'}")
    
    # Use nvcc version if available, otherwise use nvidia-smi version
    cuda_version = nvcc_version or nvidia_smi_version
    
    if cuda_version:
        print(f"✅ CUDA {cuda_version} detected")
        pytorch_cuda = get_pytorch_cuda_url(cuda_version)
        
        if pytorch_cuda == 'cpu':
            print(f"⚠️  CUDA {cuda_version} not supported, installing CPU version")
            print("   Supported CUDA versions: 12.4, 12.1, 11.8, 11.7, 11.6, 11.5, 11.4, 11.3, 11.2, 11.1, 11.0, 10.2, 10.1, 10.0")
            pytorch_cuda = 'cpu'
        else:
            print(f"🚀 Installing PyTorch with CUDA {pytorch_cuda} support")
    else:
        print("❌ CUDA not detected, installing CPU version")
        pytorch_cuda = 'cpu'
    
    # Install PyTorch
    if pytorch_cuda == 'cpu':
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
    else:
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', f'https://download.pytorch.org/whl/{pytorch_cuda}'
        ]
    
    print(f"📦 Installing PyTorch with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed successfully")
        
        # Verify installation
        verify_cuda_installation()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False
    
    return True

def verify_cuda_installation():
    """Verify CUDA installation and availability."""
    print("\n🔍 Verifying CUDA installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")

def install_requirements():
    """Install other requirements."""
    print("\n📦 Installing other requirements...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    """Main installation function."""
    print("🚀 Waste Detection System - CUDA Setup")
    print("=" * 50)
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        print("❌ CUDA setup failed")
        return False
    
    # Install other requirements
    if not install_requirements():
        print("❌ Requirements installation failed")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python validate_secrets.py")
    print("2. Run: python run_tests.py")
    print("3. Start training: python main_colab.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 