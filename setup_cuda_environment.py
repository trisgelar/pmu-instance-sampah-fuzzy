#!/usr/bin/env python3
"""
CUDA Environment Setup Script
Configures environment to use existing CUDA 12.4 installation at Local Disk
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_cuda_installation():
    """Check if CUDA 12.4 is installed at the expected location."""
    cuda_path = Path("D:/Programs/Nvidia/CUDA/v12.4")
    
    print("🔍 Checking CUDA 12.4 Installation...")
    print("-" * 50)
    
    if cuda_path.exists():
        print(f"✅ CUDA 12.4 found at: {cuda_path}")
        
        # Check key directories
        bin_path = cuda_path / "bin"
        lib_path = cuda_path / "lib" / "x64"
        include_path = cuda_path / "include"
        
        print(f"  📁 Bin directory: {'✅' if bin_path.exists() else '❌'} {bin_path}")
        print(f"  📁 Lib directory: {'✅' if lib_path.exists() else '❌'} {lib_path}")
        print(f"  📁 Include directory: {'✅' if include_path.exists() else '❌'} {include_path}")
        
        # Check for nvcc
        nvcc_path = bin_path / "nvcc.exe"
        if nvcc_path.exists():
            print(f"  🔧 nvcc compiler: ✅ {nvcc_path}")
        else:
            print(f"  🔧 nvcc compiler: ❌ Not found")
        
        return True
    else:
        print(f"❌ CUDA 12.4 not found at: {cuda_path}")
        return False

def set_environment_variables():
    """Set CUDA environment variables."""
    cuda_home = "D:/Programs/Nvidia/CUDA/v12.4"
    
    print("\n🔧 Setting Environment Variables...")
    print("-" * 50)
    
    # Set CUDA_HOME
    os.environ['CUDA_HOME'] = cuda_home
    print(f"  CUDA_HOME = {cuda_home}")
    
    # Add CUDA bin to PATH
    cuda_bin = f"{cuda_home}/bin"
    current_path = os.environ.get('PATH', '')
    
    if cuda_bin not in current_path:
        os.environ['PATH'] = f"{cuda_bin};{current_path}"
        print(f"  Added to PATH: {cuda_bin}")
    else:
        print(f"  PATH already contains: {cuda_bin}")
    
    # Set CUDA_PATH
    os.environ['CUDA_PATH'] = cuda_home
    print(f"  CUDA_PATH = {cuda_home}")
    
    return True

def verify_cuda_setup():
    """Verify CUDA setup is working."""
    print("\n🔍 Verifying CUDA Setup...")
    print("-" * 50)
    
    # Test nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ nvcc is working:")
        for line in result.stdout.split('\n'):
            if 'release' in line:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"❌ nvcc test failed: {e}")
        return False
    
    # Test nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, check=True)
        print("✅ nvidia-smi is working")
    except Exception as e:
        print(f"❌ nvidia-smi test failed: {e}")
        return False
    
    return True

def install_pytorch_cuda124():
    """Install PyTorch with CUDA 12.4 support."""
    print("\n📦 Installing PyTorch with CUDA 12.4...")
    print("-" * 50)
    
    try:
        # Uninstall existing PyTorch if any
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 
                       'torch', 'torchvision', 'torchaudio'], 
                      capture_output=True)
        
        # Install PyTorch with CUDA 12.4
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu124'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("✅ PyTorch with CUDA 12.4 installed successfully")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def verify_pytorch_cuda():
    """Verify PyTorch CUDA installation."""
    print("\n🔍 Verifying PyTorch CUDA...")
    print("-" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ PyTorch CUDA version: {torch.version.cuda}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name()}")
            
            # Test CUDA operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA operations working correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        return False

def create_batch_file():
    """Create a batch file to set environment variables."""
    batch_content = """@echo off
REM CUDA Environment Setup Script
REM Run this before using CUDA applications

set CUDA_HOME=D:\\Programs\\Nvidia\\CUDA\\v12.4
set CUDA_PATH=D:\\Programs\\Nvidia\\CUDA\\v12.4
set PATH=%CUDA_HOME%\\bin;%PATH%

echo CUDA environment variables set:
echo CUDA_HOME=%CUDA_HOME%
echo CUDA_PATH=%CUDA_PATH%
echo.

REM Test CUDA
nvcc --version
echo.
nvidia-smi
echo.

REM Activate Python environment if needed
REM conda activate your_env_name

echo CUDA environment ready!
"""
    
    with open('setup_cuda_env.bat', 'w') as f:
        f.write(batch_content)
    
    print("\n📄 Created setup_cuda_env.bat")
    print("  Run this batch file to set CUDA environment variables")

def main():
    """Main setup function."""
    print("🚀 CUDA 12.4 Environment Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check CUDA installation
    if not check_cuda_installation():
        print("\n❌ CUDA 12.4 not found at expected location")
        print("Please verify your CUDA installation path")
        return False
    
    # Set environment variables
    set_environment_variables()
    
    # Verify CUDA setup
    if not verify_cuda_setup():
        print("\n❌ CUDA setup verification failed")
        return False
    
    # Install PyTorch with CUDA 12.4
    if not install_pytorch_cuda124():
        print("\n❌ PyTorch installation failed")
        return False
    
    # Verify PyTorch CUDA
    if not verify_pytorch_cuda():
        print("\n❌ PyTorch CUDA verification failed")
        return False
    
    # Create batch file
    create_batch_file()
    
    print("\n🎉 CUDA 12.4 Setup Completed Successfully!")
    print("\n📋 Next Steps:")
    print("1. Run: setup_cuda_env.bat (to set environment variables)")
    print("2. Run: python test_cuda.py (to test CUDA functionality)")
    print("3. Run: python main_colab.py (to start your project)")
    
    print("\n💡 Tips:")
    print("- Always run setup_cuda_env.bat before using CUDA applications")
    print("- Your CUDA installation at D:\\Programs\\Nvidia\\CUDA\\v12.4 is now configured")
    print("- PyTorch will use your existing CUDA 12.4 installation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 