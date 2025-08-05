#!/usr/bin/env python3
"""
CUDA Version Checker
Checks and diagnoses CUDA version issues on Windows and Colab.
"""

import subprocess
import sys
import platform
import os

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def check_nvidia_smi():
    """Check NVIDIA driver and CUDA version."""
    output = run_command(['nvidia-smi'])
    if output:
        print("📋 NVIDIA Driver Information:")
        print("-" * 40)
        for line in output.split('\n'):
            if 'Driver Version:' in line or 'CUDA Version:' in line:
                print(f"  {line.strip()}")
        return output
    return None

def check_nvcc():
    """Check CUDA toolkit version."""
    output = run_command(['nvcc', '--version'])
    if output:
        print("\n📋 CUDA Toolkit Information:")
        print("-" * 40)
        for line in output.split('\n'):
            if 'release' in line:
                version = line.split('release ')[1].split(',')[0]
                print(f"  CUDA Toolkit Version: {version}")
                return version
    return None

def check_pytorch_cuda():
    """Check PyTorch CUDA version."""
    try:
        import torch
        print("\n📋 PyTorch CUDA Information:")
        print("-" * 40)
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  PyTorch CUDA Version: {torch.version.cuda}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return torch.version.cuda
    except ImportError:
        print("\n❌ PyTorch not installed")
        return None
    except Exception as e:
        print(f"\n❌ Error checking PyTorch: {e}")
        return None

def diagnose_issues(nvidia_smi_output, nvcc_version, pytorch_cuda):
    """Diagnose CUDA version issues."""
    print("\n🔍 CUDA Version Diagnosis:")
    print("-" * 40)
    
    issues = []
    
    # Check if nvidia-smi shows CUDA version
    nvidia_cuda = None
    if nvidia_smi_output:
        for line in nvidia_smi_output.split('\n'):
            if 'CUDA Version:' in line:
                nvidia_cuda = line.split('CUDA Version:')[1].strip()
                break
    
    if nvidia_cuda:
        print(f"  NVIDIA Driver CUDA: {nvidia_cuda}")
    else:
        print("  NVIDIA Driver CUDA: Not detected")
        issues.append("NVIDIA driver CUDA version not detected")
    
    if nvcc_version:
        print(f"  CUDA Toolkit: {nvcc_version}")
    else:
        print("  CUDA Toolkit: Not installed")
        issues.append("CUDA toolkit not installed")
    
    if pytorch_cuda:
        print(f"  PyTorch CUDA: {pytorch_cuda}")
    else:
        print("  PyTorch CUDA: Not available")
        issues.append("PyTorch CUDA not available")
    
    # Check for version mismatches
    if nvcc_version and pytorch_cuda:
        if nvcc_version != pytorch_cuda:
            issues.append(f"CUDA toolkit ({nvcc_version}) and PyTorch CUDA ({pytorch_cuda}) versions don't match")
    
    if nvidia_cuda and pytorch_cuda:
        # Check if PyTorch CUDA version is supported by driver
        driver_major = int(nvidia_cuda.split('.')[0])
        driver_minor = int(nvidia_cuda.split('.')[1])
        pytorch_major = int(pytorch_cuda.split('.')[0])
        pytorch_minor = int(pytorch_cuda.split('.')[1])
        
        if pytorch_major > driver_major or (pytorch_major == driver_major and pytorch_minor > driver_minor):
            issues.append(f"PyTorch CUDA {pytorch_cuda} is newer than driver supports ({nvidia_cuda})")
    
    return issues

def suggest_solutions(issues):
    """Suggest solutions based on detected issues."""
    if not issues:
        print("\n✅ No CUDA issues detected!")
        return
    
    print("\n🔧 Suggested Solutions:")
    print("-" * 40)
    
    for issue in issues:
        if "CUDA toolkit not installed" in issue:
            print("  1. Install CUDA 12.4 Toolkit:")
            print("     - Download from: https://developer.nvidia.com/cuda-12-4-0-download-archive")
            print("     - Or use: conda install cudatoolkit=12.4")
        
        elif "PyTorch CUDA not available" in issue:
            print("  2. Install PyTorch with CUDA support:")
            print("     - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        elif "versions don't match" in issue:
            print("  3. Reinstall PyTorch to match CUDA toolkit:")
            print("     - pip uninstall torch torchvision torchaudio")
            print("     - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        elif "newer than driver supports" in issue:
            print("  4. Install PyTorch with compatible CUDA version:")
            print("     - Use CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("     - Or update NVIDIA driver")
        
        elif "NVIDIA driver CUDA version not detected" in issue:
            print("  5. Check NVIDIA driver installation:")
            print("     - Update NVIDIA driver from: https://www.nvidia.com/Download/index.aspx")
            print("     - Verify GPU is detected: nvidia-smi")

def main():
    """Main function."""
    print("🔍 CUDA Version Checker")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check all CUDA versions
    nvidia_smi_output = check_nvidia_smi()
    nvcc_version = check_nvcc()
    pytorch_cuda = check_pytorch_cuda()
    
    # Diagnose issues
    issues = diagnose_issues(nvidia_smi_output, nvcc_version, pytorch_cuda)
    
    # Suggest solutions
    suggest_solutions(issues)
    
    print("\n📋 Quick Fix Commands:")
    print("-" * 40)
    print("For CUDA 12.4:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("\nFor CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nFor CPU only:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n🎯 Next Steps:")
    print("-" * 40)
    print("1. Run the suggested installation command")
    print("2. Verify with: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("3. Test CUDA: python test_cuda.py")

if __name__ == "__main__":
    main() 