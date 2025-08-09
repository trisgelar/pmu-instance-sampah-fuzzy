#!/usr/bin/env python3
"""
Google Colab Setup Script for PMU Instance Sampah Fuzzy

This script automates the setup process for running the waste detection system
on Google Colab. It handles repository cloning, dependency installation,
configuration setup, and system initialization.

Updated to align with requirements-dev-linux-rknn-local.txt and use
specific RKNN Toolkit2 wheel from GitHub.

Usage:
    Run this script in a Google Colab cell to set up the environment.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install all required dependencies aligned with requirements-dev-linux-rknn-local.txt."""
    
    # Core dependencies from requirements file
    core_dependencies = [
        "ultralytics==8.3.174",
        "roboflow==1.2.3", 
        "opencv-python==4.11.0.86",
        "matplotlib==3.10.5",
        "seaborn==0.13.2",
        "pandas==2.3.1",
        "numpy==1.26.4",
        "PyYAML==6.0.2",
        "scikit-fuzzy==0.5.0",
        "tensorboard==2.20.0",
        "onnx==1.18.0",
        "onnxruntime==1.22.1",
        "torch==2.4.0+cu124",
        "torchvision==0.19.0+cu124",
        "torchaudio==2.4.0+cu124",
        "pillow==11.3.0",
        "requests==2.32.4",
        "python-dotenv==1.1.1",
        "tqdm==4.67.1",
        "protobuf==4.25.4",
        "flatbuffers==25.2.10",
        "filelock==3.13.1",
        "psutil==7.0.0",
        "scipy==1.16.1",
        "networkx==3.3",
        "kiwisolver==1.4.8",
        "cycler==0.12.1",
        "contourpy==1.3.3",
        "fonttools==4.59.0",
        "markdown==3.8.2",
        "jinja2==3.1.4",
        "packaging==25.0",
        "python-dateutil==2.9.0.post0",
        "pytz==2025.2",
        "six==1.17.0",
        "urllib3==2.5.0",
        "certifi==2025.8.3",
        "charset-normalizer==3.4.2",
        "idna==3.7",
        "typing_extensions==4.12.2",
        "sympy==1.13.3",
        "mpmath==1.3.0",
        "fsspec==2024.6.1",
        "gitdb==4.0.12",
        "GitPython==3.1.45",
        "smmap==5.0.2",
        "grpcio==1.74.0",
        "absl-py==2.3.1",
        "antlr4-python3-runtime==4.9.3",
        "asttokens==3.0.0",
        "coloredlogs==15.0.1",
        "decorator==5.2.1",
        "executing==2.2.0",
        "fast-histogram==0.14",
        "filetype==1.2.0",
        "humanfriendly==10.0",
        "hydra-core==1.3.2",
        "ipython==9.4.0",
        "ipython_pygments_lexers==1.1.1",
        "jedi==0.19.2",
        "markupsafe==2.1.5",
        "matplotlib-inline==0.1.7",
        "omegaconf==2.3.0",
        "parso==0.8.4",
        "pexpect==4.9.0",
        "pi_heif==1.1.0",
        "pillow-avif-plugin==1.5.2",
        "prompt_toolkit==3.0.51",
        "ptyprocess==0.7.0",
        "pure_eval==0.2.3",
        "py-cpuinfo==9.0.0",
        "pygments==2.19.2",
        "pyparsing==3.2.3",
        "requests-toolbelt==1.0.0",
        "ruamel.yaml==0.18.14",
        "ruamel.yaml.clib==0.2.12",
        "stack-data==0.6.3",
        "tensorboard-data-server==0.7.2",
        "thop==0.1.1.post2209072238",
        "triton==3.0.0",
        "tzdata==2025.2",
        "ultralytics-thop==2.0.15",
        "wcwidth==0.2.13",
        "werkzeug==3.1.3"
    ]
    
    print("📦 Installing core dependencies...")
    for dep in core_dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Install RKNN Toolkit2 from GitHub wheel
    print("📦 Installing RKNN Toolkit2 from GitHub...")
    rknn_wheel_url = "https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    
    if not run_command(f"pip install {rknn_wheel_url}", "Installing RKNN Toolkit2"):
        print("⚠️  RKNN Toolkit2 installation failed. This is optional for basic functionality.")
        print("   You can still use ONNX models without RKNN conversion.")
    
    return True

def install_cuda_dependencies():
    """Install CUDA-specific dependencies for GPU support."""
    cuda_dependencies = [
        "nvidia-cublas-cu12==12.4.2.65",
        "nvidia-cuda-cupti-cu12==12.4.99",
        "nvidia-cuda-nvrtc-cu12==12.4.99",
        "nvidia-cuda-runtime-cu12==12.4.99",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-cufft-cu12==11.2.0.44",
        "nvidia-curand-cu12==10.3.5.119",
        "nvidia-cusolver-cu12==11.6.0.99",
        "nvidia-cusparse-cu12==12.3.0.142",
        "nvidia-nccl-cu12==2.20.5",
        "nvidia-nvjitlink-cu12==12.4.99",
        "nvidia-nvtx-cu12==12.4.99"
    ]
    
    print("🚀 Installing CUDA dependencies...")
    for dep in cuda_dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  CUDA dependency {dep} failed to install. GPU acceleration may be limited.")
    
    return True

def clone_repository():
    """Clone the repository from GitHub."""
    repo_url = "https://github.com/trisgelar/pmu-instance-sampah-fuzzy.git"
    
    if os.path.exists("pmu-instance-sampah-fuzzy"):
        print("📁 Repository already exists, skipping clone")
        return True
    
    return run_command(f"git clone {repo_url}", "Cloning repository")

def create_secrets_file(api_key):
    """Create the secrets.yaml file with the provided API key."""
    secrets_data = {
        'roboflow_api_key': api_key
    }
    
    try:
        with open('secrets.yaml', 'w') as f:
            yaml.dump(secrets_data, f)
        print("✅ secrets.yaml created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create secrets.yaml: {e}")
        return False

def verify_setup():
    """Verify that the setup was successful."""
    print("🔍 Verifying setup...")
    
    # Check if repository exists
    if not os.path.exists("pmu-instance-sampah-fuzzy"):
        print("❌ Repository not found")
        return False
    
    # Check if main file exists
    main_file = "pmu-instance-sampah-fuzzy/main_colab.py"
    if not os.path.exists(main_file):
        print("❌ main_colab.py not found")
        return False
    
    # Check if config file exists
    config_file = "pmu-instance-sampah-fuzzy/config.yaml"
    if not os.path.exists(config_file):
        print("❌ config.yaml not found")
        return False
    
    # Check if secrets file exists
    secrets_file = "secrets.yaml"
    if not os.path.exists(secrets_file):
        print("❌ secrets.yaml not found")
        return False
    
    # Check if key dependencies are installed
    try:
        import ultralytics
        import roboflow
        import torch
        import onnx
        print("✅ Key dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    print("✅ Setup verification completed successfully")
    return True

def setup_colab(api_key=None):
    """
    Complete setup for Google Colab.
    
    Args:
        api_key (str): Roboflow API key. If None, will prompt user.
    """
    print("🚀 Starting Google Colab Setup for PMU Instance Sampah Fuzzy")
    print("=" * 60)
    
    # Step 1: Clone repository
    if not clone_repository():
        print("❌ Setup failed at repository cloning")
        return False
    
    # Step 2: Change to project directory
    os.chdir("pmu-instance-sampah-fuzzy")
    print("📁 Changed to project directory")
    
    # Step 3: Install core dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Step 4: Install CUDA dependencies (optional)
    install_cuda_dependencies()
    
    # Step 5: Create secrets file
    if api_key is None:
        print("⚠️  No API key provided. Please create secrets.yaml manually.")
        print("Example secrets.yaml content:")
        print("roboflow_api_key: YOUR_API_KEY_HERE")
    else:
        if not create_secrets_file(api_key):
            print("❌ Setup failed at secrets file creation")
            return False
    
    # Step 6: Verify setup
    if not verify_setup():
        print("❌ Setup verification failed")
        return False
    
    print("=" * 60)
    print("✅ Google Colab setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. If you haven't provided an API key, create secrets.yaml with your Roboflow API key")
    print("2. Mount Google Drive (optional): from google.colab import drive; drive.mount('/content/drive')")
    print("3. Import and run the system: from main_colab import WasteDetectionSystemColab")
    print("4. Initialize the system: system = WasteDetectionSystemColab()")
    print("5. Start training: system.train_and_export_model('v8n')")
    print("\n🔧 RKNN Toolkit2:")
    print("- RKNN Toolkit2 has been installed from GitHub wheel")
    print("- You can now convert models to RKNN format for edge deployment")
    print("- Use system.convert_and_zip_rknn_models() for RKNN conversion")
    
    return True

def quick_start():
    """Quick start function for immediate execution."""
    print("🎯 Quick Start Mode")
    print("This will set up the environment and start training immediately.")
    
    # Get API key from user
    api_key = input("Enter your Roboflow API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️  No API key provided. Setup will continue but you'll need to create secrets.yaml manually.")
    
    # Run setup
    if setup_colab(api_key if api_key else None):
        print("\n🚀 Ready to start training!")
        
        # Import and initialize system
        try:
            from main_colab import WasteDetectionSystemColab
            system = WasteDetectionSystemColab()
            print("✅ System initialized successfully!")
            
            # Start training
            print("🚀 Starting YOLOv8n training...")
            run_dir = system.train_and_export_model("v8n", epochs=50, batch_size=16)
            
            if run_dir:
                print(f"✅ Training completed! Run directory: {run_dir}")
                
                # Run analysis and inference
                system.analyze_training_run(run_dir, "v8n")
                system.run_inference_and_visualization(run_dir, "v8n", num_inference_images=6)
                
                # Try RKNN conversion
                try:
                    system.convert_and_zip_rknn_models("v8n")
                    print("✅ RKNN conversion completed!")
                except Exception as e:
                    print(f"⚠️  RKNN conversion failed: {e}")
                
                print("🎉 Complete pipeline executed successfully!")
            else:
                print("❌ Training failed")
                
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            print("Please check your configuration and try again.")
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    # Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        print("🔍 Detected Google Colab environment")
        quick_start()
    else:
        print("⚠️  This script is designed for Google Colab")
        print("For local execution, please follow the manual setup instructions.")
        setup_colab() 