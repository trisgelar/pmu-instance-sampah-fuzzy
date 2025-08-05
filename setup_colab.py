#!/usr/bin/env python3
"""
Google Colab Setup Script for PMU Instance Sampah Fuzzy

This script automates the setup process for running the waste detection system
on Google Colab. It handles repository cloning, dependency installation,
configuration setup, and system initialization.

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
    """Install all required dependencies."""
    dependencies = [
        "ultralytics",
        "roboflow", 
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "PyYAML",
        "scikit-fuzzy",
        "tensorboard",
        "onnx",
        "onnxruntime"
    ]
    
    print("📦 Installing dependencies...")
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
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
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Step 4: Create secrets file
    if api_key is None:
        print("⚠️  No API key provided. Please create secrets.yaml manually.")
        print("Example secrets.yaml content:")
        print("roboflow_api_key: YOUR_API_KEY_HERE")
    else:
        if not create_secrets_file(api_key):
            print("❌ Setup failed at secrets file creation")
            return False
    
    # Step 5: Verify setup
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