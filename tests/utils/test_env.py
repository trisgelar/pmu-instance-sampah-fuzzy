#!/usr/bin/env python3
"""
Test script to verify Python environment and yaml import.
Run this with: ./run_with_venv.sh python test_env.py
"""

import sys
import os

def test_environment():
    """Test the Python environment and imports."""
    print("=== Environment Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Test yaml import
    try:
        import yaml
        print(f"✅ yaml imported successfully (version: {yaml.__version__})")
    except ImportError as e:
        print(f"❌ yaml import failed: {e}")
        return False
    
    # Test other common imports
    try:
        import torch
        print(f"✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
    
    try:
        import ultralytics
        print(f"✅ ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ ultralytics import failed: {e}")
    
    print("=== Environment Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1) 