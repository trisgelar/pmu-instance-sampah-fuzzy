#!/usr/bin/env python3
"""
Simple runner script for the dataset fix integration test.

This script provides an easy way to run the dataset fix integration test
with proper path setup and error handling.

Usage:
    python tests/run_dataset_fix_test.py
    # or
    python -m tests.run_dataset_fix_test
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the dataset fix integration test."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    test_file = project_root / "tests" / "dataset_tools" / "test_dataset_fix_integration.py"
    
    print("🧪 Running Dataset Fix Integration Test")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Test file: {test_file}")
    print()
    
    # Check if test file exists
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    # Change to project root directory
    os.chdir(project_root)
    
    try:
        # Run the test
        print("🚀 Starting test execution...")
        result = subprocess.run([
            sys.executable, 
            str(test_file)
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
            return 0
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            return result.returncode
            
    except Exception as e:
        print(f"❌ Error running test: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 