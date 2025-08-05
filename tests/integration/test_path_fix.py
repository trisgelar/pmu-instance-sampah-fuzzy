#!/usr/bin/env python3
"""
Test script to verify the path fixing functionality.

This script tests the new path fixing methods in DatasetManager
to ensure absolute paths are converted to relative paths.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from modules.dataset_manager import DatasetManager
from modules.exceptions import ConfigurationError, DatasetError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {str(e)}")
        return {}

def test_path_fixing():
    """Test the path fixing functionality."""
    print("🧪 Testing Path Fixing Functionality")
    print("=" * 50)
    
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    try:
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        # Check if dataset exists
        dataset_path = os.path.join(dataset_manager.DATASET_DIR, dataset_manager.ROBOFLOW_IS_PROJECT)
        
        if not os.path.exists(dataset_path):
            print(f"📁 Dataset not found at: {dataset_path}")
            print("💡 You need to run prepare_datasets() first to download/extract the dataset")
            return False
        
        print(f"📁 Found dataset at: {dataset_path}")
        
        # Check current data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                current_data = yaml.safe_load(f)
            print(f"📋 Current data.yaml path: {current_data.get('path', 'N/A')}")
            
            # Check if it's an absolute path
            current_path = current_data.get('path', '')
            if current_path and (current_path.startswith('D:\\') or current_path.startswith('/') or '\\' in current_path):
                print("⚠️ Found absolute path in data.yaml - will fix this")
            else:
                print("✅ Data.yaml already uses relative path")
        else:
            print("📋 No data.yaml found")
        
        # Fix path issues
        print("\n🔧 Fixing path issues...")
        path_fix_success = dataset_manager.fix_data_yaml_paths()
        
        if path_fix_success:
            print("✅ Path fixing completed successfully!")
            
            # Check the updated data.yaml
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    updated_data = yaml.safe_load(f)
                print(f"📋 Updated data.yaml path: {updated_data.get('path', 'N/A')}")
                
                # Verify it's now relative
                updated_path = updated_data.get('path', '')
                if updated_path == '.':
                    print("✅ Successfully converted to relative path!")
                else:
                    print(f"⚠️ Path is still: {updated_path}")
            
            return True
        else:
            print("❌ Path fixing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during path fixing: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Path Fix Test")
    print("=" * 60)
    print("This script tests the path fixing functionality")
    print("to ensure absolute paths are converted to relative paths.")
    print()
    
    success = test_path_fixing()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🎯 Data.yaml should now use relative paths")
    else:
        print("\n❌ Test failed!")
        print("🔧 Check the logs for detailed error information")
    
    print("\n📚 Next steps:")
    print("  1. Run training to verify the fix works")
    print("  2. Check that no more path-related errors occur")
    print("  3. Verify YOLO training works with relative paths")

if __name__ == "__main__":
    main() 