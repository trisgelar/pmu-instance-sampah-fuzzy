#!/usr/bin/env python3
"""
Test script to verify the dataset fix works correctly.

This script tests the fixed DatasetManager methods to ensure
the YOLO conversion works properly.
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

def test_dataset_fix():
    """Test the dataset fixing functionality."""
    print("🧪 Testing Dataset Fix Verification")
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
        
        # Fix dataset classes
        print("🔧 Fixing dataset classes...")
        success = dataset_manager.fix_dataset_classes()
        
        if success:
            print("✅ Dataset fixing completed successfully!")
            
            # Check the new data.yaml
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    new_data = yaml.safe_load(f)
                print(f"📋 New data.yaml path: {new_data.get('path', 'N/A')}")
                print(f"📋 New data.yaml names: {new_data.get('names', {})}")
            
            # Validate again after fixing
            print("\n🔍 Re-validating dataset after fixing...")
            validation_results = dataset_manager.validate_dataset_format()
            
            if validation_results['issues']:
                print("⚠️ Remaining issues:")
                for issue in validation_results['issues']:
                    print(f"  - {issue}")
            else:
                print("✅ All issues resolved!")
            
            return True
        else:
            print("❌ Dataset fixing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during fixing: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Dataset Fix Verification Test")
    print("=" * 60)
    print("This script tests the fixed dataset fixing functionality")
    print("to ensure YOLO conversion works properly.")
    print()
    
    success = test_dataset_fix()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🎯 Dataset should now be properly formatted for YOLO training")
    else:
        print("\n❌ Test failed!")
        print("🔧 Check the logs for detailed error information")
    
    print("\n📚 Next steps:")
    print("  1. Run training to verify the fix works")
    print("  2. Check that only 'sampah' class appears in training")
    print("  3. Verify no more 'no label found' warnings")

if __name__ == "__main__":
    main() 