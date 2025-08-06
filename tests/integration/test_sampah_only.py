#!/usr/bin/env python3
"""
Test script to verify only 'sampah' label is used.
"""

import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sampah_only():
    """Test that only 'sampah' label is used."""
    try:
        logger.info("Testing that only 'sampah' label is used...")
        
        # Import the dataset manager
        from modules.dataset_manager import DatasetManager
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            dataset_dir="datasets",
            is_project="abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
            is_version="1"
        )
        
        logger.info("Dataset manager created successfully")
        
        # Test the _get_class_names method directly
        class_names = dataset_manager._get_class_names()
        logger.info(f"Class names returned: {class_names}")
        
        if len(class_names) == 1 and class_names[0] == 'sampah':
            logger.info("✅ Only 'sampah' label returned")
            return True
        else:
            logger.error(f"❌ Unexpected class names: {class_names}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Sampah Only Test")
    print("=" * 50)
    
    success = test_sampah_only()
    
    if success:
        print("\n🎉 Sampah only test passed!")
        print("📋 The dataset now uses only 'sampah' label.")
        print("✅ You can now run: python main_colab.py")
    else:
        print("\n❌ Sampah only test failed!")
        print("📋 Check the logs above for details")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 