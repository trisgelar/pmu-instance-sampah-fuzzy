#!/usr/bin/env python3
"""
Test script to verify the dataset fix works with correct labels.
"""

import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_preparation():
    """Test the dataset preparation with the corrected labels."""
    try:
        logger.info("Testing dataset preparation with correct labels...")
        
        # Import the dataset manager
        from modules.dataset_manager import DatasetManager
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            dataset_dir="datasets",
            is_project="abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
            is_version="1"
        )
        
        logger.info("Dataset manager created successfully")
        
        # Test the prepare_datasets method
        success = dataset_manager.prepare_datasets()
        
        if success:
            logger.info("✅ Dataset preparation successful!")
            
            # Verify data.yaml was created correctly
            data_yaml_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation/data.yaml"
            
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Check if it has the correct format
                if 'names' in data and 'train' in data and 'val' in data:
                    logger.info(f"✅ data.yaml created successfully")
                    logger.info(f"  - Names: {data['names']}")
                    logger.info(f"  - Train: {data['train']}")
                    logger.info(f"  - Val: {data['val']}")
                    logger.info(f"  - Test: {data['test']}")
                    
                    # Check if it uses only 'sampah' label
                    if len(data['names']) == 1 and 0 in data['names'] and data['names'][0] == 'sampah':
                        logger.info("✅ Correct label 'sampah' found")
                        return True
                    else:
                        logger.error(f"❌ Unexpected labels: {data['names']} - should only be 'sampah'")
                        return False
                else:
                    logger.error("❌ data.yaml missing required fields")
                    return False
            else:
                logger.error("❌ data.yaml file not found")
                return False
        else:
            logger.error("❌ Dataset preparation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Dataset Fix Test (Correct Labels)")
    print("=" * 50)
    
    success = test_dataset_preparation()
    
    if success:
        print("\n🎉 Dataset fix test passed!")
        print("📋 The dataset now uses the correct labels from Roboflow.")
        print("✅ You can now run: python main_colab.py")
    else:
        print("\n❌ Dataset fix test failed!")
        print("📋 Check the logs above for details")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 