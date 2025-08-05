#!/usr/bin/env python3
"""
Test script to verify sampah normalization works correctly.
"""

import os
import json
import logging
from modules.dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sampah_normalization():
    """Test that all segmentations are normalized to 'sampah' category."""
    try:
        logger.info("Testing sampah normalization...")
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            dataset_dir="datasets",
            is_project="abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
            is_version="1"
        )
        
        logger.info("Dataset manager created successfully")
        
        # Test the normalization on existing dataset
        dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
        
        if not os.path.exists(dataset_path):
            logger.error("Dataset path not found, please run dataset preparation first")
            return False
        
        # Test normalization
        dataset_manager._normalize_coco_annotations(dataset_path)
        
        # Verify the results
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            coco_file = os.path.join(split_path, "_annotations.coco.json")
            
            if not os.path.exists(coco_file):
                logger.warning(f"COCO file not found for {split}")
                continue
            
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                categories = data.get('categories', [])
                annotations = data.get('annotations', [])
                
                logger.info(f"\n📄 {split} verification:")
                logger.info(f"  - Categories: {[cat.get('name') for cat in categories]}")
                logger.info(f"  - Total annotations: {len(annotations)}")
                
                # Check if all categories are 'sampah'
                category_names = [cat.get('name') for cat in categories]
                if len(category_names) == 1 and category_names[0] == 'sampah':
                    logger.info(f"  - ✅ All categories are 'sampah'")
                else:
                    logger.error(f"  - ❌ Unexpected categories: {category_names}")
                    return False
                
                # Check if all annotations use 'sampah' category
                sampah_category_id = None
                for cat in categories:
                    if cat.get('name') == 'sampah':
                        sampah_category_id = cat.get('id')
                        break
                
                if sampah_category_id is None:
                    logger.error(f"  - ❌ No 'sampah' category found")
                    return False
                
                non_sampah_annotations = [ann for ann in annotations 
                                        if ann.get('category_id') != sampah_category_id]
                
                if len(non_sampah_annotations) == 0:
                    logger.info(f"  - ✅ All annotations use 'sampah' category")
                else:
                    logger.error(f"  - ❌ {len(non_sampah_annotations)} annotations not using 'sampah' category")
                    return False
                
            except Exception as e:
                logger.error(f"  - ❌ Error reading {split} COCO file: {e}")
                return False
        
        logger.info("\n🎉 Sampah normalization test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main function."""
    print("🧪 Testing Sampah Normalization")
    print("=" * 50)
    
    success = test_sampah_normalization()
    
    if success:
        print("\n✅ All segmentations are now normalized to 'sampah'!")
        print("📋 Next steps:")
        print("1. Run: python main_colab.py")
        print("2. The 'no labels found' warning should be gone")
        print("3. Training should work with only 'sampah' category")
    else:
        print("\n❌ Sampah normalization test failed")
        print("📋 Troubleshooting:")
        print("1. Check if dataset exists")
        print("2. Verify COCO annotation files")
        print("3. Run dataset preparation first")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 