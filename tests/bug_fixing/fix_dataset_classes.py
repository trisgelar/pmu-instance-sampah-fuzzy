#!/usr/bin/env python3
"""
Fix Dataset Classes Script

This script fixes the issue where multiple classes appear in training results
when only 'sampah' should be used. It normalizes COCO annotations to use
only the 'sampah' category and removes other classes.

Usage:
    python -m tests.dataset_tools.fix_dataset_classes
    # or
    from tests.dataset_tools import fix_dataset_classes
    fix_dataset_classes()
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_coco_annotations(dataset_path: str) -> bool:
    """
    Fix COCO annotations to use only 'sampah' category.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Fixing COCO annotations in {dataset_path}")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return False
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        if not os.path.exists(coco_file):
            logger.warning(f"COCO file not found for {split}: {coco_file}")
            continue
        
        logger.info(f"Processing {split} annotations...")
        
        try:
            # Read COCO file
            with open(coco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find 'sampah' category
            sampah_category = None
            for cat in data.get('categories', []):
                if cat.get('name') == 'sampah':
                    sampah_category = cat
                    break
            
            if sampah_category is None:
                logger.error(f"No 'sampah' category found in {split}")
                continue
            
            # Keep only 'sampah' category
            data['categories'] = [sampah_category]
            
            # Update all annotations to use 'sampah' category ID
            sampah_id = sampah_category['id']
            annotations_updated = 0
            
            for ann in data.get('annotations', []):
                old_category_id = ann.get('category_id')
                if old_category_id != sampah_id:
                    ann['category_id'] = sampah_id
                    annotations_updated += 1
            
            # Save updated annotations
            with open(coco_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated {split}: {annotations_updated} annotations normalized to 'sampah'")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {split}: {str(e)}")
            return False
    
    return True

def fix_data_yaml(dataset_path: str) -> bool:
    """
    Fix data.yaml to use only 'sampah' class.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        logger.error(f"data.yaml not found: {data_yaml_path}")
        return False
    
    logger.info(f"Fixing data.yaml: {data_yaml_path}")
    
    try:
        # Read current data.yaml
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Update to use only 'sampah' class
        data['names'] = {0: 'sampah'}
        
        # Save updated data.yaml
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        
        logger.info("✅ data.yaml updated to use only 'sampah' class")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update data.yaml: {str(e)}")
        return False

def verify_fix(dataset_path: str) -> bool:
    """
    Verify that the fix was successful.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if verification passed, False otherwise
    """
    logger.info("Verifying fix...")
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'names' in data and data['names'] == {0: 'sampah'}:
                logger.info("✅ data.yaml correctly configured with only 'sampah'")
            else:
                logger.error(f"❌ data.yaml still has wrong classes: {data.get('names', {})}")
                return False
        except Exception as e:
            logger.error(f"❌ Error reading data.yaml: {str(e)}")
            return False
    
    # Check COCO files
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        if os.path.exists(coco_file):
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                categories = data.get('categories', [])
                if len(categories) == 1 and categories[0].get('name') == 'sampah':
                    logger.info(f"✅ {split} COCO file correctly configured")
                else:
                    logger.error(f"❌ {split} COCO file still has wrong categories: {categories}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error reading {split} COCO file: {str(e)}")
                return False
    
    logger.info("✅ All verifications passed!")
    return True

def find_dataset_path() -> Optional[str]:
    """
    Find the dataset path automatically.
    
    Returns:
        Optional[str]: Path to dataset or None if not found
    """
    possible_paths = [
        "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
        "datasets",
        "datasets/train",
        "datasets/valid",
        "datasets/test"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if this is a dataset directory
            if os.path.exists(os.path.join(path, "data.yaml")) or \
               any(os.path.exists(os.path.join(path, split, "_annotations.coco.json")) 
                  for split in ['train', 'valid', 'test']):
                return path
    
    return None

def fix_dataset_classes(dataset_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Main function to fix dataset classes.
    
    Args:
        dataset_path: Path to dataset directory (auto-detected if None)
        verbose: Whether to print progress information
        
    Returns:
        Dict[str, Any]: Results of the fix operation
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()
    
    if dataset_path is None:
        error_msg = "❌ Dataset path not found. Please run this script from the project root directory."
        if verbose:
            print(error_msg)
        return {'error': error_msg}
    
    if verbose:
        print("🔧 Dataset Classes Fix Script")
        print("=" * 50)
        print(f"Found dataset at: {dataset_path}")
    
    results = {
        'dataset_path': dataset_path,
        'coco_fixed': False,
        'data_yaml_fixed': False,
        'verification_passed': False
    }
    
    # Step 1: Fix COCO annotations
    if verbose:
        print("\n📋 Step 1: Fixing COCO annotations...")
    if fix_coco_annotations(dataset_path):
        results['coco_fixed'] = True
        if verbose:
            print("✅ COCO annotations fixed successfully")
    else:
        if verbose:
            print("❌ Failed to fix COCO annotations")
        return results
    
    # Step 2: Fix data.yaml
    if verbose:
        print("\n📄 Step 2: Fixing data.yaml...")
    if fix_data_yaml(dataset_path):
        results['data_yaml_fixed'] = True
        if verbose:
            print("✅ data.yaml fixed successfully")
    else:
        if verbose:
            print("❌ Failed to fix data.yaml")
        return results
    
    # Step 3: Verify the fix
    if verbose:
        print("\n✅ Step 3: Verifying fix...")
    if verify_fix(dataset_path):
        results['verification_passed'] = True
        if verbose:
            print("✅ All verifications passed!")
    else:
        if verbose:
            print("❌ Verification failed")
        return results
    
    if verbose:
        print("\n🎉 Dataset classes fix completed successfully!")
        print("📋 Summary:")
        print("  - COCO annotations normalized to use only 'sampah' category")
        print("  - data.yaml updated to use only 'sampah' class")
        print("  - All verifications passed")
        print("\n🚀 You can now retrain your model with the correct classes.")
    
    results['success'] = True
    return results

def main():
    """Main function for command-line usage."""
    fix_dataset_classes()

if __name__ == "__main__":
    main() 