#!/usr/bin/env python3
"""
Fix Dataset Classes using Ultralytics Tools

This script uses Ultralytics' built-in conversion tools to properly convert
COCO annotations to YOLO format and fix class issues.

Based on:
- https://github.com/ultralytics/JSON2YOLO
- https://docs.ultralytics.com/datasets/segment/

Usage:
    python -m tests.dataset_tools.fix_dataset_ultralytics
    # or
    from tests.dataset_tools import fix_dataset_ultralytics
    fix_dataset_ultralytics()
"""

import os
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_coco_to_yolo(dataset_path: str) -> bool:
    """
    Convert COCO annotations to YOLO format using Ultralytics tools.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Converting COCO to YOLO format in {dataset_path}")
    
    try:
        from ultralytics.data.converter import convert_coco
        
        # Create YOLO format directory
        yolo_dir = os.path.join(dataset_path, "yolo_format")
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Convert each split
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            coco_file = os.path.join(split_path, "_annotations.coco.json")
            
            if not os.path.exists(coco_file):
                logger.warning(f"COCO file not found for {split}: {coco_file}")
                continue
            
            logger.info(f"Converting {split} annotations...")
            
            # Convert COCO to YOLO format
            convert_coco(
                labels_dir=coco_file,
                save_dir=os.path.join(yolo_dir, split),
                use_keypoints=False,  # Set to True if you have keypoints
            )
            
            logger.info(f"✅ Converted {split} to YOLO format")
        
        return True
        
    except ImportError:
        logger.error("❌ Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        logger.error(f"❌ Conversion failed: {str(e)}")
        return False

def create_yolo_data_yaml(dataset_path: str) -> bool:
    """
    Create YOLO format data.yaml file.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    yolo_dir = os.path.join(dataset_path, "yolo_format")
    
    if not os.path.exists(yolo_dir):
        logger.error(f"YOLO format directory not found: {yolo_dir}")
        return False
    
    # Create YOLO format data.yaml
    yolo_data_yaml = {
        'path': os.path.abspath(yolo_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {0: 'sampah'}  # Only 'sampah' class
    }
    
    yolo_data_yaml_path = os.path.join(yolo_dir, "data.yaml")
    
    try:
        with open(yolo_data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_data_yaml, f, sort_keys=False, default_flow_style=False)
        
        logger.info(f"✅ Created YOLO data.yaml: {yolo_data_yaml_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create YOLO data.yaml: {str(e)}")
        return False

def backup_original_dataset(dataset_path: str) -> bool:
    """
    Create a backup of the original dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    backup_path = f"{dataset_path}_backup"
    
    try:
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        shutil.copytree(dataset_path, backup_path)
        logger.info(f"✅ Created backup at: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {str(e)}")
        return False

def replace_with_yolo_format(dataset_path: str) -> bool:
    """
    Replace the original dataset with YOLO format.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    yolo_dir = os.path.join(dataset_path, "yolo_format")
    
    if not os.path.exists(yolo_dir):
        logger.error(f"YOLO format directory not found: {yolo_dir}")
        return False
    
    try:
        # Remove original COCO files
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                shutil.rmtree(split_path)
        
        # Move YOLO format files to main dataset directory
        for item in os.listdir(yolo_dir):
            src = os.path.join(yolo_dir, item)
            dst = os.path.join(dataset_path, item)
            shutil.move(src, dst)
        
        # Remove empty yolo_format directory
        os.rmdir(yolo_dir)
        
        logger.info("✅ Replaced dataset with YOLO format")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to replace dataset: {str(e)}")
        return False

def verify_yolo_format(dataset_path: str) -> bool:
    """
    Verify that the dataset is now in YOLO format.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if verification passed, False otherwise
    """
    logger.info("Verifying YOLO format...")
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        logger.error("❌ data.yaml not found")
        return False
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data and data['names'] == {0: 'sampah'}:
            logger.info("✅ data.yaml correctly configured with only 'sampah'")
        else:
            logger.error(f"❌ data.yaml has wrong classes: {data.get('names', {})}")
            return False
        
        # Check YOLO format directories
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                images_dir = os.path.join(split_path, "images")
                labels_dir = os.path.join(split_path, "labels")
                
                if os.path.exists(images_dir) and os.path.exists(labels_dir):
                    image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                    logger.info(f"✅ {split}: {image_count} images, {label_count} labels")
                else:
                    logger.error(f"❌ {split}: Missing images or labels directory")
                    return False
            else:
                logger.warning(f"⚠️ {split}: Directory not found")
        
        logger.info("✅ All verifications passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during verification: {str(e)}")
        return False

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
            if os.path.exists(os.path.join(path, "data.yaml")) or \
               any(os.path.exists(os.path.join(path, split, "_annotations.coco.json")) 
                  for split in ['train', 'valid', 'test']):
                return path
    
    return None

def fix_dataset_ultralytics(dataset_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Main function to fix dataset using Ultralytics tools.
    
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
        print("🔧 Dataset Classes Fix using Ultralytics Tools")
        print("=" * 60)
        print(f"Found dataset at: {dataset_path}")
    
    results = {
        'dataset_path': dataset_path,
        'backup_created': False,
        'conversion_successful': False,
        'data_yaml_created': False,
        'replacement_successful': False,
        'verification_passed': False,
        'backup_path': None
    }
    
    # Step 1: Create backup
    if verbose:
        print("\n📁 Step 1: Creating backup...")
    if backup_original_dataset(dataset_path):
        results['backup_created'] = True
        results['backup_path'] = f"{dataset_path}_backup"
        if verbose:
            print("✅ Backup created successfully")
    else:
        if verbose:
            print("❌ Failed to create backup")
        return results
    
    # Step 2: Convert COCO to YOLO format
    if verbose:
        print("\n🔄 Step 2: Converting COCO to YOLO format...")
    if convert_coco_to_yolo(dataset_path):
        results['conversion_successful'] = True
        if verbose:
            print("✅ COCO to YOLO conversion successful")
    else:
        if verbose:
            print("❌ Failed to convert COCO to YOLO format")
        return results
    
    # Step 3: Create YOLO data.yaml
    if verbose:
        print("\n📄 Step 3: Creating YOLO data.yaml...")
    if create_yolo_data_yaml(dataset_path):
        results['data_yaml_created'] = True
        if verbose:
            print("✅ YOLO data.yaml created successfully")
    else:
        if verbose:
            print("❌ Failed to create YOLO data.yaml")
        return results
    
    # Step 4: Replace original dataset with YOLO format
    if verbose:
        print("\n🔄 Step 4: Replacing dataset with YOLO format...")
    if replace_with_yolo_format(dataset_path):
        results['replacement_successful'] = True
        if verbose:
            print("✅ Dataset replaced with YOLO format")
    else:
        if verbose:
            print("❌ Failed to replace dataset with YOLO format")
        return results
    
    # Step 5: Verify the fix
    if verbose:
        print("\n✅ Step 5: Verifying YOLO format...")
    if verify_yolo_format(dataset_path):
        results['verification_passed'] = True
        if verbose:
            print("✅ All verifications passed!")
    else:
        if verbose:
            print("❌ Verification failed")
        return results
    
    if verbose:
        print("\n🎉 Dataset conversion completed successfully!")
        print("📋 Summary:")
        print("  - Created backup of original dataset")
        print("  - Converted COCO annotations to YOLO format")
        print("  - Updated data.yaml to use only 'sampah' class")
        print("  - All verifications passed")
        print("\n🚀 You can now retrain your model with the correct YOLO format.")
        print(f"📁 Backup available at: {results['backup_path']}")
    
    results['success'] = True
    return results

def main():
    """Main function for command-line usage."""
    fix_dataset_ultralytics()

if __name__ == "__main__":
    main() 