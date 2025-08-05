#!/usr/bin/env python3
"""
Test script to verify our dataset structure matches COCO format exactly.

This script verifies that our dataset follows the same structure as COCO:
- Images directory with image files
- Labels directory with YOLO format annotations
- COCO JSON file (_annotations.coco.json) for conversion
- Proper YOLO format data.yaml
"""

import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

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

def verify_coco_structure(dataset_path: str) -> Dict[str, Any]:
    """
    Verify that the dataset follows COCO structure exactly.
    
    Expected structure:
    dataset/
    ├── train/
    │   ├── images/
    │   ├── labels/
    │   └── _annotations.coco.json
    ├── valid/
    │   ├── images/
    │   ├── labels/
    │   └── _annotations.coco.json
    ├── test/
    │   ├── images/
    │   ├── labels/
    │   └── _annotations.coco.json
    └── data.yaml
    """
    results = {
        'structure_valid': True,
        'issues': [],
        'details': {}
    }
    
    print("🔍 Verifying COCO Structure Compliance")
    print("=" * 50)
    
    # Check main dataset directory
    if not os.path.exists(dataset_path):
        results['structure_valid'] = False
        results['issues'].append(f"Dataset path does not exist: {dataset_path}")
        return results
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            
            print(f"📋 Data.yaml found:")
            print(f"  - Path: {data_yaml.get('path', 'N/A')}")
            print(f"  - Train: {data_yaml.get('train', 'N/A')}")
            print(f"  - Val: {data_yaml.get('val', 'N/A')}")
            print(f"  - Test: {data_yaml.get('test', 'N/A')}")
            print(f"  - Names: {data_yaml.get('names', {})}")
            
            # Verify it uses relative path
            if data_yaml.get('path') == '.':
                print("✅ Data.yaml uses relative path (correct)")
            else:
                print("⚠️ Data.yaml should use relative path '.'")
                results['issues'].append("Data.yaml should use relative path '.'")
            
            # Verify only 'sampah' class
            names = data_yaml.get('names', {})
            if names == {0: 'sampah'}:
                print("✅ Data.yaml has correct class configuration")
            else:
                print(f"⚠️ Data.yaml has unexpected classes: {names}")
                results['issues'].append(f"Data.yaml has unexpected classes: {names}")
                
        except Exception as e:
            results['issues'].append(f"Failed to read data.yaml: {str(e)}")
    else:
        results['issues'].append("data.yaml not found")
    
    # Check each split (train, valid, test)
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 Checking {split} split...")
        
        if not os.path.exists(split_path):
            print(f"⚠️ {split} directory not found")
            results['issues'].append(f"{split} directory not found")
            continue
        
        # Check images directory
        images_dir = os.path.join(split_path, "images")
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  📸 Images directory: {len(image_files)} images")
            results['details'][f'{split}_images'] = len(image_files)
        else:
            print(f"  ❌ Images directory not found in {split}")
            results['issues'].append(f"{split}/images directory not found")
        
        # Check labels directory
        labels_dir = os.path.join(split_path, "labels")
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"  🏷️ Labels directory: {len(label_files)} label files")
            results['details'][f'{split}_labels'] = len(label_files)
        else:
            print(f"  ❌ Labels directory not found in {split}")
            results['issues'].append(f"{split}/labels directory not found")
        
        # Check COCO JSON file
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        if os.path.exists(coco_file):
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                annotations = coco_data.get('annotations', [])
                images = coco_data.get('images', [])
                categories = coco_data.get('categories', [])
                
                print(f"  📄 COCO JSON: {len(annotations)} annotations, {len(images)} images, {len(categories)} categories")
                results['details'][f'{split}_coco'] = {
                    'annotations': len(annotations),
                    'images': len(images),
                    'categories': len(categories)
                }
                
                # Check if categories contain only 'sampah'
                category_names = [cat.get('name', '') for cat in categories]
                if 'sampah' in category_names and len(category_names) == 1:
                    print("  ✅ COCO JSON has correct 'sampah' category")
                else:
                    print(f"  ⚠️ COCO JSON has unexpected categories: {category_names}")
                    results['issues'].append(f"{split} COCO JSON has unexpected categories: {category_names}")
                    
            except Exception as e:
                print(f"  ❌ Failed to read COCO JSON: {str(e)}")
                results['issues'].append(f"Failed to read {split} COCO JSON: {str(e)}")
        else:
            print(f"  ⚠️ COCO JSON file not found in {split}")
            results['issues'].append(f"{split}/_annotations.coco.json not found")
    
    # Summary
    if results['issues']:
        print(f"\n❌ Found {len(results['issues'])} issues:")
        for issue in results['issues']:
            print(f"  - {issue}")
        results['structure_valid'] = False
    else:
        print("\n✅ COCO structure verification passed!")
        results['structure_valid'] = True
    
    return results

def test_coco_structure_compliance():
    """Test that our dataset follows COCO structure exactly."""
    print("🧪 Testing COCO Structure Compliance")
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
        
        # Verify COCO structure
        verification_results = verify_coco_structure(dataset_path)
        
        if verification_results['structure_valid']:
            print("\n✅ COCO structure compliance test passed!")
            print("🎯 Dataset follows the exact same structure as COCO dataset")
            print("📊 Structure details:")
            for key, value in verification_results['details'].items():
                print(f"  - {key}: {value}")
            return True
        else:
            print("\n❌ COCO structure compliance test failed!")
            print("🔧 Issues need to be resolved before training")
            return False
            
    except Exception as e:
        print(f"❌ Error during COCO structure verification: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 COCO Structure Compliance Test")
    print("=" * 60)
    print("This script verifies that our dataset follows the exact")
    print("same structure as the COCO dataset format.")
    print()
    
    success = test_coco_structure_compliance()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🎯 Dataset structure matches COCO format exactly")
        print("📋 Structure includes:")
        print("  - Images directory with image files")
        print("  - Labels directory with YOLO format annotations")
        print("  - COCO JSON file for conversion")
        print("  - Proper YOLO format data.yaml")
    else:
        print("\n❌ Test failed!")
        print("🔧 Check the issues above and fix dataset structure")
    
    print("\n📚 Next steps:")
    print("  1. Ensure dataset follows COCO structure")
    print("  2. Verify YOLO conversion worked correctly")
    print("  3. Run training to confirm structure is valid")

if __name__ == "__main__":
    main() 