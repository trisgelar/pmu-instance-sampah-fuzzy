#!/usr/bin/env python3
"""
Diagnostic script to check YOLO label matching.

This script verifies that YOLO can find labels for each image
by checking the parallel structure and file naming conventions.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, List, Tuple
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

def check_yolo_label_matching(dataset_path: str) -> Dict[str, Any]:
    """
    Check if YOLO can find labels for each image.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dict with matching results and issues
    """
    results = {
        'total_images': 0,
        'total_labels': 0,
        'matched_pairs': 0,
        'missing_labels': 0,
        'orphaned_labels': 0,
        'issues': [],
        'details': {}
    }
    
    print("🔍 Checking YOLO Label Matching")
    print("=" * 50)
    
    # Check each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"⚠️ {split} directory not found")
            continue
            
        print(f"\n📂 Checking {split} split...")
        
        # Get image and label directories
        images_dir = os.path.join(split_path, "images")
        labels_dir = os.path.join(split_path, "labels")
        
        if not os.path.exists(images_dir):
            print(f"  ❌ Images directory not found: {images_dir}")
            results['issues'].append(f"{split}: Images directory not found")
            continue
            
        if not os.path.exists(labels_dir):
            print(f"  ❌ Labels directory not found: {labels_dir}")
            results['issues'].append(f"{split}: Labels directory not found")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
        # Get all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        print(f"  📸 Images: {len(image_files)} files")
        print(f"  🏷️ Labels: {len(label_files)} files")
        
        # Check for matching pairs
        matched_pairs = 0
        missing_labels = 0
        
        for img_file in image_files:
            # Get base name without extension
            base_name = os.path.splitext(img_file)[0]
            expected_label = f"{base_name}.txt"
            
            if expected_label in label_files:
                matched_pairs += 1
            else:
                missing_labels += 1
                print(f"    ⚠️ Missing label for: {img_file} (expected: {expected_label})")
        
        # Check for orphaned labels
        orphaned_labels = 0
        for label_file in label_files:
            base_name = os.path.splitext(label_file)[0]
            # Check if any image matches this label
            has_matching_image = False
            for img_file in image_files:
                if os.path.splitext(img_file)[0] == base_name:
                    has_matching_image = True
                    break
            
            if not has_matching_image:
                orphaned_labels += 1
                print(f"    ⚠️ Orphaned label: {label_file}")
        
        # Update results
        results['total_images'] += len(image_files)
        results['total_labels'] += len(label_files)
        results['matched_pairs'] += matched_pairs
        results['missing_labels'] += missing_labels
        results['orphaned_labels'] += orphaned_labels
        
        results['details'][split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'matched_pairs': matched_pairs,
            'missing_labels': missing_labels,
            'orphaned_labels': orphaned_labels
        }
        
        print(f"  ✅ Matched pairs: {matched_pairs}")
        print(f"  ❌ Missing labels: {missing_labels}")
        print(f"  ⚠️ Orphaned labels: {orphaned_labels}")
    
    return results

def check_data_yaml_configuration(dataset_path: str) -> Dict[str, Any]:
    """
    Check data.yaml configuration for YOLO compatibility.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dict with configuration analysis
    """
    results = {
        'valid': False,
        'issues': [],
        'config': {}
    }
    
    print("\n📋 Checking data.yaml Configuration")
    print("=" * 50)
    
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print("❌ data.yaml not found")
        results['issues'].append("data.yaml not found")
        return results
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"📄 data.yaml found: {data_yaml_path}")
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'test', 'names']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                results['issues'].append(f"Missing required field: {field}")
            else:
                print(f"✅ Found field: {field} = {config[field]}")
        
        # Check path configuration
        if config.get('path') == '.':
            print("✅ Using relative path (correct)")
        else:
            print(f"⚠️ Path is: {config.get('path')} (should be '.')")
            results['issues'].append("Path should be '.' for relative paths")
        
        # Check image paths
        for split_name, split_path in [('train', config.get('train')), 
                                     ('val', config.get('val')), 
                                     ('test', config.get('test'))]:
            if split_path:
                full_path = os.path.join(dataset_path, split_path)
                if os.path.exists(full_path):
                    print(f"✅ {split_name} path exists: {split_path}")
                else:
                    print(f"❌ {split_name} path not found: {split_path}")
                    results['issues'].append(f"{split_name} path not found: {split_path}")
        
        # Check class names
        names = config.get('names', {})
        if names == {0: 'sampah'}:
            print("✅ Class configuration correct")
        else:
            print(f"⚠️ Unexpected class names: {names}")
            results['issues'].append(f"Unexpected class names: {names}")
        
        results['config'] = config
        results['valid'] = len(results['issues']) == 0
        
    except Exception as e:
        print(f"❌ Error reading data.yaml: {str(e)}")
        results['issues'].append(f"Error reading data.yaml: {str(e)}")
    
    return results

def main():
    """Main diagnostic function."""
    print("🔍 YOLO Label Matching Diagnostic")
    print("=" * 60)
    print("This script checks if YOLO can find labels for each image")
    print("by verifying the parallel structure and file naming.")
    print()
    
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
        
        # Check label matching
        matching_results = check_yolo_label_matching(dataset_path)
        
        # Check data.yaml configuration
        config_results = check_data_yaml_configuration(dataset_path)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        
        print(f"📸 Total images: {matching_results['total_images']}")
        print(f"🏷️ Total labels: {matching_results['total_labels']}")
        print(f"✅ Matched pairs: {matching_results['matched_pairs']}")
        print(f"❌ Missing labels: {matching_results['missing_labels']}")
        print(f"⚠️ Orphaned labels: {matching_results['orphaned_labels']}")
        
        if matching_results['missing_labels'] > 0:
            print(f"\n❌ ISSUE: {matching_results['missing_labels']} images are missing labels!")
            print("This will cause YOLO training to fail.")
            print("\nPossible solutions:")
            print("1. Run dataset preparation again")
            print("2. Check if COCO to YOLO conversion worked properly")
            print("3. Verify that _annotations.coco.json files exist")
        
        if config_results['valid']:
            print("\n✅ data.yaml configuration is valid")
        else:
            print(f"\n❌ data.yaml has {len(config_results['issues'])} issues:")
            for issue in config_results['issues']:
                print(f"  - {issue}")
        
        # Overall assessment
        if matching_results['missing_labels'] == 0 and config_results['valid']:
            print("\n🎉 SUCCESS: YOLO should be able to find all labels!")
            return True
        else:
            print("\n⚠️ ISSUES FOUND: YOLO may have trouble finding labels")
            return False
            
    except Exception as e:
        print(f"❌ Error during diagnostic: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Diagnostic completed successfully!")
    else:
        print("\n❌ Issues found that need to be resolved!") 