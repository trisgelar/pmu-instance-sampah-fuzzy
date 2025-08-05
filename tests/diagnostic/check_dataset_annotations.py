#!/usr/bin/env python3
"""
Diagnostic script to check dataset annotations and identify the issue.
"""

import os
import json
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset_structure():
    """Check the dataset structure and files."""
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    print("🔍 Checking Dataset Structure")
    print("=" * 50)
    print(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset directory exists")
    
    # Check for train, valid, test directories
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"✅ {split} directory exists")
            
            # Count images
            images = [f for f in os.listdir(split_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  - Images: {len(images)}")
            
            # Check for COCO annotations
            coco_file = os.path.join(split_path, "_annotations.coco.json")
            if os.path.exists(coco_file):
                print(f"  - COCO annotations: ✅ exists")
                check_coco_file(coco_file)
            else:
                print(f"  - COCO annotations: ❌ missing")
        else:
            print(f"❌ {split} directory missing")
    
    return True

def check_coco_file(coco_path):
    """Check a specific COCO annotations file."""
    try:
        with open(coco_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"    - File size: {os.path.getsize(coco_path)} bytes")
        print(f"    - Images: {len(data.get('images', []))}")
        print(f"    - Annotations: {len(data.get('annotations', []))}")
        print(f"    - Categories: {len(data.get('categories', []))}")
        
        if len(data.get('annotations', [])) == 0:
            print(f"    - ⚠️  WARNING: No annotations found!")
            return False
        
        # Check if annotations have segmentation data
        annotations = data.get('annotations', [])
        if annotations:
            first_ann = annotations[0]
            if 'segmentation' in first_ann:
                print(f"    - ✅ Segmentation data present")
            else:
                print(f"    - ❌ No segmentation data found")
        
        return True
        
    except Exception as e:
        print(f"    - ❌ Error reading COCO file: {e}")
        return False

def check_data_yaml():
    """Check the data.yaml file."""
    data_yaml_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation/data.yaml"
    
    print(f"\n📄 Checking data.yaml")
    print("=" * 30)
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ data.yaml not found")
        return False
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"✅ data.yaml is valid")
        print(f"  - Path: {data.get('path', 'Not found')}")
        print(f"  - Train: {data.get('train', 'Not found')}")
        print(f"  - Val: {data.get('val', 'Not found')}")
        print(f"  - Test: {data.get('test', 'Not found')}")
        print(f"  - Names: {data.get('names', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False

def check_yolo_cache():
    """Check YOLO cache files."""
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    print(f"\n🗂️  Checking YOLO Cache Files")
    print("=" * 30)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        cache_file = os.path.join(split_path, f"{split}.cache")
        
        if os.path.exists(cache_file):
            size = os.path.getsize(cache_file)
            print(f"  - {split}.cache: {size} bytes")
            if size == 0:
                print(f"    ⚠️  WARNING: Empty cache file!")
        else:
            print(f"  - {split}.cache: not found")

def suggest_fixes():
    """Suggest fixes for the annotation issues."""
    print(f"\n🔧 Suggested Fixes")
    print("=" * 30)
    
    print("1. **Regenerate dataset from Roboflow**:")
    print("   - Delete the current dataset folder")
    print("   - Run: python main_colab.py")
    print("   - This will download fresh annotations")
    
    print("\n2. **Check Roboflow project settings**:")
    print("   - Ensure your Roboflow project has proper annotations")
    print("   - Verify that 'sampah' labels are correctly assigned")
    print("   - Make sure segmentation masks are properly drawn")
    
    print("\n3. **Manual verification**:")
    print("   - Open a few images in your dataset")
    print("   - Check if they have proper 'sampah' annotations")
    print("   - Verify segmentation masks are visible")

def main():
    """Main diagnostic function."""
    print("🔍 Dataset Annotations Diagnostic")
    print("=" * 50)
    
    # Check dataset structure
    structure_ok = check_dataset_structure()
    
    # Check data.yaml
    yaml_ok = check_data_yaml()
    
    # Check YOLO cache
    check_yolo_cache()
    
    # Suggest fixes
    suggest_fixes()
    
    if not structure_ok:
        print(f"\n❌ Dataset structure issues found!")
        return False
    elif not yaml_ok:
        print(f"\n❌ data.yaml issues found!")
        return False
    else:
        print(f"\n✅ Dataset structure looks good!")
        print(f"📋 The warning might be due to empty annotations in COCO files")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 