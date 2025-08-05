#!/usr/bin/env python3
"""
Script to check and fix polygon segmentation issues in the dataset.
"""

import os
import json
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_polygon_segmentation():
    """Check polygon segmentation in COCO annotations."""
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    print("🔍 Checking Polygon Segmentation")
    print("=" * 50)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        if not os.path.exists(coco_file):
            print(f"❌ COCO file not found for {split}")
            continue
            
        print(f"\n📄 Checking {split} annotations...")
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            print(f"  - Total annotations: {len(annotations)}")
            
            if len(annotations) == 0:
                print(f"  - ⚠️  WARNING: No annotations found!")
                continue
            
            # Check first few annotations for polygon data
            for i, ann in enumerate(annotations[:3]):  # Check first 3 annotations
                print(f"  - Annotation {i+1}:")
                
                # Check for segmentation field
                if 'segmentation' in ann:
                    segmentation = ann['segmentation']
                    print(f"    - Segmentation type: {type(segmentation)}")
                    
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        if isinstance(segmentation[0], list):
                            print(f"    - ✅ Polygon format detected")
                            print(f"    - Polygon points: {len(segmentation[0])}")
                            print(f"    - Sample points: {segmentation[0][:6]}...")
                        else:
                            print(f"    - ⚠️  Unexpected segmentation format")
                    else:
                        print(f"    - ❌ Empty or invalid segmentation")
                else:
                    print(f"    - ❌ No segmentation field found")
                
                # Check category
                category_id = ann.get('category_id', 'unknown')
                print(f"    - Category ID: {category_id}")
                
                # Check if it's the 'sampah' category
                categories = data.get('categories', [])
                for cat in categories:
                    if cat.get('id') == category_id:
                        print(f"    - Category name: {cat.get('name', 'unknown')}")
                        break
            
            # Check categories
            categories = data.get('categories', [])
            print(f"  - Categories: {[cat.get('name', 'unknown') for cat in categories]}")
            
        except Exception as e:
            print(f"  - ❌ Error reading COCO file: {e}")

def check_data_yaml_for_segmentation():
    """Check if data.yaml is configured for segmentation."""
    data_yaml_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation/data.yaml"
    
    print(f"\n📄 Checking data.yaml for segmentation config")
    print("=" * 40)
    
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
        
        # Check if it's configured for segmentation
        if 'names' in data and len(data['names']) > 0:
            print(f"  - ✅ Labels configured: {data['names']}")
            return True
        else:
            print(f"  - ❌ No labels configured")
            return False
        
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False

def suggest_polygon_fixes():
    """Suggest fixes for polygon segmentation issues."""
    print(f"\n🔧 Suggested Fixes for Polygon Segmentation")
    print("=" * 50)
    
    print("1. **Verify Roboflow Polygon Format**:")
    print("   - Open your Roboflow project")
    print("   - Check that 'sampah' annotations use polygon tool")
    print("   - Ensure polygons are properly closed")
    print("   - Verify segmentation masks are visible")
    
    print("\n2. **Regenerate Dataset with Proper Format**:")
    print("   - Delete current dataset folder")
    print("   - Run: python fix_dataset_annotations.py")
    print("   - This will download fresh polygon annotations")
    
    print("\n3. **Check COCO Format Requirements**:")
    print("   - Segmentation should be list of polygon points")
    print("   - Points should be [x1, y1, x2, y2, x3, y3, ...]")
    print("   - Polygon should be closed (first point = last point)")
    
    print("\n4. **Manual Verification**:")
    print("   - Open _annotations.coco.json files")
    print("   - Check 'segmentation' field has polygon arrays")
    print("   - Verify 'category_id' matches 'sampah' category")

def main():
    """Main function."""
    print("🔍 Polygon Segmentation Diagnostic")
    print("=" * 50)
    
    # Check polygon segmentation
    check_polygon_segmentation()
    
    # Check data.yaml
    yaml_ok = check_data_yaml_for_segmentation()
    
    # Suggest fixes
    suggest_polygon_fixes()
    
    if yaml_ok:
        print(f"\n✅ data.yaml looks good!")
        print(f"📋 The issue is likely with polygon format in COCO annotations")
    else:
        print(f"\n❌ data.yaml issues found!")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 