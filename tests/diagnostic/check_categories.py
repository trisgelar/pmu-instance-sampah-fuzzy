#!/usr/bin/env python3
"""
Script to check categories and their annotations in the dataset.
"""

import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_categories():
    """Check categories and their annotations."""
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    print("🔍 Checking Categories and Annotations")
    print("=" * 50)
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        if not os.path.exists(coco_file):
            print(f"❌ COCO file not found for {split}")
            continue
            
        print(f"\n📄 Checking {split} categories...")
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            categories = data.get('categories', [])
            
            print(f"  - Total annotations: {len(annotations)}")
            print(f"  - Categories: {[cat.get('name', 'unknown') for cat in categories]}")
            
            # Count annotations per category
            category_counts = {}
            for ann in annotations:
                category_id = ann.get('category_id', 'unknown')
                category_name = 'unknown'
                
                # Find category name
                for cat in categories:
                    if cat.get('id') == category_id:
                        category_name = cat.get('name', 'unknown')
                        break
                
                if category_name not in category_counts:
                    category_counts[category_name] = 0
                category_counts[category_name] += 1
            
            print(f"  - Annotations per category:")
            for cat_name, count in category_counts.items():
                print(f"    - {cat_name}: {count} annotations")
            
            # Check segmentation for each category
            print(f"  - Segmentation check per category:")
            for cat_name in category_counts.keys():
                cat_annotations = [ann for ann in annotations 
                                 if any(cat.get('id') == ann.get('category_id') 
                                       and cat.get('name') == cat_name 
                                       for cat in categories)]
                
                if cat_annotations:
                    first_ann = cat_annotations[0]
                    has_segmentation = 'segmentation' in first_ann and first_ann['segmentation']
                    print(f"    - {cat_name}: {'✅' if has_segmentation else '❌'} segmentation")
                else:
                    print(f"    - {cat_name}: ❌ no annotations found")
            
        except Exception as e:
            print(f"  - ❌ Error reading COCO file: {e}")

def suggest_category_fixes():
    """Suggest fixes for category issues."""
    print(f"\n🔧 Suggested Fixes for Category Issues")
    print("=" * 50)
    
    print("1. **Filter to Only 'sampah' Category**:")
    print("   - The DatasetManager should only use 'sampah' category")
    print("   - Ignore 'objects-mG6e' category")
    print("   - This will fix the 'no labels found' warning")
    
    print("\n2. **Check Roboflow Project**:")
    print("   - Open your Roboflow project")
    print("   - Verify which images have 'sampah' annotations")
    print("   - Check if 'objects-mG6e' is an old/unused category")
    print("   - Consider removing 'objects-mG6e' category if unused")
    
    print("\n3. **Regenerate with Category Filter**:")
    print("   - Run: python fix_dataset_annotations.py")
    print("   - This will use only 'sampah' category")
    print("   - Ignore other categories automatically")
    
    print("\n4. **Manual Category Cleanup**:")
    print("   - In Roboflow, delete 'objects-mG6e' category if unused")
    print("   - Re-export dataset with only 'sampah' category")
    print("   - This will give you a clean dataset")

def main():
    """Main function."""
    print("🔍 Category Analysis")
    print("=" * 50)
    
    # Check categories
    check_categories()
    
    # Suggest fixes
    suggest_category_fixes()
    
    print(f"\n📋 Summary:")
    print(f"- Your dataset has 2 categories: 'objects-mG6e' and 'sampah'")
    print(f"- This is causing YOLO confusion")
    print(f"- Solution: Use only 'sampah' category")
    print(f"- Run the fix script to filter categories properly")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 