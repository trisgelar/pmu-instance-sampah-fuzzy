#!/usr/bin/env python3
"""
Basic diagnostic script to check dataset structure without dependencies.

This script checks if the dataset has the basic structure needed for YOLO training.
"""

import os
import sys
from pathlib import Path

def check_basic_structure():
    """Check basic dataset structure."""
    print("🔍 Basic Dataset Structure Check")
    print("=" * 50)
    
    # Check for common dataset paths
    possible_paths = [
        "datasets/sampah-detection-1",
        "datasets/sampah-detection",
        "dataset",
        "data"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found dataset at: {path}")
            break
    
    if not dataset_path:
        print("❌ No dataset found in common locations")
        print("💡 You need to run dataset preparation first")
        return False
    
    print(f"\n📁 Checking structure of: {dataset_path}")
    
    # Check for splits
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 Checking {split}...")
        
        if not os.path.exists(split_path):
            print(f"  ⚠️ {split} directory not found")
            continue
        
        # Check images
        images_dir = os.path.join(split_path, "images")
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  📸 Images: {len(image_files)} files")
            total_images += len(image_files)
        else:
            print(f"  ❌ Images directory not found: {images_dir}")
        
        # Check labels
        labels_dir = os.path.join(split_path, "labels")
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"  🏷️ Labels: {len(label_files)} files")
            total_labels += len(label_files)
        else:
            print(f"  ❌ Labels directory not found: {labels_dir}")
        
        # Check COCO JSON
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        if os.path.exists(coco_file):
            print(f"  📄 COCO JSON: Found")
        else:
            print(f"  ⚠️ COCO JSON: Not found")
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        print(f"\n📄 data.yaml: Found")
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  📋 Content preview:")
                lines = content.split('\n')[:10]
                for line in lines:
                    print(f"    {line}")
        except Exception as e:
            print(f"  ❌ Error reading data.yaml: {e}")
    else:
        print(f"\n❌ data.yaml: Not found")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"📸 Total images: {total_images}")
    print(f"🏷️ Total labels: {total_labels}")
    
    if total_images > 0 and total_labels > 0:
        if total_images == total_labels:
            print("✅ Perfect! Image and label counts match")
            return True
        else:
            print(f"⚠️ Mismatch: {total_images} images vs {total_labels} labels")
            return False
    else:
        print("❌ No images or labels found")
        return False

def main():
    """Main function."""
    print("🔍 Basic Dataset Structure Diagnostic")
    print("=" * 60)
    print("This script checks the basic dataset structure without dependencies.")
    print()
    
    success = check_basic_structure()
    
    if success:
        print("\n✅ Basic structure looks good!")
        print("💡 You can now try training. If it fails, we'll convert to YOLO format.")
    else:
        print("\n❌ Issues found with dataset structure")
        print("💡 We may need to convert to a different format for YOLO training.")
    
    return success

if __name__ == "__main__":
    main() 