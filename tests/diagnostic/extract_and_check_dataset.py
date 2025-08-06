#!/usr/bin/env python3
"""
Extract and check dataset structure.

This script extracts the datasets.zip file and checks if it's ready for YOLO training.
"""

import os
import zipfile
import shutil
from pathlib import Path

def extract_dataset():
    """Extract the datasets.zip file."""
    print("📦 Extracting datasets.zip...")
    
    if not os.path.exists("datasets.zip"):
        print("❌ datasets.zip not found")
        return False
    
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile("datasets.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets")
        
        print("✅ Successfully extracted datasets.zip")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting datasets.zip: {e}")
        return False

def check_extracted_structure():
    """Check the structure of the extracted dataset."""
    print("\n🔍 Checking extracted dataset structure...")
    
    # Check for the actual dataset path
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    print(f"✅ Found dataset at: {dataset_path}")
    print(f"\n📁 Checking structure of: {dataset_path}")
    
    # List all contents
    print("📋 Contents:")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"  📂 {item}/")
        else:
            print(f"  📄 {item}")
    
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
    print("📦 Dataset Extraction and Structure Check")
    print("=" * 60)
    
    # Extract dataset
    if not extract_dataset():
        return False
    
    # Check structure
    success = check_extracted_structure()
    
    if success:
        print("\n✅ Dataset is ready for YOLO training!")
        print("💡 You can now try training. If it fails, we'll convert to YOLO format.")
    else:
        print("\n❌ Issues found with dataset structure")
        print("💡 We may need to convert to a different format for YOLO training.")
    
    return success

if __name__ == "__main__":
    main() 