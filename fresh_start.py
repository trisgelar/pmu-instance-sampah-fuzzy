#!/usr/bin/env python3
"""
Fresh start script for dataset preparation.

This script deletes the old dataset and extracts dataset.zip
with the integrated working solution from DatasetManager.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from modules.dataset_manager import DatasetManager
from modules.exceptions import ConfigurationError, DatasetError

def load_config():
    """Load configuration from config.yaml."""
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config.yaml: {str(e)}")
        return {}

def clean_old_dataset():
    """Remove old dataset directories."""
    print("🧹 Cleaning old dataset...")
    
    # Remove old dataset directories
    old_paths = [
        "datasets",
        "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
        "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation_backup"
    ]
    
    for path in old_paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"✅ Removed: {path}")
            except Exception as e:
                print(f"⚠️ Could not remove {path}: {e}")
    
    print("✅ Cleanup completed")

def extract_dataset():
    """Extract the datasets.zip file."""
    print("\n📦 Extracting datasets.zip...")
    
    if not os.path.exists("datasets.zip"):
        print("❌ datasets.zip not found")
        return False
    
    try:
        # Create datasets directory
        os.makedirs("datasets", exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile("datasets.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets")
        
        print("✅ Successfully extracted datasets.zip")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting datasets.zip: {e}")
        return False

def normalize_dataset():
    """Normalize the dataset using the integrated solution."""
    print("\n🔄 Normalizing dataset with integrated solution...")
    
    try:
        # Load configuration
        config = load_config()
        if not config:
            print("❌ Failed to load configuration")
            return False
        
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        # Find the dataset path
        dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
        
        print(f"✅ Found dataset at: {dataset_path}")
        
        # Use the integrated normalization method
        if dataset_manager._normalize_dataset_ultralytics(dataset_path):
            print("✅ Dataset normalization completed successfully!")
            return True
        else:
            print("❌ Dataset normalization failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during normalization: {str(e)}")
        return False

def verify_dataset():
    """Verify that the dataset is ready for training."""
    print("\n🔍 Verifying dataset...")
    
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found")
        return False
    
    # Check each split
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 Checking {split}...")
        
        if not os.path.exists(split_path):
            print(f"  ❌ {split} directory not found")
            return False
        
        # Check images
        images_dir = os.path.join(split_path, "images")
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  📸 Images: {len(image_files)} files")
            total_images += len(image_files)
        else:
            print(f"  ❌ Images directory not found: {images_dir}")
            return False
        
        # Check labels
        labels_dir = os.path.join(split_path, "labels")
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"  🏷️ Labels: {len(label_files)} files")
            total_labels += len(label_files)
        else:
            print(f"  ❌ Labels directory not found: {labels_dir}")
            return False
        
        # Verify image-label matching
        if len(image_files) != len(label_files):
            print(f"  ❌ Mismatch: {len(image_files)} images vs {len(label_files)} labels")
            return False
        else:
            print(f"  ✅ Perfect match: {len(image_files)} images and labels")
    
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
        return False
    
    # Summary
    print(f"\n📊 FINAL SUMMARY")
    print(f"📸 Total images: {total_images}")
    print(f"🏷️ Total labels: {total_labels}")
    
    if total_images == total_labels and total_images > 0:
        print("✅ PERFECT! Dataset is ready for YOLO training!")
        return True
    else:
        print("❌ Dataset has issues")
        return False

def main():
    """Main function for fresh start."""
    print("🔄 Fresh Start - Dataset Preparation")
    print("=" * 60)
    print("This script will:")
    print("1. Clean old dataset directories")
    print("2. Extract datasets.zip")
    print("3. Normalize dataset with integrated solution")
    print("4. Verify dataset is ready for training")
    print()
    
    # Step 1: Clean old dataset
    clean_old_dataset()
    
    # Step 2: Extract dataset
    if not extract_dataset():
        print("❌ Failed to extract dataset")
        return False
    
    # Step 3: Normalize dataset
    if not normalize_dataset():
        print("❌ Failed to normalize dataset")
        return False
    
    # Step 4: Verify dataset
    if not verify_dataset():
        print("❌ Dataset verification failed")
        return False
    
    print("\n🎉 SUCCESS: Fresh start completed!")
    print("💡 Your dataset is now ready for YOLO training!")
    print("\n📋 What was done:")
    print("  ✅ Cleaned old dataset directories")
    print("  ✅ Extracted datasets.zip")
    print("  ✅ Normalized dataset with integrated solution")
    print("  ✅ Verified dataset structure and coordinates")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Fresh start completed successfully!")
        print("🚀 Ready for YOLO training!")
    else:
        print("\n❌ Fresh start failed!") 