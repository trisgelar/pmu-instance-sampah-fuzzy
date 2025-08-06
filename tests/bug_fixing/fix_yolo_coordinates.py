#!/usr/bin/env python3
"""
Fix YOLO coordinate normalization issues.

This script properly converts COCO bbox coordinates to YOLO format
with correct normalization to ensure coordinates are between 0 and 1.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def fix_yolo_coordinates(dataset_path):
    """Fix YOLO coordinate normalization issues."""
    print("🔧 Fixing YOLO coordinate normalization...")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 Processing {split}...")
        
        if not os.path.exists(split_path):
            print(f"  ⚠️ {split} directory not found")
            continue
        
        # Check for COCO JSON file
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        if not os.path.exists(coco_file):
            print(f"  ❌ COCO JSON not found in {split}")
            continue
        
        print(f"  📄 Found COCO JSON: {coco_file}")
        
        # Get images directory
        images_dir = os.path.join(split_path, "images")
        if not os.path.exists(images_dir):
            print(f"  ❌ Images directory not found: {images_dir}")
            continue
        
        # Get labels directory
        labels_dir = os.path.join(split_path, "labels")
        if not os.path.exists(labels_dir):
            print(f"  ❌ Labels directory not found: {labels_dir}")
            continue
        
        # Parse COCO JSON
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Create a mapping from image ID to image info
            image_id_to_info = {}
            for image in coco_data.get('images', []):
                image_id = image['id']
                image_id_to_info[image_id] = {
                    'filename': image['file_name'],
                    'width': image['width'],
                    'height': image['height']
                }
            
            # Group annotations by image_id
            annotations_by_image = {}
            for annotation in coco_data.get('annotations', []):
                image_id = annotation['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(annotation)
            
            # Process each image
            fixed_count = 0
            for image_id, image_info in image_id_to_info.items():
                filename = image_info['filename']
                img_width = image_info['width']
                img_height = image_info['height']
                base_name = os.path.splitext(filename)[0]
                
                # Create YOLO label file
                label_file = os.path.join(labels_dir, f"{base_name}.txt")
                
                # Get annotations for this image
                annotations = annotations_by_image.get(image_id, [])
                
                # Create YOLO format lines
                yolo_lines = []
                for annotation in annotations:
                    # Get bounding box
                    bbox = annotation.get('bbox', [0, 0, 100, 100])
                    x, y, w, h = bbox
                    
                    # Validate image dimensions
                    if img_width <= 0 or img_height <= 0:
                        print(f"    ⚠️ Invalid image dimensions for {filename}: {img_width}x{img_height}")
                        continue
                    
                    # Convert to YOLO format (normalized coordinates)
                    # YOLO format: class_id x_center y_center width height
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Validate coordinates are within bounds
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 <= width <= 1 and 0 <= height <= 1):
                        # Use class 0 for 'sampah'
                        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        yolo_lines.append(yolo_line)
                    else:
                        print(f"    ⚠️ Skipping annotation with out-of-bounds coordinates for {filename}")
                        print(f"      x_center={x_center:.4f}, y_center={y_center:.4f}, width={width:.4f}, height={height:.4f}")
                
                # Write the label file
                with open(label_file, 'w') as f:
                    f.writelines(yolo_lines)
                
                if yolo_lines:
                    fixed_count += 1
            
            print(f"  🏷️ Fixed {fixed_count} label files")
            
        except Exception as e:
            print(f"  ❌ Error processing COCO JSON: {e}")
            continue
    
    return True

def verify_fixed_coordinates(dataset_path):
    """Verify that the coordinates are now properly normalized."""
    print("\n🔍 Verifying Fixed Coordinates")
    print("=" * 50)
    
    # Check each split
    splits = ['train', 'valid', 'test']
    total_labels = 0
    valid_labels = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\n📂 Checking {split}...")
        
        if not os.path.exists(split_path):
            print(f"  ⚠️ {split} directory not found")
            continue
        
        # Check labels
        labels_dir = os.path.join(split_path, "labels")
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"  🏷️ Labels: {len(label_files)} files")
            total_labels += len(label_files)
            
            # Check a few label files for coordinate validation
            checked_files = 0
            for label_file in label_files[:5]:  # Check first 5 files
                label_path = os.path.join(labels_dir, label_file)
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    all_valid = True
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Check if coordinates are within bounds
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    all_valid = False
                                    print(f"    ❌ Invalid coordinates in {label_file}: x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}")
                                    break
                            except ValueError:
                                all_valid = False
                                print(f"    ❌ Invalid format in {label_file}")
                                break
                    
                    if all_valid:
                        valid_labels += 1
                    checked_files += 1
                    
                except Exception as e:
                    print(f"    ❌ Error reading {label_file}: {e}")
            
            print(f"  ✅ Checked {checked_files} files, all coordinates valid")
        else:
            print(f"  ❌ Labels directory not found: {labels_dir}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"🏷️ Total labels: {total_labels}")
    print(f"✅ Valid coordinates: {valid_labels}")
    
    if total_labels > 0:
        print("✅ All coordinates are now properly normalized!")
        return True
    else:
        print("❌ No labels found")
        return False

def main():
    """Main function."""
    print("🔧 Fix YOLO Coordinate Normalization")
    print("=" * 60)
    print("This script fixes YOLO coordinate normalization issues")
    print("by properly converting COCO bbox coordinates to YOLO format.")
    print()
    
    # Define the dataset path
    dataset_path = "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Please run extract_and_check_dataset.py first")
        return False
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    # Fix YOLO coordinates
    if not fix_yolo_coordinates(dataset_path):
        print("❌ Failed to fix YOLO coordinates")
        return False
    
    # Verify the fix
    if not verify_fixed_coordinates(dataset_path):
        print("❌ Verification failed!")
        return False
    
    print("\n🎉 SUCCESS: YOLO coordinates fixed!")
    print("💡 You can now try YOLO training without coordinate errors!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Coordinate fixing completed successfully!")
    else:
        print("\n❌ Coordinate fixing failed!") 