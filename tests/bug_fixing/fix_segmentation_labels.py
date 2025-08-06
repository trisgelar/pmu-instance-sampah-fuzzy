#!/usr/bin/env python3
"""
Fix Segmentation Labels

This script converts YOLO detection labels to segmentation labels using
polygon coordinates from COCO annotations.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def normalize_polygon(polygon: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Normalize polygon coordinates to 0-1 range.
    
    Args:
        polygon: List of [x1, y1, x2, y2, ...] coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of normalized coordinates
    """
    if len(polygon) % 2 != 0:
        logger.warning(f"Invalid polygon: odd number of coordinates {len(polygon)}")
        return []
    
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        
        # Clamp to 0-1 range
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        normalized.extend([x, y])
    
    return normalized

def convert_to_segmentation_labels(dataset_path: str = None) -> bool:
    """
    Convert YOLO detection labels to segmentation labels.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    if dataset_path is None:
        # Try to find dataset in common locations
        possible_paths = [
            "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
            "datasets",
            "dataset"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        else:
            logger.error("No dataset found in common locations")
            return False
    
    logger.info(f"Converting labels to segmentation format at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return False
    
    total_converted = 0
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory not found: {split}")
            continue
        
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        labels_dir = os.path.join(split_path, "labels")
        
        if not os.path.exists(coco_file):
            logger.warning(f"COCO file not found for {split}")
            continue
        
        if not os.path.exists(labels_dir):
            logger.warning(f"Labels directory not found for {split}")
            continue
        
        try:
            # Load COCO annotations
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Create mapping from image ID to image info
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
            split_converted = 0
            for image_id, image_info in image_id_to_info.items():
                filename = image_info['filename']
                img_width = image_info['width']
                img_height = image_info['height']
                base_name = os.path.splitext(filename)[0]
                
                # Create YOLO label file
                label_file = os.path.join(labels_dir, f"{base_name}.txt")
                
                # Get annotations for this image
                annotations = annotations_by_image.get(image_id, [])
                
                # Create YOLO segmentation format lines
                yolo_lines = []
                for annotation in annotations:
                    # Check for segmentation data
                    segmentation = annotation.get('segmentation', [])
                    if not segmentation:
                        logger.warning(f"No segmentation data for annotation in {filename}")
                        continue
                    
                    # Get the first polygon (usually there's only one)
                    polygon = segmentation[0] if isinstance(segmentation, list) else segmentation
                    
                    if not polygon or len(polygon) < 6:  # Need at least 3 points (6 coordinates)
                        logger.warning(f"Invalid polygon for annotation in {filename}")
                        continue
                    
                    # Normalize polygon coordinates
                    normalized_polygon = normalize_polygon(polygon, img_width, img_height)
                    
                    if not normalized_polygon:
                        logger.warning(f"Failed to normalize polygon for {filename}")
                        continue
                    
                    # Create YOLO segmentation line
                    # Format: class_id x1 y1 x2 y2 x3 y3 ...
                    yolo_line = f"0 {' '.join([f'{coord:.6f}' for coord in normalized_polygon])}\n"
                    yolo_lines.append(yolo_line)
                
                # Write the label file
                if yolo_lines:
                    with open(label_file, 'w') as f:
                        f.writelines(yolo_lines)
                    split_converted += 1
            
            logger.info(f"✅ {split}: Converted {split_converted} label files to segmentation format")
            total_converted += split_converted
            
        except Exception as e:
            logger.error(f"Error processing {split}: {e}")
            continue
    
    logger.info(f"✅ Total converted: {total_converted} label files")
    return total_converted > 0

def main():
    """Main function to convert labels to segmentation format."""
    print("🔧 Converting YOLO Labels to Segmentation Format")
    print("=" * 55)
    
    success = convert_to_segmentation_labels()
    
    if success:
        print("\n✅ Successfully converted labels to segmentation format!")
        print("💡 Your dataset is now ready for YOLO segmentation training")
    else:
        print("\n❌ Failed to convert labels")
        print("💡 Check the error messages above for issues")

if __name__ == "__main__":
    main() 