#!/usr/bin/env python3
"""
Check Segmentation Dataset Format

This script checks if the dataset is properly formatted for YOLO segmentation training
and provides guidance on fixing any issues.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_segmentation_format(dataset_path: str = None) -> Dict[str, Any]:
    """
    Check if dataset is properly formatted for YOLO segmentation.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict with analysis results
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
            return {"error": "Dataset not found"}
    
    results = {
        "dataset_path": dataset_path,
        "has_segmentation_annotations": False,
        "has_bounding_box_annotations": False,
        "data_yaml_issues": [],
        "recommendations": [],
        "is_segmentation_ready": False
    }
    
    logger.info(f"Checking dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        results["error"] = f"Dataset path not found: {dataset_path}"
        return results
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            logger.info(f"data.yaml content: {data}")
            
            # Check if using absolute paths
            path_value = data.get('path', '')
            if path_value and (path_value.startswith('D:\\') or path_value.startswith('/') or ':' in path_value):
                logger.info("✅ Using absolute paths in data.yaml")
            else:
                logger.warning("⚠️ Using relative paths in data.yaml")
                results["data_yaml_issues"].append("Using relative paths instead of absolute")
                results["recommendations"].append("Convert to absolute paths for better compatibility")
            
            # Check class names
            names = data.get('names', {})
            if names == {0: 'sampah'}:
                logger.info("✅ Correct class configuration: only 'sampah' class")
            else:
                logger.warning(f"⚠️ Incorrect class configuration: {names}")
                results["data_yaml_issues"].append(f"Incorrect class names: {names}")
                results["recommendations"].append("Fix class configuration to use only 'sampah'")
                
        except Exception as e:
            logger.error(f"Error reading data.yaml: {e}")
            results["data_yaml_issues"].append(f"Error reading data.yaml: {e}")
    else:
        logger.error("❌ data.yaml not found")
        results["data_yaml_issues"].append("data.yaml not found")
        results["recommendations"].append("Create data.yaml file")
    
    # Check COCO annotations for segmentation
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory not found: {split}")
            continue
        
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        if not os.path.exists(coco_file):
            logger.warning(f"COCO file not found for {split}")
            continue
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            annotations = coco_data.get('annotations', [])
            if annotations:
                # Check for segmentation annotations
                has_segmentation = any('segmentation' in ann and ann['segmentation'] for ann in annotations)
                has_bbox = any('bbox' in ann for ann in annotations)
                
                if has_segmentation:
                    logger.info(f"✅ {split}: Found segmentation annotations")
                    results["has_segmentation_annotations"] = True
                else:
                    logger.warning(f"⚠️ {split}: No segmentation annotations found")
                    results["recommendations"].append(f"Add segmentation masks to {split} annotations")
                
                if has_bbox:
                    logger.info(f"✅ {split}: Found bounding box annotations")
                    results["has_bounding_box_annotations"] = True
                
                # Check annotation format
                sample_ann = annotations[0] if annotations else {}
                if 'segmentation' in sample_ann:
                    seg = sample_ann['segmentation']
                    if isinstance(seg, list) and len(seg) > 0:
                        if isinstance(seg[0], list):
                            logger.info(f"✅ {split}: Segmentation format looks correct (polygon)")
                        else:
                            logger.warning(f"⚠️ {split}: Unexpected segmentation format")
                    else:
                        logger.warning(f"⚠️ {split}: Empty or invalid segmentation")
                else:
                    logger.warning(f"⚠️ {split}: No segmentation field in annotations")
                    
        except Exception as e:
            logger.error(f"Error reading COCO file for {split}: {e}")
    
    # Check YOLO labels
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        labels_dir = os.path.join(split_path, "labels")
        
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            if label_files:
                # Check first label file for format
                first_label = os.path.join(labels_dir, label_files[0])
                try:
                    with open(first_label, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            parts = first_line.split()
                            if len(parts) >= 5:
                                logger.info(f"✅ {split}: YOLO labels found (detection format)")
                                # For segmentation, we might need polygon coordinates
                                if len(parts) > 5:
                                    logger.info(f"✅ {split}: YOLO labels contain additional data (possibly segmentation)")
                                else:
                                    logger.warning(f"⚠️ {split}: YOLO labels appear to be detection format only")
                                    results["recommendations"].append(f"Convert {split} labels to segmentation format")
                except Exception as e:
                    logger.error(f"Error reading label file: {e}")
            else:
                logger.warning(f"⚠️ {split}: No YOLO label files found")
        else:
            logger.warning(f"⚠️ {split}: Labels directory not found")
    
    # Determine if ready for segmentation
    if (results["has_segmentation_annotations"] and 
        not results["data_yaml_issues"] and
        results["has_bounding_box_annotations"]):
        results["is_segmentation_ready"] = True
        logger.info("✅ Dataset appears ready for segmentation training")
    else:
        logger.warning("⚠️ Dataset may not be ready for segmentation training")
        results["recommendations"].append("Ensure dataset has proper segmentation annotations")
    
    return results

def main():
    """Main function to check segmentation format."""
    print("🔍 Checking Segmentation Dataset Format")
    print("=" * 50)
    
    results = check_segmentation_format()
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return
    
    print(f"\n📊 Analysis Results:")
    print(f"Dataset Path: {results['dataset_path']}")
    print(f"Has Segmentation Annotations: {results['has_segmentation_annotations']}")
    print(f"Has Bounding Box Annotations: {results['has_bounding_box_annotations']}")
    print(f"Ready for Segmentation: {results['is_segmentation_ready']}")
    
    if results["data_yaml_issues"]:
        print(f"\n⚠️ Data.yaml Issues:")
        for issue in results["data_yaml_issues"]:
            print(f"  - {issue}")
    
    if results["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")
    
    if results["is_segmentation_ready"]:
        print(f"\n✅ Dataset is ready for segmentation training!")
        print("💡 You can now run YOLO segmentation training")
    else:
        print(f"\n⚠️ Dataset may need fixes before segmentation training")
        print("💡 Follow the recommendations above to fix issues")

if __name__ == "__main__":
    main() 