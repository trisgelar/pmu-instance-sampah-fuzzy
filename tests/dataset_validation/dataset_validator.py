#!/usr/bin/env python3
"""
Dataset Validator Script

This script provides comprehensive validation of dataset structure and format,
ensuring compatibility with YOLO training.

Usage:
    python -m tests.dataset_tools.dataset_validator
    # or
    from tests.dataset_tools import validate_dataset
    validate_dataset()
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_directory_structure(dataset_path: str) -> Dict[str, Any]:
    """
    Validate dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'structure': {}
    }
    
    # Check main dataset directory
    if not os.path.exists(dataset_path):
        result['valid'] = False
        result['issues'].append(f"Dataset directory does not exist: {dataset_path}")
        return result
    
    # Check required splits
    required_splits = ['train', 'valid', 'test']
    for split in required_splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            result['structure'][split] = {
                'exists': True,
                'path': split_path,
                'files': os.listdir(split_path) if os.path.exists(split_path) else []
            }
        else:
            result['structure'][split] = {
                'exists': False,
                'path': split_path,
                'files': []
            }
            result['issues'].append(f"Missing required split: {split}")
            result['valid'] = False
    
    # Check data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        result['structure']['data_yaml'] = {
            'exists': True,
            'path': data_yaml_path
        }
    else:
        result['structure']['data_yaml'] = {
            'exists': False,
            'path': data_yaml_path
        }
        result['issues'].append("Missing data.yaml file")
        result['valid'] = False
    
    return result

def validate_data_yaml(dataset_path: str) -> Dict[str, Any]:
    """
    Validate data.yaml configuration.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'content': None
    }
    
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        result['valid'] = False
        result['issues'].append("data.yaml file not found")
        return result
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        result['content'] = data
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'names']
        for field in required_fields:
            if field not in data:
                result['issues'].append(f"Missing required field: {field}")
                result['valid'] = False
        
        # Check path field
        if 'path' in data:
            if not os.path.isabs(data['path']):
                # Convert relative path to absolute
                data['path'] = os.path.abspath(os.path.join(dataset_path, data['path']))
            
            if not os.path.exists(data['path']):
                result['issues'].append(f"Path does not exist: {data['path']}")
                result['valid'] = False
        
        # Check class names
        if 'names' in data:
            if not isinstance(data['names'], dict):
                result['issues'].append("'names' field must be a dictionary")
                result['valid'] = False
            elif len(data['names']) == 0:
                result['issues'].append("No classes defined in 'names' field")
                result['valid'] = False
            elif 0 not in data['names']:
                result['issues'].append("Class ID 0 not found in 'names' field")
                result['valid'] = False
            elif data['names'][0] != 'sampah':
                result['issues'].append(f"Class 0 should be 'sampah', found: {data['names'][0]}")
                result['valid'] = False
        
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Error reading data.yaml: {str(e)}")
    
    return result

def validate_coco_annotations(dataset_path: str) -> Dict[str, Any]:
    """
    Validate COCO annotation files.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'splits': {}
    }
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        split_result = {
            'valid': True,
            'issues': [],
            'annotations_count': 0,
            'categories': [],
            'images_count': 0
        }
        
        if not os.path.exists(coco_file):
            split_result['valid'] = False
            split_result['issues'].append("COCO file not found")
            result['splits'][split] = split_result
            result['valid'] = False
            continue
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required COCO fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in data:
                    split_result['issues'].append(f"Missing required field: {field}")
                    split_result['valid'] = False
            
            # Count annotations and images
            split_result['annotations_count'] = len(data.get('annotations', []))
            split_result['images_count'] = len(data.get('images', []))
            split_result['categories'] = data.get('categories', [])
            
            # Validate categories
            if len(split_result['categories']) == 0:
                split_result['issues'].append("No categories defined")
                split_result['valid'] = False
            elif len(split_result['categories']) > 1:
                split_result['issues'].append(f"Multiple categories found: {[c.get('name') for c in split_result['categories']]}")
                split_result['valid'] = False
            elif split_result['categories'][0].get('name') != 'sampah':
                split_result['issues'].append(f"Category should be 'sampah', found: {split_result['categories'][0].get('name')}")
                split_result['valid'] = False
            
            # Check annotation format
            for i, ann in enumerate(data.get('annotations', [])):
                if 'category_id' not in ann:
                    split_result['issues'].append(f"Annotation {i} missing category_id")
                    split_result['valid'] = False
                elif 'segmentation' not in ann:
                    split_result['issues'].append(f"Annotation {i} missing segmentation")
                    split_result['valid'] = False
                elif 'bbox' not in ann:
                    split_result['issues'].append(f"Annotation {i} missing bbox")
                    split_result['valid'] = False
            
            if not split_result['valid']:
                result['valid'] = False
            
        except Exception as e:
            split_result['valid'] = False
            split_result['issues'].append(f"Error reading COCO file: {str(e)}")
            result['valid'] = False
        
        result['splits'][split] = split_result
    
    return result

def validate_image_files(dataset_path: str) -> Dict[str, Any]:
    """
    Validate image files in dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'splits': {}
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        
        split_result = {
            'valid': True,
            'issues': [],
            'image_files': [],
            'image_count': 0
        }
        
        if not os.path.exists(split_path):
            split_result['valid'] = False
            split_result['issues'].append("Split directory not found")
            result['splits'][split] = split_result
            result['valid'] = False
            continue
        
        # Find image files
        image_files = []
        for file in os.listdir(split_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)
        
        split_result['image_files'] = image_files
        split_result['image_count'] = len(image_files)
        
        if len(image_files) == 0:
            split_result['issues'].append("No image files found")
            split_result['valid'] = False
        
        # Check if image files are readable
        for img_file in image_files[:5]:  # Check first 5 images
            img_path = os.path.join(split_path, img_file)
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                split_result['issues'].append(f"Invalid image file: {img_file} - {str(e)}")
                split_result['valid'] = False
        
        if not split_result['valid']:
            result['valid'] = False
        
        result['splits'][split] = split_result
    
    return result

def find_dataset_path() -> Optional[str]:
    """
    Find the dataset path automatically.
    
    Returns:
        Optional[str]: Path to dataset or None if not found
    """
    possible_paths = [
        "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
        "datasets",
        "datasets/train",
        "datasets/valid",
        "datasets/test"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if os.path.exists(os.path.join(path, "data.yaml")) or \
               any(os.path.exists(os.path.join(path, split, "_annotations.coco.json")) 
                  for split in ['train', 'valid', 'test']):
                return path
    
    return None

def validate_dataset(dataset_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Main validation function.
    
    Args:
        dataset_path: Path to dataset directory (auto-detected if None)
        verbose: Whether to print validation information
        
    Returns:
        Dict[str, Any]: Complete validation results
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()
    
    if dataset_path is None:
        error_msg = "❌ Dataset path not found. Please run this script from the project root directory."
        if verbose:
            print(error_msg)
        return {'error': error_msg}
    
    if verbose:
        print("🔍 Dataset Validation Script")
        print("=" * 50)
        print(f"📁 Dataset path: {dataset_path}")
        print()
    
    # Run all validations
    structure_validation = validate_directory_structure(dataset_path)
    yaml_validation = validate_data_yaml(dataset_path)
    coco_validation = validate_coco_annotations(dataset_path)
    image_validation = validate_image_files(dataset_path)
    
    # Combine results
    all_valid = (
        structure_validation['valid'] and
        yaml_validation['valid'] and
        coco_validation['valid'] and
        image_validation['valid']
    )
    
    all_issues = []
    all_issues.extend(structure_validation['issues'])
    all_issues.extend(yaml_validation['issues'])
    all_issues.extend(coco_validation['issues'])
    all_issues.extend(image_validation['issues'])
    
    results = {
        'dataset_path': dataset_path,
        'valid': all_valid,
        'structure_validation': structure_validation,
        'yaml_validation': yaml_validation,
        'coco_validation': coco_validation,
        'image_validation': image_validation,
        'all_issues': all_issues,
        'issue_count': len(all_issues)
    }
    
    if verbose:
        # Print structure validation results
        print("📁 Directory Structure Validation")
        print("-" * 30)
        if structure_validation['valid']:
            print("✅ Directory structure is valid")
        else:
            print("❌ Directory structure issues found:")
            for issue in structure_validation['issues']:
                print(f"  - {issue}")
        print()
        
        # Print YAML validation results
        print("📄 Data.yaml Validation")
        print("-" * 20)
        if yaml_validation['valid']:
            print("✅ data.yaml is valid")
            if yaml_validation['content']:
                print(f"  Classes: {yaml_validation['content'].get('names', {})}")
        else:
            print("❌ data.yaml issues found:")
            for issue in yaml_validation['issues']:
                print(f"  - {issue}")
        print()
        
        # Print COCO validation results
        print("📋 COCO Annotations Validation")
        print("-" * 30)
        if coco_validation['valid']:
            print("✅ COCO annotations are valid")
            for split, info in coco_validation['splits'].items():
                if info['valid']:
                    print(f"  ✅ {split}: {info['annotations_count']} annotations, {info['images_count']} images")
        else:
            print("❌ COCO annotation issues found:")
            for split, info in coco_validation['splits'].items():
                if not info['valid']:
                    print(f"  {split}:")
                    for issue in info['issues']:
                        print(f"    - {issue}")
        print()
        
        # Print image validation results
        print("🖼️ Image Files Validation")
        print("-" * 25)
        if image_validation['valid']:
            print("✅ Image files are valid")
            for split, info in image_validation['splits'].items():
                if info['valid']:
                    print(f"  ✅ {split}: {info['image_count']} images")
        else:
            print("❌ Image file issues found:")
            for split, info in image_validation['splits'].items():
                if not info['valid']:
                    print(f"  {split}:")
                    for issue in info['issues']:
                        print(f"    - {issue}")
        print()
        
        # Print summary
        print("📋 Validation Summary")
        print("=" * 20)
        if all_valid:
            print("🎉 All validations passed! Your dataset is ready for training.")
        else:
            print(f"❌ {len(all_issues)} issues found:")
            for issue in all_issues:
                print(f"  - {issue}")
            print("\n🔧 Recommended fixes:")
            print("  1. Run: python -m tests.dataset_tools.fix_dataset_ultralytics")
            print("  2. Or run: python -m tests.dataset_tools.fix_dataset_classes")
    
    return results

def main():
    """Main function for command-line usage."""
    validate_dataset()

if __name__ == "__main__":
    main() 