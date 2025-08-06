#!/usr/bin/env python3
"""
Dataset Diagnostic Script

This script helps diagnose issues with your dataset, particularly
the class configuration problems you're experiencing.

Usage:
    python -m tests.dataset_tools.diagnose_dataset
    # or
    from tests.dataset_tools import diagnose_dataset
    diagnose_dataset()
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_data_yaml(dataset_path: str) -> dict:
    """
    Diagnose data.yaml configuration.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        dict: Diagnostic information
    """
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    result = {
        'exists': False,
        'path': data_yaml_path,
        'content': None,
        'classes': None,
        'issues': []
    }
    
    if not os.path.exists(data_yaml_path):
        result['issues'].append("data.yaml file not found")
        return result
    
    result['exists'] = True
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        result['content'] = data
        result['classes'] = data.get('names', {})
        
        # Check for issues
        if 'names' not in data:
            result['issues'].append("No 'names' field found")
        elif len(data['names']) > 1:
            result['issues'].append(f"Multiple classes found: {data['names']} - should only be 'sampah'")
        elif len(data['names']) == 0:
            result['issues'].append("No classes defined")
        elif 0 not in data['names'] or data['names'][0] != 'sampah':
            result['issues'].append(f"Wrong class configuration: {data['names']} - should be {{0: 'sampah'}}")
            
    except Exception as e:
        result['issues'].append(f"Error reading data.yaml: {str(e)}")
    
    return result

def diagnose_coco_files(dataset_path: str) -> dict:
    """
    Diagnose COCO annotation files.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        dict: Diagnostic information
    """
    result = {
        'splits': {},
        'total_annotations': 0,
        'categories_found': set(),
        'issues': []
    }
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        coco_file = os.path.join(split_path, "_annotations.coco.json")
        
        split_info = {
            'exists': False,
            'path': coco_file,
            'annotations_count': 0,
            'categories': [],
            'issues': []
        }
        
        if not os.path.exists(coco_file):
            split_info['issues'].append("COCO file not found")
            result['splits'][split] = split_info
            continue
        
        split_info['exists'] = True
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count annotations
            annotations = data.get('annotations', [])
            split_info['annotations_count'] = len(annotations)
            result['total_annotations'] += len(annotations)
            
            # Analyze categories
            categories = data.get('categories', [])
            split_info['categories'] = categories
            
            for cat in categories:
                result['categories_found'].add(cat.get('name', 'unknown'))
            
            # Check for issues
            if len(categories) > 1:
                split_info['issues'].append(f"Multiple categories found: {[c.get('name') for c in categories]}")
                result['issues'].append(f"{split}: Multiple categories - should only be 'sampah'")
            
            if not any(cat.get('name') == 'sampah' for cat in categories):
                split_info['issues'].append("No 'sampah' category found")
                result['issues'].append(f"{split}: No 'sampah' category found")
            
            # Check annotation category distribution
            category_counts = {}
            for ann in annotations:
                cat_id = ann.get('category_id')
                for cat in categories:
                    if cat.get('id') == cat_id:
                        cat_name = cat.get('name', 'unknown')
                        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                        break
            
            if len(category_counts) > 1:
                split_info['issues'].append(f"Annotations use multiple categories: {category_counts}")
                result['issues'].append(f"{split}: Annotations use multiple categories")
            
        except Exception as e:
            split_info['issues'].append(f"Error reading COCO file: {str(e)}")
            result['issues'].append(f"{split}: Error reading file")
        
        result['splits'][split] = split_info
    
    return result

def diagnose_roboflow_project() -> dict:
    """
    Diagnose Roboflow project configuration.
    
    Returns:
        dict: Diagnostic information
    """
    result = {
        'api_key_exists': False,
        'project_info': None,
        'classes': [],
        'issues': []
    }
    
    # Check if secrets.yaml exists
    secrets_path = "secrets.yaml"
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                secrets = yaml.safe_load(f)
            
            if 'roboflow_api_key' in secrets:
                result['api_key_exists'] = True
            else:
                result['issues'].append("No 'roboflow_api_key' in secrets.yaml")
        except Exception as e:
            result['issues'].append(f"Error reading secrets.yaml: {str(e)}")
    else:
        result['issues'].append("secrets.yaml not found")
    
    # Try to get project info if API key exists
    if result['api_key_exists']:
        try:
            from roboflow import Roboflow
            with open(secrets_path, 'r') as f:
                secrets = yaml.safe_load(f)
            
            rf = Roboflow(api_key=secrets['roboflow_api_key'])
            project = rf.workspace().project("abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation")
            
            result['project_info'] = {
                'name': project.name,
                'version': project.version,
                'type': project.type
            }
            
            result['classes'] = list(project.classes.keys())
            
            if len(result['classes']) > 1:
                result['issues'].append(f"Roboflow project has multiple classes: {result['classes']}")
            elif 'sampah' not in result['classes']:
                result['issues'].append(f"Roboflow project doesn't have 'sampah' class: {result['classes']}")
                
        except Exception as e:
            result['issues'].append(f"Error accessing Roboflow: {str(e)}")
    
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

def diagnose_dataset(dataset_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Main diagnostic function.
    
    Args:
        dataset_path: Path to dataset directory (auto-detected if None)
        verbose: Whether to print diagnostic information
        
    Returns:
        Dict[str, Any]: Complete diagnostic results
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()
    
    if dataset_path is None:
        error_msg = "❌ Dataset path not found. Please run this script from the project root directory."
        if verbose:
            print(error_msg)
        return {'error': error_msg}
    
    if verbose:
        print("🔍 Dataset Diagnostic Script")
        print("=" * 50)
        print(f"📁 Dataset path: {dataset_path}")
        print()
    
    # Diagnose data.yaml
    if verbose:
        print("📄 Diagnosing data.yaml...")
    data_yaml_diag = diagnose_data_yaml(dataset_path)
    
    if verbose:
        if data_yaml_diag['exists']:
            print(f"  ✅ data.yaml found")
            print(f"  📋 Classes: {data_yaml_diag['classes']}")
            if data_yaml_diag['issues']:
                print(f"  ⚠️  Issues: {data_yaml_diag['issues']}")
            else:
                print(f"  ✅ No issues found")
        else:
            print(f"  ❌ data.yaml not found")
            print(f"  ⚠️  Issues: {data_yaml_diag['issues']}")
        print()
    
    # Diagnose COCO files
    if verbose:
        print("📋 Diagnosing COCO annotation files...")
    coco_diag = diagnose_coco_files(dataset_path)
    
    if verbose:
        for split, info in coco_diag['splits'].items():
            if info['exists']:
                print(f"  ✅ {split}: {info['annotations_count']} annotations, {len(info['categories'])} categories")
                if info['issues']:
                    print(f"     ⚠️  Issues: {info['issues']}")
            else:
                print(f"  ❌ {split}: COCO file not found")
        
        if coco_diag['categories_found']:
            print(f"  📊 Total categories found: {list(coco_diag['categories_found'])}")
        
        if coco_diag['issues']:
            print(f"  ⚠️  COCO Issues: {coco_diag['issues']}")
        print()
    
    # Diagnose Roboflow project
    if verbose:
        print("🌐 Diagnosing Roboflow project...")
    roboflow_diag = diagnose_roboflow_project()
    
    if verbose:
        if roboflow_diag['api_key_exists']:
            print(f"  ✅ API key found")
            if roboflow_diag['project_info']:
                print(f"  📋 Project: {roboflow_diag['project_info']['name']} v{roboflow_diag['project_info']['version']}")
            print(f"  🏷️  Classes: {roboflow_diag['classes']}")
        else:
            print(f"  ❌ API key not found")
        
        if roboflow_diag['issues']:
            print(f"  ⚠️  Issues: {roboflow_diag['issues']}")
        print()
    
    # Summary and recommendations
    if verbose:
        print("📋 Summary and Recommendations")
        print("=" * 50)
    
    all_issues = []
    all_issues.extend(data_yaml_diag['issues'])
    all_issues.extend(coco_diag['issues'])
    all_issues.extend(roboflow_diag['issues'])
    
    results = {
        'dataset_path': dataset_path,
        'data_yaml': data_yaml_diag,
        'coco_files': coco_diag,
        'roboflow': roboflow_diag,
        'all_issues': all_issues,
        'has_issues': len(all_issues) > 0
    }
    
    if verbose:
        if all_issues:
            print("❌ Issues found:")
            for issue in all_issues:
                print(f"  - {issue}")
            
            print("\n🔧 Recommended fixes:")
            print("  1. Run: python -m tests.dataset_tools.fix_dataset_ultralytics")
            print("  2. Delete existing training runs: rm -rf runs/")
            print("  3. Retrain your model")
        else:
            print("✅ No issues found! Your dataset is correctly configured.")
        
        print("\n📊 Current state:")
        print(f"  - Total annotations: {coco_diag['total_annotations']}")
        print(f"  - Categories found: {list(coco_diag['categories_found'])}")
        print(f"  - Roboflow classes: {roboflow_diag['classes']}")
    
    return results

def main():
    """Main function for command-line usage."""
    diagnose_dataset()

if __name__ == "__main__":
    main() 