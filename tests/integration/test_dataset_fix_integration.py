#!/usr/bin/env python3
"""
Test script for integrated dataset fixing functionality in DatasetManager.

This script demonstrates how to use the new dataset fixing methods
that are integrated into the DatasetManager class.

Usage:
    python -m tests.test_dataset_fix_integration
    # or
    python tests/test_dataset_fix_integration.py
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from modules.dataset_manager import DatasetManager
from modules.exceptions import ConfigurationError, DatasetError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        config_path = os.path.join(project_root, 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {str(e)}")
        return {}

def test_dataset_validation():
    """Test dataset validation functionality."""
    print("🔍 Testing Dataset Validation")
    print("=" * 50)
    
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return None
    
    try:
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        # Validate dataset format
        validation_results = dataset_manager.validate_dataset_format()
        
        print(f"📁 Dataset Path: {validation_results['dataset_path']}")
        print(f"📂 Exists: {validation_results['exists']}")
        print(f"📄 data.yaml exists: {validation_results['data_yaml_exists']}")
        
        if validation_results['data_yaml_content']:
            print(f"📋 data.yaml content: {validation_results['data_yaml_content']}")
        
        if validation_results['splits']:
            print("\n📊 Dataset Splits:")
            for split, info in validation_results['splits'].items():
                print(f"  {split}: {info['images_count']} images, {info['labels_count']} labels")
        
        if validation_results['issues']:
            print("\n⚠️ Issues Found:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
        
        if validation_results['recommendations']:
            print("\n💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"  - {rec}")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        return None

def test_dataset_fixing():
    """Test dataset fixing functionality."""
    print("\n🔧 Testing Dataset Fixing")
    print("=" * 50)
    
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    try:
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        # Check if dataset exists
        dataset_path = os.path.join(dataset_manager.DATASET_DIR, dataset_manager.ROBOFLOW_IS_PROJECT)
        
        if not os.path.exists(dataset_path):
            print(f"📁 Dataset not found at: {dataset_path}")
            print("💡 You need to run prepare_datasets() first to download/extract the dataset")
            return False
        
        print(f"📁 Found dataset at: {dataset_path}")
        
        # Fix dataset classes
        print("🔧 Fixing dataset classes...")
        success = dataset_manager.fix_dataset_classes()
        
        if success:
            print("✅ Dataset fixing completed successfully!")
            
            # Validate again after fixing
            print("\n🔍 Re-validating dataset after fixing...")
            validation_results = dataset_manager.validate_dataset_format()
            
            if validation_results['issues']:
                print("⚠️ Remaining issues:")
                for issue in validation_results['issues']:
                    print(f"  - {issue}")
            else:
                print("✅ All issues resolved!")
            
            return True
        else:
            print("❌ Dataset fixing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during fixing: {str(e)}")
        return False

def test_prepare_datasets():
    """Test the complete dataset preparation process."""
    print("\n🚀 Testing Complete Dataset Preparation")
    print("=" * 50)
    
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    try:
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        print("📥 Preparing datasets with integrated Ultralytics normalization...")
        success = dataset_manager.prepare_datasets()
        
        if success:
            print("✅ Dataset preparation completed successfully!")
            print("🎯 Dataset is now in proper YOLO format with only 'sampah' class")
            
            # Validate the prepared dataset
            print("\n🔍 Validating prepared dataset...")
            validation_results = dataset_manager.validate_dataset_format()
            
            if not validation_results['issues']:
                print("✅ Dataset validation passed!")
                print("🚀 Ready for training!")
            else:
                print("⚠️ Validation issues found:")
                for issue in validation_results['issues']:
                    print(f"  - {issue}")
            
            return True
        else:
            print("❌ Dataset preparation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during preparation: {str(e)}")
        return False

def test_individual_methods():
    """Test individual DatasetManager methods."""
    print("\n🧪 Testing Individual Methods")
    print("=" * 50)
    
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    try:
        # Initialize DatasetManager
        dataset_config = config.get('dataset', {})
        dataset_manager = DatasetManager(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            is_project=dataset_config.get('roboflow_project', ''),
            is_version=dataset_config.get('roboflow_version', '1')
        )
        
        # Test 1: Validate dataset format
        print("1. Testing validate_dataset_format()...")
        validation_results = dataset_manager.validate_dataset_format()
        print(f"   ✅ Validation completed: {len(validation_results.get('issues', []))} issues found")
        
        # Test 2: Check if dataset needs fixing
        print("2. Testing fix_dataset_classes()...")
        if validation_results.get('exists', False):
            success = dataset_manager.fix_dataset_classes()
            print(f"   ✅ Fix completed: {'Success' if success else 'Failed'}")
        else:
            print("   ⚠️ Dataset doesn't exist, skipping fix test")
        
        # Test 3: Test backup functionality (if dataset exists)
        dataset_path = os.path.join(dataset_manager.DATASET_DIR, dataset_manager.ROBOFLOW_IS_PROJECT)
        if os.path.exists(dataset_path):
            print("3. Testing backup functionality...")
            backup_path = f"{dataset_path}_backup"
            if os.path.exists(backup_path):
                print(f"   ✅ Backup exists at: {backup_path}")
            else:
                print("   ℹ️ No backup found (normal if no fixing was done)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during individual method testing: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Dataset Fix Integration Test")
    print("=" * 60)
    print("This script tests the integrated dataset fixing functionality")
    print("in the DatasetManager class.")
    print()
    
    # Test 1: Validation
    validation_results = test_dataset_validation()
    
    # Test 2: Individual methods
    test_individual_methods()
    
    # Test 3: Fixing (if dataset exists)
    if validation_results and validation_results['exists']:
        test_dataset_fixing()
    else:
        print("\n💡 Dataset doesn't exist yet. Running complete preparation test...")
        test_prepare_datasets()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("\n📚 Usage Examples:")
    print("  # Initialize DatasetManager")
    print("  dataset_manager = DatasetManager(dataset_dir, project, version)")
    print()
    print("  # Validate dataset format")
    print("  results = dataset_manager.validate_dataset_format()")
    print()
    print("  # Fix dataset classes")
    print("  success = dataset_manager.fix_dataset_classes()")
    print()
    print("  # Prepare datasets (includes fixing)")
    print("  success = dataset_manager.prepare_datasets()")
    print()
    print("📁 Test files location:")
    print(f"  - This test: {os.path.abspath(__file__)}")
    print(f"  - Project root: {project_root}")

if __name__ == "__main__":
    main() 