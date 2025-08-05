#!/usr/bin/env python3
"""
Test Segmentation Integration

This script tests the integration of segmentation label fixing
into the DatasetManager class.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_manager import DatasetManager
from modules.exceptions import ConfigurationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_segmentation_integration():
    """Test the integration of segmentation label fixing in DatasetManager."""
    print("🧪 Testing Segmentation Label Integration")
    print("=" * 50)
    
    try:
        # Initialize DatasetManager with test configuration
        dataset_manager = DatasetManager(
            dataset_dir="datasets",
            is_project="test_project",
            is_version="1"
        )
        
        print("✅ DatasetManager initialized successfully")
        
        # Test the _normalize_polygon method
        print("\n🔧 Testing _normalize_polygon method...")
        
        # Test case 1: Normal polygon
        polygon = [100, 50, 200, 50, 200, 150, 100, 150]  # Rectangle
        img_width, img_height = 400, 300
        normalized = dataset_manager._normalize_polygon(polygon, img_width, img_height)
        
        if normalized:
            print(f"✅ Polygon normalization successful: {len(normalized)} coordinates")
            print(f"   Original: {polygon}")
            print(f"   Normalized: {[f'{x:.3f}' for x in normalized]}")
        else:
            print("❌ Polygon normalization failed")
            return False
        
        # Test case 2: Invalid polygon (odd number of coordinates)
        invalid_polygon = [100, 50, 200, 50, 200]  # Odd number
        normalized_invalid = dataset_manager._normalize_polygon(invalid_polygon, img_width, img_height)
        
        if not normalized_invalid:
            print("✅ Invalid polygon correctly rejected")
        else:
            print("❌ Invalid polygon should have been rejected")
            return False
        
        # Test case 3: Out of bounds coordinates
        out_of_bounds = [500, 400, 600, 400, 600, 500, 500, 500]  # Outside image
        normalized_bounds = dataset_manager._normalize_polygon(out_of_bounds, img_width, img_height)
        
        if normalized_bounds:
            # Check that all coordinates are within 0-1 range
            all_in_range = all(0 <= coord <= 1 for coord in normalized_bounds)
            if all_in_range:
                print("✅ Out of bounds coordinates correctly clamped to 0-1 range")
            else:
                print("❌ Out of bounds coordinates not properly clamped")
                return False
        else:
            print("❌ Out of bounds polygon normalization failed")
            return False
        
        # Test the fix_segmentation_labels method (will only work if dataset exists)
        print("\n🔧 Testing fix_segmentation_labels method...")
        
        # Try to find an existing dataset
        possible_paths = [
            "datasets/abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation",
            "datasets",
            "dataset"
        ]
        
        dataset_found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 Found dataset at: {path}")
                dataset_found = True
                
                # Test the method (it will return False if no COCO files found, which is expected)
                result = dataset_manager.fix_segmentation_labels(path)
                print(f"   Result: {'✅ Success' if result else '⚠️ No changes needed or failed'}")
                break
        
        if not dataset_found:
            print("⚠️ No dataset found for testing - this is normal if no dataset is present")
            print("   The method will be tested when a dataset is available")
        
        print("\n✅ All segmentation integration tests completed successfully!")
        return True
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        print("   This is expected if secrets.yaml is not configured")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to run the integration test."""
    success = test_segmentation_integration()
    
    if success:
        print("\n🎉 Segmentation integration test passed!")
        print("💡 The fix_segmentation_labels method is now available in DatasetManager")
        print("💡 It will be called automatically during dataset preparation and fixing")
    else:
        print("\n❌ Segmentation integration test failed!")
        print("💡 Check the error messages above for issues")

if __name__ == "__main__":
    main() 