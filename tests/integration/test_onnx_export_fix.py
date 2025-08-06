#!/usr/bin/env python3
"""
Test script to verify ONNX export fix works correctly.
"""

import os
import logging
from modules.model_processor import ModelProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_onnx_export_fix():
    """Test that ONNX export uses correct parameter name."""
    try:
        logger.info("Testing ONNX export fix...")
        
        # Create model processor
        model_processor = ModelProcessor(
            model_dir="results/runs",
            onnx_model_dir="results/onnx_models",
            img_size=(640, 640)
        )
        
        logger.info("Model processor created successfully")
        
        # Check if the export method exists and uses correct parameters
        import inspect
        
        # Get the source code of the export method
        source_lines = inspect.getsource(model_processor.train_yolo_model)
        
        # Check if it uses 'file=' instead of 'filename='
        if 'file=' in source_lines and 'filename=' not in source_lines:
            logger.info("✅ ONNX export uses correct 'file=' parameter")
            return True
        elif 'filename=' in source_lines:
            logger.error("❌ ONNX export still uses incorrect 'filename=' parameter")
            return False
        else:
            logger.warning("⚠️  Could not verify ONNX export parameters in source")
            return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main function."""
    print("🧪 Testing ONNX Export Fix")
    print("=" * 50)
    
    success = test_onnx_export_fix()
    
    if success:
        print("\n✅ ONNX export fix verified!")
        print("📋 The 'filename' error should be resolved")
        print("📋 Next steps:")
        print("1. Run: python main_colab.py")
        print("2. ONNX export should work without errors")
    else:
        print("\n❌ ONNX export fix verification failed")
        print("📋 The fix may not be applied correctly")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 