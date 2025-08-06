#!/usr/bin/env python3
"""
Test script to verify the configuration fix for img_size tuple conversion.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config_manager import ConfigManager

def test_config_fix():
    """Test that the configuration properly converts lists to tuples."""
    print("🔍 Testing Configuration Fix")
    print("=" * 50)
    
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        
        # Get model config
        model_config = config_manager.get_model_config()
        
        print(f"✅ Config loaded successfully")
        print(f"📋 default_img_size: {model_config.default_img_size} (type: {type(model_config.default_img_size)})")
        print(f"📋 supported_img_sizes: {model_config.supported_img_sizes}")
        
        # Check if default_img_size is a tuple
        if isinstance(model_config.default_img_size, tuple):
            print("✅ default_img_size is correctly a tuple")
        else:
            print("❌ default_img_size is not a tuple")
            return False
        
        # Check if supported_img_sizes contains tuples
        all_tuples = all(isinstance(item, tuple) for item in model_config.supported_img_sizes)
        if all_tuples:
            print("✅ supported_img_sizes contains only tuples")
        else:
            print("❌ supported_img_sizes contains non-tuple items")
            return False
        
        # Test ModelProcessor initialization
        from modules.model_processor import ModelProcessor
        
        try:
            model_processor = ModelProcessor(
                model_dir="test_model_dir",
                onnx_model_dir="test_onnx_dir",
                img_size=model_config.default_img_size
            )
            print("✅ ModelProcessor initialized successfully with config img_size")
        except Exception as e:
            print(f"❌ ModelProcessor initialization failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Configuration fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_config_fix()
    sys.exit(0 if success else 1) 