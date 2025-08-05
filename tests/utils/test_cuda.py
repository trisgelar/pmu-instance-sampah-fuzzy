#!/usr/bin/env python3
"""
CUDA Test Script
Tests CUDA installation and functionality for the waste detection system.
"""

import sys
import logging
from modules.cuda_manager import CUDAManager
from modules.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda_installation():
    """Test basic CUDA installation."""
    print("🔍 Testing CUDA Installation...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")
        return False
    
    return True

def test_cuda_manager():
    """Test CUDA manager functionality."""
    print("\n🔍 Testing CUDA Manager...")
    print("=" * 50)
    
    try:
        cuda_manager = CUDAManager()
        print(f"✅ CUDA Manager initialized: {cuda_manager}")
        
        # Test memory info
        memory_info = cuda_manager.get_memory_info()
        print(f"✅ Memory info: {memory_info}")
        
        # Test optimization
        optimization = cuda_manager.optimize_for_training(
            batch_size=16, 
            img_size=(640, 640)
        )
        print(f"✅ Optimization recommendations: {optimization}")
        
        # Test optimal batch size
        optimal_batch = cuda_manager.get_optimal_batch_size((640, 640))
        print(f"✅ Optimal batch size: {optimal_batch}")
        
        # Test memory monitoring
        memory_status = cuda_manager.monitor_memory()
        print(f"✅ Memory status: {memory_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA Manager test failed: {e}")
        return False

def test_system_integration():
    """Test system integration with CUDA."""
    print("\n🔍 Testing System Integration...")
    print("=" * 50)
    
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config()
        
        # Initialize CUDA manager
        cuda_manager = CUDAManager()
        
        # Test with system configuration
        optimization = cuda_manager.optimize_for_training(
            batch_size=model_config.default_batch_size,
            img_size=model_config.default_img_size
        )
        
        print(f"✅ System integration test passed")
        print(f"✅ Device: {optimization['device']}")
        print(f"✅ Batch size: {optimization['batch_size']}")
        print(f"✅ Image size: {optimization['img_size']}")
        print(f"✅ Optimizations: {optimization['optimizations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_memory_management():
    """Test CUDA memory management."""
    print("\n🔍 Testing Memory Management...")
    print("=" * 50)
    
    try:
        cuda_manager = CUDAManager()
        
        # Get initial memory
        initial_memory = cuda_manager.monitor_memory()
        print(f"✅ Initial memory: {initial_memory}")
        
        # Clear cache
        cuda_manager.clear_cache()
        print("✅ Cache cleared")
        
        # Get memory after clearing
        final_memory = cuda_manager.monitor_memory()
        print(f"✅ Final memory: {final_memory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def main():
    """Run all CUDA tests."""
    print("🚀 CUDA Test Suite for Waste Detection System")
    print("=" * 60)
    
    tests = [
        ("CUDA Installation", test_cuda_installation),
        ("CUDA Manager", test_cuda_manager),
        ("System Integration", test_system_integration),
        ("Memory Management", test_memory_management)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            print(f"✅ {test_name} test PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CUDA is properly configured.")
        print("\n📋 Next steps:")
        print("1. Run: python validate_secrets.py")
        print("2. Run: python run_tests.py")
        print("3. Start training: python main_colab.py")
    else:
        print("⚠️  Some tests failed. Please check CUDA installation.")
        print("\n🔧 Troubleshooting:")
        print("1. Run: python install_cuda.py")
        print("2. Check NVIDIA drivers are installed")
        print("3. Verify CUDA toolkit installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 