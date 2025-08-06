#!/usr/bin/env python3
"""
ONNX Inference Checker (check3)

This module tests ONNX model inference and validates inference performance.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXInferenceChecker:
    """
    Checker for ONNX model inference and performance validation.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            'onnx_models_exist': False,
            'inference_environment_ok': False,
            'inference_process_ok': False,
            'inference_accuracy_ok': False,
            'inference_performance_ok': False,
            'inference_memory_ok': False
        }
    
    def print_header(self, title):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_section(self, title):
        """Print a formatted section."""
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def check_onnx_models_exist(self):
        """Check if ONNX models exist for inference testing."""
        print_section("ONNX Models Check")
        
        onnx_dir = self.project_root / "results" / "onnx_models"
        if not onnx_dir.exists():
            print(f"❌ ONNX models directory not found: {onnx_dir}")
            self.results['onnx_models_exist'] = False
            return []
        
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print("❌ No ONNX model files found")
            self.results['onnx_models_exist'] = False
            return []
        
        print(f"✅ Found {len(onnx_files)} ONNX model files:")
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.1f} MB")
        
        self.results['onnx_models_exist'] = True
        return onnx_files
    
    def check_inference_environment(self):
        """Check if the environment supports ONNX inference."""
        print_section("Inference Environment Check")
        
        # Check required libraries
        required_libs = {
            'onnx': 'ONNX',
            'onnxruntime': 'ONNX Runtime',
            'numpy': 'NumPy',
            'cv2': 'OpenCV'
        }
        
        missing_libs = []
        for lib, name in required_libs.items():
            try:
                if lib == 'cv2':
                    import cv2
                    print(f"✅ {name} available (version {cv2.__version__})")
                else:
                    module = __import__(lib)
                    version = getattr(module, '__version__', 'unknown')
                    print(f"✅ {name} available (version {version})")
            except ImportError:
                print(f"❌ {name} not available")
                missing_libs.append(lib)
        
        if missing_libs:
            print(f"\n⚠️ Missing libraries: {', '.join(missing_libs)}")
            print("Install with: pip install onnx onnxruntime numpy opencv-python")
            self.results['inference_environment_ok'] = False
        else:
            print("✅ All required libraries for inference are available")
            self.results['inference_environment_ok'] = True
    
    def test_inference_process(self, onnx_files):
        """Test the actual inference process."""
        print_section("Inference Process Test")
        
        if not onnx_files:
            print("❌ No ONNX models to test inference")
            self.results['inference_process_ok'] = False
            return False
        
        # Test inference with the first available model
        test_model = onnx_files[0]
        print(f"🧪 Testing inference with: {test_model.name}")
        
        try:
            import onnxruntime
            import numpy as np
            
            # Load the model
            print("📥 Loading ONNX model...")
            session = onnxruntime.InferenceSession(str(test_model))
            
            # Get input info
            input_info = session.get_inputs()[0]
            input_shape = input_info.shape
            input_name = input_info.name
            
            print(f"📥 Input shape: {input_shape}")
            print(f"📥 Input name: {input_name}")
            
            # Create dummy input
            if len(input_shape) == 4:
                # Handle dynamic batch size
                if input_shape[0] == -1 or input_shape[0] == 'batch_size':
                    batch_size = 1
                else:
                    batch_size = input_shape[0]
                
                dummy_input = np.random.randn(batch_size, input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
            else:
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            print(f"📥 Created dummy input with shape: {dummy_input.shape}")
            
            # Test inference
            print("🔄 Running inference...")
            try:
                outputs = session.run(None, {input_name: dummy_input})
                
                print(f"✅ Inference successful!")
                print(f"📤 Output shape: {outputs[0].shape}")
                print(f"📤 Output type: {outputs[0].dtype}")
                
                self.results['inference_process_ok'] = True
                return True
                
            except Exception as e:
                print(f"❌ Inference failed: {e}")
                self.results['inference_process_ok'] = False
                return False
                
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            self.results['inference_process_ok'] = False
            return False
    
    def test_inference_accuracy(self, onnx_files):
        """Test inference accuracy with known inputs."""
        print_section("Inference Accuracy Test")
        
        if not onnx_files:
            print("❌ No ONNX models to test accuracy")
            self.results['inference_accuracy_ok'] = False
            return False
        
        try:
            import onnxruntime
            import numpy as np
            
            # Test with a simple, predictable input
            test_model = onnx_files[0]
            print(f"🧪 Testing accuracy with: {test_model.name}")
            
            session = onnxruntime.InferenceSession(str(test_model))
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            
            # Create a simple test input (all zeros)
            if len(input_info.shape) == 4:
                if input_info.shape[0] == -1 or input_info.shape[0] == 'batch_size':
                    batch_size = 1
                else:
                    batch_size = input_info.shape[0]
                
                test_input = np.zeros((batch_size, input_info.shape[1], input_info.shape[2], input_info.shape[3]), dtype=np.float32)
            else:
                test_input = np.zeros(input_info.shape, dtype=np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: test_input})
            
            # Check output properties
            output = outputs[0]
            
            # Basic sanity checks
            if np.isfinite(output).all():
                print("✅ Output contains finite values")
            else:
                print("❌ Output contains non-finite values (NaN or Inf)")
                self.results['inference_accuracy_ok'] = False
                return False
            
            if output.shape[0] > 0:
                print("✅ Output has valid batch size")
            else:
                print("❌ Output has invalid batch size")
                self.results['inference_accuracy_ok'] = False
                return False
            
            print(f"✅ Inference accuracy test passed")
            print(f"📊 Output shape: {output.shape}")
            print(f"📊 Output range: [{output.min():.6f}, {output.max():.6f}]")
            
            self.results['inference_accuracy_ok'] = True
            return True
            
        except Exception as e:
            print(f"❌ Accuracy test failed: {e}")
            self.results['inference_accuracy_ok'] = False
            return False
    
    def test_inference_performance(self, onnx_files):
        """Test inference performance and timing."""
        print_section("Inference Performance Test")
        
        if not onnx_files:
            print("❌ No ONNX models to test performance")
            self.results['inference_performance_ok'] = False
            return False
        
        try:
            import time
            import onnxruntime
            import numpy as np
            
            test_model = onnx_files[0]
            print(f"⏱️ Testing performance with: {test_model.name}")
            
            # Load model
            start_time = time.time()
            session = onnxruntime.InferenceSession(str(test_model))
            load_time = time.time() - start_time
            print(f"📥 Model load time: {load_time:.3f} seconds")
            
            # Prepare input
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            
            if len(input_info.shape) == 4:
                if input_info.shape[0] == -1 or input_info.shape[0] == 'batch_size':
                    batch_size = 1
                else:
                    batch_size = input_info.shape[0]
                
                test_input = np.random.randn(batch_size, input_info.shape[1], input_info.shape[2], input_info.shape[3]).astype(np.float32)
            else:
                test_input = np.random.randn(*input_info.shape).astype(np.float32)
            
            # Warm up
            print("🔥 Warming up...")
            for _ in range(3):
                session.run(None, {input_name: test_input})
            
            # Performance test
            print("⚡ Running performance test...")
            num_runs = 10
            inference_times = []
            
            for i in range(num_runs):
                start_time = time.time()
                session.run(None, {input_name: test_input})
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                print(f"   Run {i+1}: {inference_time:.3f}s")
            
            # Calculate statistics
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            
            print(f"\n📊 Performance Statistics:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Std Dev: {std_time:.3f}s")
            print(f"   Min: {min_time:.3f}s")
            print(f"   Max: {max_time:.3f}s")
            
            # Performance assessment
            if avg_time < 0.1:  # Less than 100ms
                print("✅ Inference performance is excellent")
                self.results['inference_performance_ok'] = True
            elif avg_time < 1.0:  # Less than 1 second
                print("✅ Inference performance is good")
                self.results['inference_performance_ok'] = True
            elif avg_time < 5.0:  # Less than 5 seconds
                print("⚠️ Inference performance is acceptable but slow")
                self.results['inference_performance_ok'] = True
            else:
                print("❌ Inference performance is too slow")
                self.results['inference_performance_ok'] = False
                
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.results['inference_performance_ok'] = False
    
    def test_inference_memory(self, onnx_files):
        """Test memory usage during inference."""
        print_section("Inference Memory Test")
        
        if not onnx_files:
            print("❌ No ONNX models to test memory usage")
            self.results['inference_memory_ok'] = False
            return False
        
        try:
            import psutil
            import onnxruntime
            import numpy as np
            
            test_model = onnx_files[0]
            print(f"💾 Testing memory usage with: {test_model.name}")
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
            
            # Load model
            session = onnxruntime.InferenceSession(str(test_model))
            load_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Memory after model load: {load_memory:.1f} MB")
            print(f"📊 Model memory usage: {load_memory - initial_memory:.1f} MB")
            
            # Prepare input
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            
            if len(input_info.shape) == 4:
                if input_info.shape[0] == -1 or input_info.shape[0] == 'batch_size':
                    batch_size = 1
                else:
                    batch_size = input_info.shape[0]
                
                test_input = np.random.randn(batch_size, input_info.shape[1], input_info.shape[2], input_info.shape[3]).astype(np.float32)
            else:
                test_input = np.random.randn(*input_info.shape).astype(np.float32)
            
            # Run inference multiple times
            print("🔄 Running inference to test memory...")
            for i in range(5):
                session.run(None, {input_name: test_input})
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                print(f"   Run {i+1}: {current_memory:.1f} MB")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            total_increase = final_memory - initial_memory
            
            print(f"\n📊 Memory Analysis:")
            print(f"   Total memory increase: {total_increase:.1f} MB")
            
            # Memory assessment
            if total_increase < 100:  # Less than 100MB
                print("✅ Memory usage is excellent")
                self.results['inference_memory_ok'] = True
            elif total_increase < 500:  # Less than 500MB
                print("✅ Memory usage is good")
                self.results['inference_memory_ok'] = True
            elif total_increase < 1000:  # Less than 1GB
                print("⚠️ Memory usage is acceptable but high")
                self.results['inference_memory_ok'] = True
            else:
                print("❌ Memory usage is too high")
                self.results['inference_memory_ok'] = False
                
        except ImportError:
            print("ℹ️ Install psutil for memory info: pip install psutil")
            self.results['inference_memory_ok'] = False
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            self.results['inference_memory_ok'] = False
    
    def run_all_checks(self):
        """Run all ONNX inference checks."""
        self.print_header("ONNX Inference Checker (check3)")
        
        # Check ONNX models exist
        onnx_files = self.check_onnx_models_exist()
        
        # Check inference environment
        self.check_inference_environment()
        
        # Test inference process
        inference_success = self.test_inference_process(onnx_files)
        
        # Test inference accuracy
        accuracy_success = self.test_inference_accuracy(onnx_files)
        
        # Test inference performance
        self.test_inference_performance(onnx_files)
        
        # Test inference memory
        self.test_inference_memory(onnx_files)
        
        self.print_header("Check Summary")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all checks."""
        print("📊 Inference Check Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All inference checks passed! ONNX inference works properly.")
        else:
            print("⚠️ Some checks failed. Please review the issues above.")

def main():
    """Run ONNX inference checker."""
    checker = ONNXInferenceChecker()
    results = checker.run_all_checks()
    
    print("\n💡 For ONNX inference testing, ensure models are created first:")
    print("   python main_colab.py --models v8n --onnx-export")

if __name__ == "__main__":
    main() 