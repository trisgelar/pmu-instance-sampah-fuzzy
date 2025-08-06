#!/usr/bin/env python3
"""
ONNX Conversion Checker (check2)

This module tests ONNX conversion processes and validates the conversion pipeline.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXConversionChecker:
    """
    Checker for ONNX conversion processes and pipeline validation.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            'pytorch_models_exist': False,
            'conversion_environment_ok': False,
            'conversion_process_ok': False,
            'conversion_output_ok': False,
            'conversion_validation_ok': False,
            'conversion_performance_ok': False
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
    
    def check_pytorch_models_exist(self):
        """Check if PyTorch models exist for conversion."""
        print_section("PyTorch Models Check")
        
        runs_dir = self.project_root / "results" / "runs"
        if not runs_dir.exists():
            print(f"❌ Results/runs directory not found: {runs_dir}")
            self.results['pytorch_models_exist'] = False
            return []
        
        pytorch_models = list(runs_dir.glob("*/weights/best.pt"))
        if not pytorch_models:
            print("❌ No PyTorch model files found")
            self.results['pytorch_models_exist'] = False
            return []
        
        print(f"✅ Found {len(pytorch_models)} PyTorch model files:")
        for model in pytorch_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            model_name = model.parent.parent.name
            print(f"   - {model_name}/weights/best.pt: {size_mb:.1f} MB")
        
        self.results['pytorch_models_exist'] = True
        return pytorch_models
    
    def check_conversion_environment(self):
        """Check if the environment supports ONNX conversion."""
        print_section("Conversion Environment Check")
        
        # Check required libraries
        required_libs = {
            'torch': 'PyTorch',
            'torch.onnx': 'PyTorch ONNX support',
            'ultralytics': 'Ultralytics',
            'onnx': 'ONNX',
            'onnxruntime': 'ONNX Runtime'
        }
        
        missing_libs = []
        for lib, name in required_libs.items():
            try:
                if lib == 'torch.onnx':
                    import torch.onnx
                    print(f"✅ {name} available")
                else:
                    __import__(lib)
                    print(f"✅ {name} available")
            except ImportError:
                print(f"❌ {name} not available")
                missing_libs.append(lib)
        
        if missing_libs:
            print(f"\n⚠️ Missing libraries: {', '.join(missing_libs)}")
            print("Install with: pip install torch ultralytics onnx onnxruntime")
            self.results['conversion_environment_ok'] = False
        else:
            print("✅ All required libraries for conversion are available")
            self.results['conversion_environment_ok'] = True
    
    def test_conversion_process(self, pytorch_models):
        """Test the actual conversion process."""
        print_section("Conversion Process Test")
        
        if not pytorch_models:
            print("❌ No PyTorch models to test conversion")
            self.results['conversion_process_ok'] = False
            return False
        
        # Test conversion with the first available model
        test_model = pytorch_models[0]
        print(f"🧪 Testing conversion with: {test_model.parent.parent.name}")
        
        try:
            import torch
            import torch.onnx
            from ultralytics import YOLO
            
            # Load the model
            print("📥 Loading PyTorch model...")
            model = YOLO(str(test_model))
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Test ONNX export
            print("🔄 Testing ONNX export...")
            test_output_path = "test_conversion.onnx"
            
            try:
                torch.onnx.export(
                    model.model,
                    dummy_input,
                    test_output_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['images'],
                    output_names=['output0'],
                    dynamic_axes={'images': {0: 'batch_size'},
                                'output0': {0: 'batch_size'}}
                )
                
                if os.path.exists(test_output_path):
                    # Clean up test file
                    os.remove(test_output_path)
                    print("✅ ONNX conversion process works")
                    self.results['conversion_process_ok'] = True
                    return True
                else:
                    print("❌ ONNX conversion failed - no output file")
                    self.results['conversion_process_ok'] = False
                    return False
                    
            except Exception as e:
                print(f"❌ ONNX conversion failed: {e}")
                self.results['conversion_process_ok'] = False
                return False
                
        except Exception as e:
            print(f"❌ Conversion test failed: {e}")
            self.results['conversion_process_ok'] = False
            return False
    
    def check_conversion_output(self):
        """Check if conversion output directory exists and has files."""
        print_section("Conversion Output Check")
        
        onnx_dir = self.project_root / "results" / "onnx_models"
        if not onnx_dir.exists():
            print(f"❌ ONNX models directory not found: {onnx_dir}")
            self.results['conversion_output_ok'] = False
            return
        
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print("❌ No ONNX files found in output directory")
            self.results['conversion_output_ok'] = False
            return
        
        print(f"✅ Found {len(onnx_files)} ONNX files in output directory:")
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.1f} MB")
        
        # Check file sizes are reasonable
        size_issues = []
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb < 0.1:  # Less than 100KB
                size_issues.append(f"{file.name}: Too small ({size_mb:.1f} MB)")
            elif size_mb > 500:  # More than 500MB
                size_issues.append(f"{file.name}: Too large ({size_mb:.1f} MB)")
        
        if size_issues:
            print("⚠️ Size issues found:")
            for issue in size_issues:
                print(f"   - {issue}")
            self.results['conversion_output_ok'] = False
        else:
            print("✅ All conversion outputs are reasonable")
            self.results['conversion_output_ok'] = True
    
    def validate_conversion_results(self):
        """Validate the converted ONNX models."""
        print_section("Conversion Validation")
        
        try:
            import onnx
            import onnxruntime
            
            onnx_dir = self.project_root / "results" / "onnx_models"
            onnx_files = list(onnx_dir.glob("*.onnx"))
            
            if not onnx_files:
                print("❌ No ONNX files to validate")
                self.results['conversion_validation_ok'] = False
                return
            
            validation_issues = []
            for file in onnx_files:
                try:
                    # Load and validate ONNX model
                    model = onnx.load(str(file))
                    onnx.checker.check_model(model)
                    
                    # Test ONNX Runtime inference
                    session = onnxruntime.InferenceSession(str(file))
                    
                    # Get input info
                    input_info = session.get_inputs()[0]
                    input_shape = input_info.shape
                    input_name = input_info.name
                    
                    # Create dummy input
                    dummy_input = np.random.randn(*input_shape).astype(np.float32)
                    
                    # Run inference
                    outputs = session.run(None, {input_name: dummy_input})
                    
                    print(f"✅ {file.name}: Valid and runnable")
                    
                except Exception as e:
                    print(f"❌ {file.name}: Validation failed - {e}")
                    validation_issues.append(f"{file.name}: {e}")
            
            if validation_issues:
                print("⚠️ Validation issues found:")
                for issue in validation_issues:
                    print(f"   - {issue}")
                self.results['conversion_validation_ok'] = False
            else:
                print("✅ All converted models are valid and runnable")
                self.results['conversion_validation_ok'] = True
                
        except ImportError as e:
            print(f"❌ Validation libraries not available: {e}")
            self.results['conversion_validation_ok'] = False
    
    def check_conversion_performance(self):
        """Check conversion performance and timing."""
        print_section("Conversion Performance Check")
        
        try:
            import time
            import torch
            from ultralytics import YOLO
            
            # Find a PyTorch model to test
            runs_dir = self.project_root / "results" / "runs"
            pytorch_models = list(runs_dir.glob("*/weights/best.pt"))
            
            if not pytorch_models:
                print("❌ No PyTorch models to test performance")
                self.results['conversion_performance_ok'] = False
                return
            
            test_model = pytorch_models[0]
            print(f"⏱️ Testing conversion performance with: {test_model.parent.parent.name}")
            
            # Load model
            start_time = time.time()
            model = YOLO(str(test_model))
            load_time = time.time() - start_time
            print(f"📥 Model load time: {load_time:.2f} seconds")
            
            # Test conversion timing
            dummy_input = torch.randn(1, 3, 640, 640)
            test_output_path = "test_performance.onnx"
            
            start_time = time.time()
            try:
                torch.onnx.export(
                    model.model,
                    dummy_input,
                    test_output_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['images'],
                    output_names=['output0'],
                    dynamic_axes={'images': {0: 'batch_size'},
                                'output0': {0: 'batch_size'}}
                )
                conversion_time = time.time() - start_time
                
                # Clean up
                if os.path.exists(test_output_path):
                    os.remove(test_output_path)
                
                print(f"🔄 Conversion time: {conversion_time:.2f} seconds")
                
                # Performance assessment
                if conversion_time < 60:  # Less than 1 minute
                    print("✅ Conversion performance is good")
                    self.results['conversion_performance_ok'] = True
                elif conversion_time < 300:  # Less than 5 minutes
                    print("⚠️ Conversion performance is acceptable but slow")
                    self.results['conversion_performance_ok'] = True
                else:
                    print("❌ Conversion performance is too slow")
                    self.results['conversion_performance_ok'] = False
                    
            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                self.results['conversion_performance_ok'] = False
                
        except Exception as e:
            print(f"❌ Performance check failed: {e}")
            self.results['conversion_performance_ok'] = False
    
    def run_all_checks(self):
        """Run all ONNX conversion checks."""
        self.print_header("ONNX Conversion Checker (check2)")
        
        # Check PyTorch models exist
        pytorch_models = self.check_pytorch_models_exist()
        
        # Check conversion environment
        self.check_conversion_environment()
        
        # Test conversion process
        conversion_success = self.test_conversion_process(pytorch_models)
        
        # Check conversion output
        self.check_conversion_output()
        
        # Validate conversion results
        self.validate_conversion_results()
        
        # Check conversion performance
        self.check_conversion_performance()
        
        self.print_header("Check Summary")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all checks."""
        print("📊 Conversion Check Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All conversion checks passed! ONNX conversion works properly.")
        else:
            print("⚠️ Some checks failed. Please review the issues above.")

def main():
    """Run ONNX conversion checker."""
    checker = ONNXConversionChecker()
    results = checker.run_all_checks()
    
    print("\n💡 For ONNX conversion, use: python main_colab.py --models v8n --onnx-export")

if __name__ == "__main__":
    main() 