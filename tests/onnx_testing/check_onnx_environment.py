#!/usr/bin/env python3
"""
ONNX Environment Checker (check0)

This module checks the environment for ONNX dependencies and setup.
It's similar to check_onnx_rknn_environment.py but focused specifically on ONNX.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXEnvironmentChecker:
    """
    Checker for ONNX environment setup and dependencies.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            'python_version': False,
            'pytorch': False,
            'ultralytics': False,
            'onnx': False,
            'onnxruntime': False,
            'opencv': False,
            'numpy': False,
            'memory': False,
            'gpu': False
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
    
    def check_python_version(self):
        """Check Python version compatibility."""
        print_section("Python Version")
        version = sys.version_info
        print(f"Python: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 8:
            print("✅ Python version is compatible for ONNX")
            self.results['python_version'] = True
        else:
            print("❌ Python version should be 3.8+ for ONNX")
            self.results['python_version'] = False
    
    def check_pytorch(self):
        """Check PyTorch installation and ONNX support."""
        print_section("PyTorch Installation")
        
        try:
            import torch
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print("✅ GPU environment detected for ONNX")
                self.results['gpu'] = True
            else:
                print("ℹ️ CPU-only environment detected")
                self.results['gpu'] = False
            
            # Check ONNX export capability
            try:
                import torch.onnx
                print("✅ PyTorch ONNX export support available")
                self.results['pytorch'] = True
            except ImportError:
                print("❌ PyTorch ONNX export not available")
                self.results['pytorch'] = False
                
        except ImportError:
            print("❌ PyTorch not installed")
            print("Install with: pip install torch torchvision torchaudio")
            self.results['pytorch'] = False
    
    def check_ultralytics(self):
        """Check Ultralytics installation."""
        print_section("Ultralytics Installation")
        
        try:
            import ultralytics
            print(f"Ultralytics Version: {ultralytics.__version__}")
            print("✅ Ultralytics is installed")
            self.results['ultralytics'] = True
        except ImportError:
            print("❌ Ultralytics not installed")
            print("Install with: pip install ultralytics")
            self.results['ultralytics'] = False
    
    def check_onnx(self):
        """Check ONNX installation."""
        print_section("ONNX Installation")
        
        try:
            import onnx
            print(f"ONNX Version: {onnx.__version__}")
            print("✅ ONNX is installed")
            self.results['onnx'] = True
        except ImportError:
            print("❌ ONNX not installed")
            print("Install with: pip install onnx")
            self.results['onnx'] = False
        
        try:
            import onnxruntime
            print(f"ONNX Runtime Version: {onnxruntime.__version__}")
            print("✅ ONNX Runtime is installed")
            self.results['onnxruntime'] = True
        except ImportError:
            print("❌ ONNX Runtime not installed")
            print("Install with: pip install onnxruntime")
            self.results['onnxruntime'] = False
    
    def check_opencv(self):
        """Check OpenCV installation."""
        print_section("OpenCV Installation")
        
        try:
            import cv2
            print(f"OpenCV Version: {cv2.__version__}")
            print("✅ OpenCV is installed")
            self.results['opencv'] = True
        except ImportError:
            print("❌ OpenCV not installed")
            print("Install with: pip install opencv-python")
            self.results['opencv'] = False
    
    def check_numpy(self):
        """Check NumPy installation."""
        print_section("NumPy Installation")
        
        try:
            import numpy as np
            print(f"NumPy Version: {np.__version__}")
            print("✅ NumPy is installed")
            self.results['numpy'] = True
        except ImportError:
            print("❌ NumPy not installed")
            print("Install with: pip install numpy")
            self.results['numpy'] = False
    
    def check_memory(self):
        """Check available memory."""
        print_section("Memory Check")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
            print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
            print(f"RAM Usage: {memory.percent}%")
            
            if memory.available / (1024**3) > 2:
                print("✅ Sufficient RAM available for ONNX operations")
                self.results['memory'] = True
            else:
                print("⚠️ Low RAM available - may cause issues with large models")
                self.results['memory'] = False
                
        except ImportError:
            print("ℹ️ Install psutil for memory info: pip install psutil")
            self.results['memory'] = False
        
        # Check GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                print(f"GPU Allocated: {allocated:.1f} GB")
                print(f"GPU Cached: {cached:.1f} GB")
                
                if gpu_memory > 2:
                    print("✅ Sufficient GPU memory for ONNX operations")
                else:
                    print("⚠️ Limited GPU memory - may cause issues")
                    
        except Exception as e:
            print(f"ℹ️ GPU memory check failed: {e}")
    
    def check_onnx_export_capability(self):
        """Test ONNX export capability."""
        print_section("ONNX Export Capability Test")
        
        try:
            import torch
            import torch.onnx
            
            # Create a simple model for testing
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            # Test ONNX export
            model = SimpleModel()
            dummy_input = torch.randn(1, 10)
            
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    "test_export.onnx",
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                )
                
                if os.path.exists("test_export.onnx"):
                    os.remove("test_export.onnx")
                    print("✅ ONNX export capability verified")
                    return True
                else:
                    print("❌ ONNX export failed")
                    return False
                    
            except Exception as e:
                print(f"❌ ONNX export test failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ ONNX export capability test failed: {e}")
            return False
    
    def check_project_structure(self):
        """Check project structure for ONNX-related directories."""
        print_section("Project Structure Check")
        
        # Check for ONNX models directory
        onnx_dir = self.project_root / "results" / "onnx_models"
        if onnx_dir.exists():
            print(f"✅ ONNX models directory exists: {onnx_dir}")
            onnx_files = list(onnx_dir.glob("*.onnx"))
            if onnx_files:
                print(f"   Found {len(onnx_files)} ONNX models:")
                for file in onnx_files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   - {file.name}: {size_mb:.1f} MB")
            else:
                print("   No ONNX models found")
        else:
            print(f"❌ ONNX models directory not found: {onnx_dir}")
        
        # Check for PyTorch models
        runs_dir = self.project_root / "results" / "runs"
        if runs_dir.exists():
            pytorch_models = list(runs_dir.glob("*/weights/best.pt"))
            if pytorch_models:
                print(f"✅ Found {len(pytorch_models)} PyTorch models:")
                for model in pytorch_models:
                    size_mb = model.stat().st_size / (1024 * 1024)
                    print(f"   - {model.parent.parent.name}/weights/best.pt: {size_mb:.1f} MB")
            else:
                print("❌ No PyTorch models found")
        else:
            print("❌ Results/runs directory not found")
    
    def run_all_checks(self):
        """Run all environment checks."""
        self.print_header("ONNX Environment Checker (check0)")
        
        self.check_python_version()
        self.check_pytorch()
        self.check_ultralytics()
        self.check_onnx()
        self.check_opencv()
        self.check_numpy()
        self.check_memory()
        self.check_project_structure()
        
        # Test ONNX export capability
        export_success = self.check_onnx_export_capability()
        
        self.print_header("Check Summary")
        self.print_summary()
        
        return self.results, export_success
    
    def print_summary(self):
        """Print summary of all checks."""
        print("📊 Environment Check Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All environment checks passed! ONNX operations should work.")
        else:
            print("⚠️ Some checks failed. Please install missing dependencies.")
            print("💡 Run: pip install torch ultralytics onnx onnxruntime opencv-python numpy")

def main():
    """Run ONNX environment checker."""
    checker = ONNXEnvironmentChecker()
    results, export_success = checker.run_all_checks()
    
    if export_success:
        print("✅ ONNX export capability verified")
    else:
        print("❌ ONNX export capability failed")
    
    print("\n💡 For ONNX export, use: python main_colab.py --models v8n --onnx-export")

if __name__ == "__main__":
    main() 