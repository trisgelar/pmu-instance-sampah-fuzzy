#!/usr/bin/env python3
"""
ONNX Model Checker (check1)

This module validates ONNX model files and their properties.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXModelChecker:
    """
    Checker for ONNX model files and their properties.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {
            'model_files_exist': False,
            'model_files_valid': False,
            'model_sizes_ok': False,
            'model_metadata_ok': False,
            'model_inputs_ok': False,
            'model_outputs_ok': False,
            'model_opsets_ok': False
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
    
    def check_onnx_files_exist(self):
        """Check if ONNX model files exist."""
        print_section("ONNX Model Files Check")
        
        onnx_dir = self.project_root / "results" / "onnx_models"
        if not onnx_dir.exists():
            print(f"❌ ONNX models directory not found: {onnx_dir}")
            self.results['model_files_exist'] = False
            return []
        
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print("❌ No ONNX model files found")
            self.results['model_files_exist'] = False
            return []
        
        print(f"✅ Found {len(onnx_files)} ONNX model files:")
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.1f} MB")
        
        self.results['model_files_exist'] = True
        return onnx_files
    
    def check_onnx_file_validity(self, onnx_files):
        """Check if ONNX files are valid."""
        print_section("ONNX File Validity Check")
        
        try:
            import onnx
            
            valid_files = []
            for file in onnx_files:
                try:
                    # Load and validate ONNX model
                    model = onnx.load(str(file))
                    onnx.checker.check_model(model)
                    
                    print(f"✅ {file.name}: Valid ONNX model")
                    valid_files.append(file)
                    
                except Exception as e:
                    print(f"❌ {file.name}: Invalid ONNX model - {e}")
            
            if valid_files:
                print(f"✅ {len(valid_files)}/{len(onnx_files)} ONNX files are valid")
                self.results['model_files_valid'] = True
            else:
                print("❌ No valid ONNX files found")
                self.results['model_files_valid'] = False
            
            return valid_files
            
        except ImportError:
            print("❌ ONNX library not available")
            print("Install with: pip install onnx")
            self.results['model_files_valid'] = False
            return []
    
    def check_model_sizes(self, onnx_files):
        """Check if model sizes are reasonable."""
        print_section("Model Size Check")
        
        if not onnx_files:
            print("❌ No ONNX files to check")
            self.results['model_sizes_ok'] = False
            return
        
        size_issues = []
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            
            # Check for reasonable sizes (not too small, not too large)
            if size_mb < 0.1:  # Less than 100KB
                size_issues.append(f"{file.name}: Too small ({size_mb:.1f} MB)")
            elif size_mb > 500:  # More than 500MB
                size_issues.append(f"{file.name}: Too large ({size_mb:.1f} MB)")
            else:
                print(f"✅ {file.name}: {size_mb:.1f} MB (reasonable size)")
        
        if size_issues:
            print("⚠️ Size issues found:")
            for issue in size_issues:
                print(f"   - {issue}")
            self.results['model_sizes_ok'] = False
        else:
            print("✅ All model sizes are reasonable")
            self.results['model_sizes_ok'] = True
    
    def check_model_metadata(self, onnx_files):
        """Check model metadata and properties."""
        print_section("Model Metadata Check")
        
        try:
            import onnx
            
            metadata_issues = []
            for file in onnx_files:
                try:
                    model = onnx.load(str(file))
                    
                    # Check model properties
                    print(f"\n📊 {file.name} metadata:")
                    print(f"   - IR Version: {model.ir_version}")
                    print(f"   - Opset Version: {model.opset_import[0].version}")
                    print(f"   - Producer: {model.producer_name}")
                    print(f"   - Domain: {model.domain}")
                    
                    # Check for required properties
                    if not model.producer_name:
                        metadata_issues.append(f"{file.name}: Missing producer name")
                    
                    if model.opset_import[0].version < 11:
                        metadata_issues.append(f"{file.name}: Opset version too old ({model.opset_import[0].version})")
                    
                except Exception as e:
                    metadata_issues.append(f"{file.name}: Metadata check failed - {e}")
            
            if metadata_issues:
                print("⚠️ Metadata issues found:")
                for issue in metadata_issues:
                    print(f"   - {issue}")
                self.results['model_metadata_ok'] = False
            else:
                print("✅ All model metadata is valid")
                self.results['model_metadata_ok'] = True
                
        except ImportError:
            print("❌ ONNX library not available")
            self.results['model_metadata_ok'] = False
    
    def check_model_inputs(self, onnx_files):
        """Check model input specifications."""
        print_section("Model Inputs Check")
        
        try:
            import onnx
            
            input_issues = []
            for file in onnx_files:
                try:
                    model = onnx.load(str(file))
                    
                    print(f"\n📥 {file.name} inputs:")
                    for i, input_info in enumerate(model.graph.input):
                        print(f"   - Input {i}: {input_info.name}")
                        print(f"     Type: {input_info.type.tensor_type.elem_type}")
                        
                        # Check shape
                        shape = []
                        for dim in input_info.type.tensor_type.shape.dim:
                            if dim.dim_param:
                                shape.append(dim.dim_param)
                            else:
                                shape.append(dim.dim_value)
                        print(f"     Shape: {shape}")
                        
                        # Check for common issues
                        if len(shape) != 4:
                            input_issues.append(f"{file.name}: Input {i} should have 4 dimensions")
                        
                        if shape[1] not in [1, 3]:  # Channels should be 1 or 3
                            input_issues.append(f"{file.name}: Input {i} channels should be 1 or 3, got {shape[1]}")
                    
                except Exception as e:
                    input_issues.append(f"{file.name}: Input check failed - {e}")
            
            if input_issues:
                print("⚠️ Input issues found:")
                for issue in input_issues:
                    print(f"   - {issue}")
                self.results['model_inputs_ok'] = False
            else:
                print("✅ All model inputs are valid")
                self.results['model_inputs_ok'] = True
                
        except ImportError:
            print("❌ ONNX library not available")
            self.results['model_inputs_ok'] = False
    
    def check_model_outputs(self, onnx_files):
        """Check model output specifications."""
        print_section("Model Outputs Check")
        
        try:
            import onnx
            
            output_issues = []
            for file in onnx_files:
                try:
                    model = onnx.load(str(file))
                    
                    print(f"\n📤 {file.name} outputs:")
                    for i, output_info in enumerate(model.graph.output):
                        print(f"   - Output {i}: {output_info.name}")
                        print(f"     Type: {output_info.type.tensor_type.elem_type}")
                        
                        # Check shape
                        shape = []
                        for dim in output_info.type.tensor_type.shape.dim:
                            if dim.dim_param:
                                shape.append(dim.dim_param)
                            else:
                                shape.append(dim.dim_value)
                        print(f"     Shape: {shape}")
                        
                        # Check for common issues
                        if len(shape) != 2:
                            output_issues.append(f"{file.name}: Output {i} should have 2 dimensions")
                    
                except Exception as e:
                    output_issues.append(f"{file.name}: Output check failed - {e}")
            
            if output_issues:
                print("⚠️ Output issues found:")
                for issue in output_issues:
                    print(f"   - {issue}")
                self.results['model_outputs_ok'] = False
            else:
                print("✅ All model outputs are valid")
                self.results['model_outputs_ok'] = True
                
        except ImportError:
            print("❌ ONNX library not available")
            self.results['model_outputs_ok'] = False
    
    def check_model_opsets(self, onnx_files):
        """Check model opset compatibility."""
        print_section("Model Opset Check")
        
        try:
            import onnx
            
            opset_issues = []
            for file in onnx_files:
                try:
                    model = onnx.load(str(file))
                    
                    opset_version = model.opset_import[0].version
                    print(f"📦 {file.name}: Opset version {opset_version}")
                    
                    # Check for compatibility
                    if opset_version < 11:
                        opset_issues.append(f"{file.name}: Opset version {opset_version} is too old (need >= 11)")
                    elif opset_version > 17:
                        opset_issues.append(f"{file.name}: Opset version {opset_version} is very new (may have compatibility issues)")
                    else:
                        print(f"   ✅ Opset version {opset_version} is compatible")
                    
                except Exception as e:
                    opset_issues.append(f"{file.name}: Opset check failed - {e}")
            
            if opset_issues:
                print("⚠️ Opset issues found:")
                for issue in opset_issues:
                    print(f"   - {issue}")
                self.results['model_opsets_ok'] = False
            else:
                print("✅ All model opsets are compatible")
                self.results['model_opsets_ok'] = True
                
        except ImportError:
            print("❌ ONNX library not available")
            self.results['model_opsets_ok'] = False
    
    def run_all_checks(self):
        """Run all ONNX model checks."""
        self.print_header("ONNX Model Checker (check1)")
        
        # Check if files exist
        onnx_files = self.check_onnx_files_exist()
        
        if not onnx_files:
            print("❌ No ONNX files to check")
            return self.results
        
        # Check file validity
        valid_files = self.check_onnx_file_validity(onnx_files)
        
        if not valid_files:
            print("❌ No valid ONNX files to check")
            return self.results
        
        # Check model properties
        self.check_model_sizes(valid_files)
        self.check_model_metadata(valid_files)
        self.check_model_inputs(valid_files)
        self.check_model_outputs(valid_files)
        self.check_model_opsets(valid_files)
        
        self.print_header("Check Summary")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all checks."""
        print("📊 Model Check Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All model checks passed! ONNX models are valid.")
        else:
            print("⚠️ Some checks failed. Please review the issues above.")

def main():
    """Run ONNX model checker."""
    checker = ONNXModelChecker()
    results = checker.run_all_checks()
    
    print("\n💡 For ONNX model creation, use: python main_colab.py --models v8n --onnx-export")

if __name__ == "__main__":
    main() 