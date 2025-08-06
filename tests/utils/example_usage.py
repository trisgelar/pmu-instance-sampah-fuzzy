#!/usr/bin/env python3
"""
Example Usage of Dataset Tools

This script demonstrates how to use the dataset tools for diagnosing
and fixing dataset issues.

Usage:
    python -m tests.dataset_tools.example_usage
"""

from tests.dataset_tools import (
    diagnose_dataset,
    validate_dataset,
    fix_dataset_ultralytics,
    fix_dataset_classes
)

def main():
    """Demonstrate dataset tools usage."""
    print("🔧 Dataset Tools Example Usage")
    print("=" * 40)
    
    # Step 1: Diagnose the dataset
    print("\n📋 Step 1: Diagnosing dataset...")
    diagnosis = diagnose_dataset(verbose=False)
    
    if diagnosis.get('error'):
        print(f"❌ Error: {diagnosis['error']}")
        return
    
    if diagnosis.get('has_issues'):
        print(f"⚠️  Found {len(diagnosis['all_issues'])} issues:")
        for issue in diagnosis['all_issues'][:3]:  # Show first 3 issues
            print(f"  - {issue}")
        if len(diagnosis['all_issues']) > 3:
            print(f"  ... and {len(diagnosis['all_issues']) - 3} more issues")
    else:
        print("✅ No issues found!")
    
    # Step 2: Validate the dataset
    print("\n🔍 Step 2: Validating dataset...")
    validation = validate_dataset(verbose=False)
    
    if validation.get('error'):
        print(f"❌ Error: {validation['error']}")
        return
    
    if validation['valid']:
        print("✅ Dataset validation passed!")
    else:
        print(f"❌ Dataset validation failed with {validation['issue_count']} issues")
    
    # Step 3: Fix issues if needed
    if diagnosis.get('has_issues') or not validation['valid']:
        print("\n🔧 Step 3: Fixing dataset issues...")
        
        # Try Ultralytics fix first (recommended)
        print("  Trying Ultralytics fix...")
        fix_results = fix_dataset_ultralytics(verbose=False)
        
        if fix_results.get('success'):
            print("✅ Dataset fixed successfully using Ultralytics tools!")
        else:
            print("⚠️  Ultralytics fix failed, trying manual fix...")
            
            # Try manual fix as fallback
            manual_results = fix_dataset_classes(verbose=False)
            
            if manual_results.get('success'):
                print("✅ Dataset fixed successfully using manual fix!")
            else:
                print("❌ Both fixes failed. Check the error messages above.")
    
    # Step 4: Final validation
    print("\n✅ Step 4: Final validation...")
    final_validation = validate_dataset(verbose=False)
    
    if final_validation['valid']:
        print("🎉 Dataset is ready for training!")
    else:
        print("❌ Dataset still has issues. Manual intervention may be required.")
    
    print("\n📊 Summary:")
    print(f"  - Dataset path: {diagnosis.get('dataset_path', 'Unknown')}")
    print(f"  - Initial issues: {len(diagnosis.get('all_issues', []))}")
    print(f"  - Final validation: {'✅ Passed' if final_validation['valid'] else '❌ Failed'}")

if __name__ == "__main__":
    main() 