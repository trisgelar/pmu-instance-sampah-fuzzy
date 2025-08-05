#!/usr/bin/env python3
"""
Test runner for the waste detection system.
Executes all unit tests and provides comprehensive test reporting.
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests(test_pattern="test_*.py", verbosity=2, test_category=None):
    """
    Discover and run all tests in the tests directory.
    
    Args:
        test_pattern (str): Pattern to match test files
        verbosity (int): Test verbosity level (1=quiet, 2=normal, 3=verbose)
        test_category (str): Specific test category to run (unit, integration, diagnostic, fixes, utils)
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover tests in the tests directory
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return None
    
    # If specific category is requested, run only that category
    if test_category:
        category_dir = tests_dir / test_category
        if not category_dir.exists():
            print(f"❌ Test category directory not found: {category_dir}")
            return None
        search_dir = category_dir
    else:
        search_dir = tests_dir
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.discover(str(search_dir), pattern=test_pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_specific_test(test_name, verbosity=2):
    """
    Run a specific test by name.
    
    Args:
        test_name (str): Name of the test to run (e.g., "test_config_manager")
        verbosity (int): Test verbosity level
    
    Returns:
        unittest.TestResult: Test results
    """
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return None
    
    # Find the test file
    test_file = tests_dir / f"test_{test_name}.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return None
    
    # Import and run the specific test
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Create test suite for the specific module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_security_tests():
    """
    Run security-specific tests.
    
    Returns:
        unittest.TestResult: Test results
    """
    print("🔒 Running Security Tests...")
    
    # Import security test modules
    from tests.test_secrets_validation import TestSecretsValidation, TestSecretsValidationIntegration
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add security test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSecretsValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSecretsValidationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_configuration_tests():
    """
    Run configuration-specific tests.
    
    Returns:
        unittest.TestResult: Test results
    """
    print("⚙️  Running Configuration Tests...")
    
    # Import configuration test modules
    from tests.test_config_manager import (
        TestConfigManager, TestModelConfig, TestDatasetConfig, 
        TestFuzzyConfig, TestLoggingConfig, TestSystemConfig, TestEnvironment
    )
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add configuration test classes
    test_classes = [
        TestConfigManager, TestModelConfig, TestDatasetConfig,
        TestFuzzyConfig, TestLoggingConfig, TestSystemConfig, TestEnvironment
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_fuzzy_logic_tests():
    """
    Run fuzzy logic-specific tests.
    
    Returns:
        unittest.TestResult: Test results
    """
    print("🧠 Running Fuzzy Logic Tests...")
    
    # Import fuzzy logic test modules
    from tests.test_fuzzy_area_classifier import TestFuzzyAreaClassifier, TestFuzzyAreaClassifierIntegration
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add fuzzy logic test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzyAreaClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzyAreaClassifierIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_exception_tests():
    """
    Run exception-specific tests.
    
    Returns:
        unittest.TestResult: Test results
    """
    print("🚨 Running Exception Tests...")
    
    # Import exception test modules
    from tests.test_exceptions import TestExceptionHierarchy, TestExceptionUsage
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add exception test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHierarchy))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionUsage))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_integration_tests():
    """
    Run integration tests.
    
    Returns:
        unittest.TestResult: Test results
    """
    print("🔗 Running Integration Tests...")
    
    # Import integration test modules
    from tests.test_main_colab import TestWasteDetectionSystemColabIntegration
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add integration test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWasteDetectionSystemColabIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def print_test_summary(results):
    """
    Print a comprehensive test summary.
    
    Args:
        results (list): List of (test_name, result, duration) tuples
    """
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_duration = 0
    
    for test_name, result, duration in results:
        if result is None:
            print(f"❌ {test_name}: Failed to run")
            continue
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_duration += duration
        
        status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"
        print(f"{status} {test_name}: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors ({duration:.2f}s)")
        
        # Print detailed failures and errors
        if result.failures:
            print(f"   Failures:")
            for test, traceback in result.failures:
                print(f"     - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"   Errors:")
            for test, traceback in result.errors:
                print(f"     - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("\n" + "="*60)
    print(f"📈 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Total Failures: {total_failures}")
    print(f"   Total Errors: {total_errors}")
    print(f"   Total Duration: {total_duration:.2f}s")
    print(f"   Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    print("="*60)

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run waste detection system tests")
    parser.add_argument("--test", help="Run specific test (e.g., config_manager, fuzzy_logic)")
    parser.add_argument("--category", choices=["all", "unit", "integration", "diagnostic", "fixes", "utils", "security", "config", "fuzzy", "exceptions"], 
                       default="all", help="Run tests by category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 1 if args.quiet else (3 if args.verbose else 2)
    
    print("🧪 Waste Detection System - Test Runner")
    print("="*60)
    
    results = []
    
    if args.test:
        # Run specific test
        print(f"🎯 Running specific test: {args.test}")
        result, duration = run_specific_test(args.test, verbosity)
        if result:
            results.append((args.test, result, duration))
    else:
        # Run tests by category
        if args.category == "all":
            print("🚀 Running all tests...")
            
            # Run all test categories
            categories = [
                ("Security Tests", run_security_tests),
                ("Configuration Tests", run_configuration_tests),
                ("Fuzzy Logic Tests", run_fuzzy_logic_tests),
                ("Exception Tests", run_exception_tests),
                ("Integration Tests", run_integration_tests)
            ]
            
            for category_name, test_function in categories:
                try:
                    result, duration = test_function()
                    if result:
                        results.append((category_name, result, duration))
                except Exception as e:
                    print(f"❌ {category_name} failed to run: {e}")
                    results.append((category_name, None, 0))
        
        elif args.category == "unit":
            print("🧪 Running unit tests...")
            result, duration = discover_and_run_tests(test_category="unit", verbosity=verbosity)
            if result:
                results.append(("Unit Tests", result, duration))
        
        elif args.category == "integration":
            print("🔗 Running integration tests...")
            result, duration = discover_and_run_tests(test_category="integration", verbosity=verbosity)
            if result:
                results.append(("Integration Tests", result, duration))
        
        elif args.category == "diagnostic":
            print("🔍 Running diagnostic tests...")
            result, duration = discover_and_run_tests(test_category="diagnostic", verbosity=verbosity)
            if result:
                results.append(("Diagnostic Tests", result, duration))
        
        elif args.category == "fixes":
            print("🔧 Running fix verification tests...")
            result, duration = discover_and_run_tests(test_category="fixes", verbosity=verbosity)
            if result:
                results.append(("Fix Verification Tests", result, duration))
        
        elif args.category == "utils":
            print("🛠️ Running utility tests...")
            result, duration = discover_and_run_tests(test_category="utils", verbosity=verbosity)
            if result:
                results.append(("Utility Tests", result, duration))
        
        elif args.category == "security":
            result, duration = run_security_tests()
            if result:
                results.append(("Security Tests", result, duration))
        
        elif args.category == "config":
            result, duration = run_configuration_tests()
            if result:
                results.append(("Configuration Tests", result, duration))
        
        elif args.category == "fuzzy":
            result, duration = run_fuzzy_logic_tests()
            if result:
                results.append(("Fuzzy Logic Tests", result, duration))
        
        elif args.category == "exceptions":
            result, duration = run_exception_tests()
            if result:
                results.append(("Exception Tests", result, duration))
    
    # Print summary
    if results:
        print_test_summary(results)
        
        # Determine overall success
        all_successful = all(result.wasSuccessful() if result else False for _, result, _ in results)
        
        if all_successful:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please review the results above.")
            sys.exit(1)
    else:
        print("\n❌ No tests were run successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main() 