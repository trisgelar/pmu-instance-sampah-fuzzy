#!/usr/bin/env python3
"""
Unit tests for the secrets validation script.
Tests validation functionality and security checks.
"""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the validation functions
from validate_secrets import (
    check_secrets_file, check_gitignore, check_git_status,
    check_file_permissions, check_environment_variables,
    validate_api_key_format, run_security_audit
)


class TestSecretsValidation(unittest.TestCase):
    """Test cases for secrets validation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.secrets_file = os.path.join(self.temp_dir, "secrets.yaml")
        self.gitignore_file = os.path.join(self.temp_dir, ".gitignore")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.secrets_file):
            os.remove(self.secrets_file)
        if os.path.exists(self.gitignore_file):
            os.remove(self.gitignore_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_check_secrets_file_exists(self):
        """Test checking secrets file when it exists."""
        # Create a valid secrets file
        valid_secrets = {
            'roboflow_api_key': 'test_api_key_123456789'
        }
        
        with open(self.secrets_file, 'w') as f:
            yaml.dump(valid_secrets, f)
        
        result = check_secrets_file()
        self.assertTrue(result)

    def test_check_secrets_file_not_exists(self):
        """Test checking secrets file when it doesn't exist."""
        # Remove secrets file if it exists
        if os.path.exists('secrets.yaml'):
            os.remove('secrets.yaml')
        
        result = check_secrets_file()
        self.assertFalse(result)

    def test_check_secrets_file_empty(self):
        """Test checking secrets file when it's empty."""
        # Create empty secrets file
        with open(self.secrets_file, 'w') as f:
            f.write('')
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_secrets_file()
            self.assertFalse(result)

    def test_check_secrets_file_missing_required_keys(self):
        """Test checking secrets file with missing required keys."""
        # Create secrets file without required keys
        invalid_secrets = {
            'other_key': 'value'
        }
        
        with open(self.secrets_file, 'w') as f:
            yaml.dump(invalid_secrets, f)
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_secrets_file()
            self.assertFalse(result)

    def test_check_secrets_file_placeholder_api_key(self):
        """Test checking secrets file with placeholder API key."""
        # Create secrets file with placeholder
        placeholder_secrets = {
            'roboflow_api_key': 'YOUR_ROBOFLOW_API_KEY_HERE'
        }
        
        with open(self.secrets_file, 'w') as f:
            yaml.dump(placeholder_secrets, f)
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_secrets_file()
            self.assertFalse(result)

    def test_check_secrets_file_invalid_api_key(self):
        """Test checking secrets file with invalid API key."""
        # Create secrets file with short API key
        invalid_secrets = {
            'roboflow_api_key': 'short'
        }
        
        with open(self.secrets_file, 'w') as f:
            yaml.dump(invalid_secrets, f)
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_secrets_file()
            self.assertFalse(result)

    def test_check_secrets_file_invalid_yaml(self):
        """Test checking secrets file with invalid YAML."""
        # Create file with invalid YAML
        with open(self.secrets_file, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_secrets_file()
            self.assertFalse(result)

    def test_check_gitignore_exists(self):
        """Test checking .gitignore when it exists."""
        # Create a valid .gitignore file
        gitignore_content = """
        secrets.yaml
        *.key
        .env
        """
        
        with open(self.gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.gitignore_file
            
            result = check_gitignore()
            self.assertTrue(result)

    def test_check_gitignore_not_exists(self):
        """Test checking .gitignore when it doesn't exist."""
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = check_gitignore()
            self.assertFalse(result)

    def test_check_gitignore_missing_patterns(self):
        """Test checking .gitignore with missing important patterns."""
        # Create .gitignore without important patterns
        incomplete_gitignore = """
        *.log
        """
        
        with open(self.gitignore_file, 'w') as f:
            f.write(incomplete_gitignore)
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.gitignore_file
            
            result = check_gitignore()
            self.assertFalse(result)

    def test_check_git_status_ignored(self):
        """Test checking git status when file is ignored."""
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0  # File is ignored
            
            result = check_git_status()
            self.assertTrue(result)

    def test_check_git_status_not_ignored(self):
        """Test checking git status when file is not ignored."""
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # File is not ignored
            
            result = check_git_status()
            self.assertFalse(result)

    def test_check_git_status_git_not_found(self):
        """Test checking git status when git is not found."""
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            
            result = check_git_status()
            self.assertTrue(result)  # Should not fail if git not found

    def test_check_file_permissions_exists(self):
        """Test checking file permissions when file exists."""
        # Create a test file
        with open(self.secrets_file, 'w') as f:
            f.write('test content')
        
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__str__ = lambda x: self.secrets_file
            
            result = check_file_permissions()
            self.assertTrue(result)

    def test_check_file_permissions_not_exists(self):
        """Test checking file permissions when file doesn't exist."""
        with patch('validate_secrets.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = check_file_permissions()
            self.assertFalse(result)

    def test_check_environment_variables_found(self):
        """Test checking environment variables when they exist."""
        with patch.dict(os.environ, {'ROBOFLOW_API_KEY': 'test_key'}):
            result = check_environment_variables()
            self.assertTrue(result)

    def test_check_environment_variables_not_found(self):
        """Test checking environment variables when they don't exist."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_environment_variables()
            self.assertTrue(result)  # Should not fail if no env vars

    def test_validate_api_key_format_valid(self):
        """Test API key format validation with valid keys."""
        valid_keys = [
            'valid_api_key_123456789',
            'another_valid_key_987654321',
            'test_key_with_underscores_and_numbers_123'
        ]
        
        for key in valid_keys:
            is_valid, message = validate_api_key_format(key)
            self.assertTrue(is_valid, f"Key '{key}' should be valid: {message}")

    def test_validate_api_key_format_invalid(self):
        """Test API key format validation with invalid keys."""
        invalid_cases = [
            ('', "Empty key"),
            ('YOUR_ROBOFLOW_API_KEY_HERE', "Placeholder key"),
            ('short', "Too short"),
            ('A' * 201, "Too long"),  # 201 characters
            ('invalid@key#with$special%chars', "Special characters")
        ]
        
        for key, description in invalid_cases:
            is_valid, message = validate_api_key_format(key)
            self.assertFalse(is_valid, f"{description}: '{key}' should be invalid")

    def test_run_security_audit_all_passed(self):
        """Test security audit when all checks pass."""
        with patch('validate_secrets.check_secrets_file', return_value=True), \
             patch('validate_secrets.check_gitignore', return_value=True), \
             patch('validate_secrets.check_git_status', return_value=True), \
             patch('validate_secrets.check_file_permissions', return_value=True), \
             patch('validate_secrets.check_environment_variables', return_value=True):
            
            result = run_security_audit()
            self.assertTrue(result)

    def test_run_security_audit_some_failed(self):
        """Test security audit when some checks fail."""
        with patch('validate_secrets.check_secrets_file', return_value=False), \
             patch('validate_secrets.check_gitignore', return_value=True), \
             patch('validate_secrets.check_git_status', return_value=True), \
             patch('validate_secrets.check_file_permissions', return_value=True), \
             patch('validate_secrets.check_environment_variables', return_value=True):
            
            result = run_security_audit()
            self.assertFalse(result)

    def test_run_security_audit_with_exception(self):
        """Test security audit when a check raises an exception."""
        def failing_check():
            raise Exception("Test exception")
        
        with patch('validate_secrets.check_secrets_file', side_effect=failing_check), \
             patch('validate_secrets.check_gitignore', return_value=True), \
             patch('validate_secrets.check_git_status', return_value=True), \
             patch('validate_secrets.check_file_permissions', return_value=True), \
             patch('validate_secrets.check_environment_variables', return_value=True):
            
            result = run_security_audit()
            self.assertFalse(result)


class TestSecretsValidationIntegration(unittest.TestCase):
    """Integration tests for secrets validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.secrets_file = os.path.join(self.temp_dir, "secrets.yaml")
        self.gitignore_file = os.path.join(self.temp_dir, ".gitignore")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.secrets_file):
            os.remove(self.secrets_file)
        if os.path.exists(self.gitignore_file):
            os.remove(self.gitignore_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_complete_validation_pipeline(self):
        """Test the complete validation pipeline with valid setup."""
        # Create valid secrets file
        valid_secrets = {
            'roboflow_api_key': 'test_api_key_12345678901234567890'
        }
        with open(self.secrets_file, 'w') as f:
            yaml.dump(valid_secrets, f)
        
        # Create valid .gitignore file
        gitignore_content = """
        secrets.yaml
        *.key
        .env
        *.log
        datasets/
        models/
        runs/
        *.pt
        *.onnx
        *.rknn
        """
        with open(self.gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        # Mock git check to return ignored
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Run security audit
            result = run_security_audit()
            self.assertTrue(result)

    def test_validation_with_missing_files(self):
        """Test validation when required files are missing."""
        # Don't create any files
        
        # Mock git check to return ignored
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Run security audit
            result = run_security_audit()
            self.assertFalse(result)

    def test_validation_with_invalid_setup(self):
        """Test validation with invalid setup."""
        # Create invalid secrets file
        invalid_secrets = {
            'roboflow_api_key': 'YOUR_ROBOFLOW_API_KEY_HERE'
        }
        with open(self.secrets_file, 'w') as f:
            yaml.dump(invalid_secrets, f)
        
        # Create incomplete .gitignore file
        incomplete_gitignore = """
        *.log
        """
        with open(self.gitignore_file, 'w') as f:
            f.write(incomplete_gitignore)
        
        # Mock git check to return not ignored
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            # Run security audit
            result = run_security_audit()
            self.assertFalse(result)

    def test_validation_with_environment_variables(self):
        """Test validation when environment variables are set."""
        # Create valid secrets file
        valid_secrets = {
            'roboflow_api_key': 'test_api_key_12345678901234567890'
        }
        with open(self.secrets_file, 'w') as f:
            yaml.dump(valid_secrets, f)
        
        # Create valid .gitignore file
        gitignore_content = """
        secrets.yaml
        *.key
        .env
        """
        with open(self.gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        # Mock git check to return ignored
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Set environment variables
            with patch.dict(os.environ, {'ROBOFLOW_API_KEY': 'env_test_key'}):
                # Run security audit
                result = run_security_audit()
                self.assertTrue(result)

    def test_validation_performance(self):
        """Test validation performance with large files."""
        import time
        
        # Create large secrets file
        large_secrets = {
            'roboflow_api_key': 'test_api_key_12345678901234567890',
            'additional_keys': ['key' + str(i) for i in range(1000)]
        }
        with open(self.secrets_file, 'w') as f:
            yaml.dump(large_secrets, f)
        
        # Create large .gitignore file
        large_gitignore = '\n'.join([f'pattern_{i}' for i in range(1000)])
        with open(self.gitignore_file, 'w') as f:
            f.write(large_gitignore)
        
        # Mock git check
        with patch('validate_secrets.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Test performance
            start_time = time.time()
            result = run_security_audit()
            end_time = time.time()
            
            duration = end_time - start_time
            self.assertLess(duration, 5.0, f"Validation took too long: {duration:.2f} seconds")
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main() 