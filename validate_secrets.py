#!/usr/bin/env python3
"""
Secrets Validation Script for Waste Detection System

This script validates the secrets configuration and security setup.
Run this script to ensure your secrets are properly configured.
"""

import os
import yaml
import sys
from pathlib import Path

def check_secrets_file():
    """Check if secrets.yaml exists and is properly configured."""
    print("🔍 Checking secrets.yaml file...")
    
    secrets_file = Path("secrets.yaml")
    
    if not secrets_file.exists():
        print("❌ secrets.yaml not found!")
        print("   Please create secrets.yaml with your API keys")
        return False
    
    try:
        with open(secrets_file, 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f)
        
        if not secrets:
            print("❌ secrets.yaml is empty!")
            return False
        
        # Check for required keys
        required_keys = ['roboflow_api_key']
        missing_keys = []
        
        for key in required_keys:
            if key not in secrets:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        # Validate API key format
        api_key = secrets['roboflow_api_key']
        if api_key == "YOUR_ROBOFLOW_API_KEY_HERE":
            print("❌ Please replace the placeholder API key with your actual Roboflow API key")
            return False
        
        if len(api_key) < 10:
            print("❌ API key appears to be invalid (too short)")
            return False
        
        print("✅ secrets.yaml is properly configured")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML format in secrets.yaml: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading secrets.yaml: {e}")
        return False

def check_gitignore():
    """Check if .gitignore properly excludes sensitive files."""
    print("\n🔍 Checking .gitignore configuration...")
    
    gitignore_file = Path(".gitignore")
    
    if not gitignore_file.exists():
        print("❌ .gitignore not found!")
        return False
    
    try:
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        # Check for important patterns
        important_patterns = [
            'secrets.yaml',
            'secrets.yml',
            '*.key',
            '*.pem',
            '.env',
            '*.log',
            'datasets/',
            'models/',
            'runs/',
            '*.pt',
            '*.onnx',
            '*.rknn'
        ]
        
        missing_patterns = []
        for pattern in important_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"❌ Missing important patterns in .gitignore: {missing_patterns}")
            return False
        
        print("✅ .gitignore properly configured")
        return True
        
    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
        return False

def check_git_status():
    """Check if secrets.yaml is being tracked by git."""
    print("\n🔍 Checking git tracking status...")
    
    try:
        import subprocess
        
        # Check if secrets.yaml is ignored
        result = subprocess.run(
            ['git', 'check-ignore', 'secrets.yaml'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ secrets.yaml is properly ignored by git")
            return True
        else:
            print("❌ secrets.yaml is NOT ignored by git!")
            print("   This is a security risk!")
            return False
            
    except FileNotFoundError:
        print("⚠️  Git not found. Skipping git status check.")
        return True
    except Exception as e:
        print(f"⚠️  Error checking git status: {e}")
        return True

def check_file_permissions():
    """Check if secrets.yaml has proper file permissions."""
    print("\n🔍 Checking file permissions...")
    
    secrets_file = Path("secrets.yaml")
    
    if not secrets_file.exists():
        print("❌ secrets.yaml not found!")
        return False
    
    try:
        # Get file permissions (Unix-like systems)
        stat = secrets_file.stat()
        mode = stat.st_mode & 0o777
        
        # Check if file is readable/writable by owner only
        if mode == 0o600 or mode == 0o400:
            print("✅ secrets.yaml has proper permissions")
            return True
        else:
            print(f"⚠️  secrets.yaml permissions are {oct(mode)}")
            print("   Consider setting permissions to 600 (owner read/write only)")
            print("   Run: chmod 600 secrets.yaml")
            return True  # Not critical, just a warning
            
    except Exception as e:
        print(f"⚠️  Could not check file permissions: {e}")
        return True

def check_environment_variables():
    """Check if environment variables are set as alternative."""
    print("\n🔍 Checking environment variables...")
    
    env_vars = [
        'ROBOFLOW_API_KEY',
        'GOOGLE_CLOUD_API_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY'
    ]
    
    found_vars = []
    for var in env_vars:
        if os.getenv(var):
            found_vars.append(var)
    
    if found_vars:
        print(f"✅ Found environment variables: {found_vars}")
        print("   Note: Environment variables take precedence over secrets.yaml")
    else:
        print("ℹ️  No relevant environment variables found")
        print("   Using secrets.yaml for configuration")
    
    return True

def validate_api_key_format(api_key):
    """Validate API key format."""
    if not api_key or api_key == "YOUR_ROBOFLOW_API_KEY_HERE":
        return False, "Placeholder or empty API key"
    
    if len(api_key) < 10:
        return False, "API key too short"
    
    if len(api_key) > 200:
        return False, "API key too long"
    
    # Basic format validation for Roboflow keys
    if not api_key.replace('_', '').replace('-', '').isalnum():
        return False, "API key contains invalid characters"
    
    return True, "Valid API key format"

def run_security_audit():
    """Run a comprehensive security audit."""
    print("\n" + "="*50)
    print("🔒 SECURITY AUDIT")
    print("="*50)
    
    checks = [
        ("Secrets File", check_secrets_file),
        ("Git Ignore", check_gitignore),
        ("Git Tracking", check_git_status),
        ("File Permissions", check_file_permissions),
        ("Environment Variables", check_environment_variables)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "="*50)
    print(f"📊 AUDIT RESULTS: {passed}/{total} checks passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All security checks passed!")
        print("✅ Your secrets configuration is secure")
    else:
        print("⚠️  Some security checks failed")
        print("   Please review the issues above")
    
    return passed == total

def main():
    """Main validation function."""
    print("🔐 Waste Detection System - Secrets Validation")
    print("="*50)
    
    # Run security audit
    success = run_security_audit()
    
    # Additional recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("1. Keep your API keys secure and never share them")
    print("2. Rotate your API keys regularly")
    print("3. Use different keys for development and production")
    print("4. Monitor API usage for suspicious activity")
    print("5. Set up alerts for unusual API usage patterns")
    
    if success:
        print("\n✅ Your secrets configuration is ready!")
        print("   You can now run the waste detection system")
        sys.exit(0)
    else:
        print("\n❌ Please fix the issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main() 