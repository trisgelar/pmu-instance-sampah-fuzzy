#!/usr/bin/env python3
"""
Bug Detector

This module detects common bugs and issues in the project.
"""

import os
import sys
from pathlib import Path
import logging
import re
import ast
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BugDetector:
    """
    Detector for common bugs and issues.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.bugs_found = []
        self.results = {
            'import_bugs': False,
            'syntax_bugs': False,
            'path_bugs': False,
            'type_bugs': False,
            'logic_bugs': False
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
    
    def detect_import_bugs(self):
        """Detect import-related bugs."""
        print_section("Import Bug Detection")
        
        python_files = list(self.project_root.rglob("*.py"))
        import_bugs = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common import issues
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Check for unused imports
                    if line.startswith('import ') or line.startswith('from '):
                        # Simple check for unused imports (basic heuristic)
                        import_name = line.split()[1].split('.')[0]
                        if import_name not in content[content.find(line) + len(line):]:
                            import_bugs.append(f"{file_path}:{line_num} - Potentially unused import: {line}")
                    
                    # Check for circular imports
                    if 'from . import' in line and 'main_colab' in str(file_path):
                        import_bugs.append(f"{file_path}:{line_num} - Potential circular import: {line}")
                    
                    # Check for missing imports
                    if 'yaml' in line and 'import yaml' not in content:
                        import_bugs.append(f"{file_path}:{line_num} - Missing yaml import but yaml used: {line}")
                    
                    if 'torch' in line and 'import torch' not in content:
                        import_bugs.append(f"{file_path}:{line_num} - Missing torch import but torch used: {line}")
                
            except Exception as e:
                import_bugs.append(f"{file_path} - Error reading file: {e}")
        
        if import_bugs:
            print("⚠️ Import bugs found:")
            for bug in import_bugs[:10]:  # Limit output
                print(f"   - {bug}")
            if len(import_bugs) > 10:
                print(f"   ... and {len(import_bugs) - 10} more")
            self.results['import_bugs'] = True
        else:
            print("✅ No import bugs detected")
            self.results['import_bugs'] = False
        
        self.bugs_found.extend(import_bugs)
    
    def detect_syntax_bugs(self):
        """Detect syntax-related bugs."""
        print_section("Syntax Bug Detection")
        
        python_files = list(self.project_root.rglob("*.py"))
        syntax_bugs = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse the file
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_bugs.append(f"{file_path}:{e.lineno} - Syntax error: {e.msg}")
                
                # Check for common syntax issues
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for missing colons
                    if re.match(r'^\s*(if|for|while|def|class|try|except|finally|else|elif)\s+[^:]*$', line):
                        if not line.strip().endswith(':'):
                            syntax_bugs.append(f"{file_path}:{line_num} - Missing colon: {line.strip()}")
                    
                    # Check for unmatched parentheses
                    open_parens = line.count('(') + line.count('[') + line.count('{')
                    close_parens = line.count(')') + line.count(']') + line.count('}')
                    if abs(open_parens - close_parens) > 0:
                        syntax_bugs.append(f"{file_path}:{line_num} - Unmatched parentheses: {line.strip()}")
                    
                    # Check for indentation issues
                    if line.strip() and not line.startswith(' ') and line_num > 1:
                        prev_line = lines[line_num - 2] if line_num > 1 else ""
                        if prev_line.strip() and prev_line.strip().endswith(':'):
                            if not line.startswith('    ') and not line.startswith('\t'):
                                syntax_bugs.append(f"{file_path}:{line_num} - Indentation error after colon: {line.strip()}")
                
            except Exception as e:
                syntax_bugs.append(f"{file_path} - Error reading file: {e}")
        
        if syntax_bugs:
            print("⚠️ Syntax bugs found:")
            for bug in syntax_bugs[:10]:  # Limit output
                print(f"   - {bug}")
            if len(syntax_bugs) > 10:
                print(f"   ... and {len(syntax_bugs) - 10} more")
            self.results['syntax_bugs'] = True
        else:
            print("✅ No syntax bugs detected")
            self.results['syntax_bugs'] = False
        
        self.bugs_found.extend(syntax_bugs)
    
    def detect_path_bugs(self):
        """Detect path-related bugs."""
        print_section("Path Bug Detection")
        
        python_files = list(self.project_root.rglob("*.py"))
        path_bugs = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for hardcoded paths
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for hardcoded paths
                    if '/home/' in line or 'C:\\' in line or 'D:\\' in line:
                        path_bugs.append(f"{file_path}:{line_num} - Hardcoded path: {line.strip()}")
                    
                    # Check for relative path issues
                    if 'results/runs/' in line and not 'os.path' in line:
                        path_bugs.append(f"{file_path}:{line_num} - Hardcoded results path: {line.strip()}")
                    
                    # Check for missing path joins
                    if 'results' in line and 'runs' in line and '+' in line:
                        path_bugs.append(f"{file_path}:{line_num} - Use os.path.join instead of +: {line.strip()}")
                
            except Exception as e:
                path_bugs.append(f"{file_path} - Error reading file: {e}")
        
        if path_bugs:
            print("⚠️ Path bugs found:")
            for bug in path_bugs[:10]:  # Limit output
                print(f"   - {bug}")
            if len(path_bugs) > 10:
                print(f"   ... and {len(path_bugs) - 10} more")
            self.results['path_bugs'] = True
        else:
            print("✅ No path bugs detected")
            self.results['path_bugs'] = False
        
        self.bugs_found.extend(path_bugs)
    
    def detect_type_bugs(self):
        """Detect type-related bugs."""
        print_section("Type Bug Detection")
        
        python_files = list(self.project_root.rglob("*.py"))
        type_bugs = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for type-related issues
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for string concatenation with non-string
                    if '+' in line and ('str(' not in line and 'int(' not in line):
                        if re.search(r'"[^"]*"\s*\+\s*[^"]', line) or re.search(r'[^"]\s*\+\s*"[^"]*"', line):
                            type_bugs.append(f"{file_path}:{line_num} - Potential type mismatch in concatenation: {line.strip()}")
                    
                    # Check for division by zero potential
                    if '/' in line and '0' in line:
                        if re.search(r'/\s*0\b', line) or re.search(r'/\s*[a-zA-Z_][a-zA-Z0-9_]*\s*$', line):
                            type_bugs.append(f"{file_path}:{line_num} - Potential division by zero: {line.strip()}")
                    
                    # Check for list access without bounds checking
                    if '[' in line and ']' in line:
                        if re.search(r'\[[^]]*\]', line) and 'len(' not in line:
                            type_bugs.append(f"{file_path}:{line_num} - Potential index out of bounds: {line.strip()}")
                
            except Exception as e:
                type_bugs.append(f"{file_path} - Error reading file: {e}")
        
        if type_bugs:
            print("⚠️ Type bugs found:")
            for bug in type_bugs[:10]:  # Limit output
                print(f"   - {bug}")
            if len(type_bugs) > 10:
                print(f"   ... and {len(type_bugs) - 10} more")
            self.results['type_bugs'] = True
        else:
            print("✅ No type bugs detected")
            self.results['type_bugs'] = False
        
        self.bugs_found.extend(type_bugs)
    
    def detect_logic_bugs(self):
        """Detect logic-related bugs."""
        print_section("Logic Bug Detection")
        
        python_files = list(self.project_root.rglob("*.py"))
        logic_bugs = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for logic issues
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for unreachable code
                    if 'return' in line and line_num < len(lines):
                        next_line = lines[line_num].strip()
                        if next_line and not next_line.startswith('#') and not next_line.startswith('"""'):
                            logic_bugs.append(f"{file_path}:{line_num} - Code after return may be unreachable: {line.strip()}")
                    
                    # Check for infinite loop potential
                    if 'while True:' in line:
                        logic_bugs.append(f"{file_path}:{line_num} - Potential infinite loop: {line.strip()}")
                    
                    # Check for empty except blocks
                    if 'except:' in line or 'except Exception:' in line:
                        # Check if the except block is empty
                        indent_level = len(line) - len(line.lstrip())
                        next_lines = lines[line_num:line_num+3]
                        has_content = any(len(l.strip()) > 0 and not l.strip().startswith('#') for l in next_lines)
                        if not has_content:
                            logic_bugs.append(f"{file_path}:{line_num} - Empty except block: {line.strip()}")
                    
                    # Check for file operations without context managers
                    if 'open(' in line and 'with ' not in content[:content.find(line)]:
                        logic_bugs.append(f"{file_path}:{line_num} - File operation without context manager: {line.strip()}")
                
            except Exception as e:
                logic_bugs.append(f"{file_path} - Error reading file: {e}")
        
        if logic_bugs:
            print("⚠️ Logic bugs found:")
            for bug in logic_bugs[:10]:  # Limit output
                print(f"   - {bug}")
            if len(logic_bugs) > 10:
                print(f"   ... and {len(logic_bugs) - 10} more")
            self.results['logic_bugs'] = True
        else:
            print("✅ No logic bugs detected")
            self.results['logic_bugs'] = False
        
        self.bugs_found.extend(logic_bugs)
    
    def run_all_detections(self):
        """Run all bug detections."""
        self.print_header("Bug Detector")
        
        self.detect_import_bugs()
        self.detect_syntax_bugs()
        self.detect_path_bugs()
        self.detect_type_bugs()
        self.detect_logic_bugs()
        
        self.print_header("Detection Summary")
        self.print_summary()
        
        return self.results, self.bugs_found
    
    def print_summary(self):
        """Print summary of all detections."""
        print("📊 Bug Detection Summary:")
        print("-" * 40)
        
        total_checks = len(self.results)
        bugs_found = sum(self.results.values())
        
        for check, has_bugs in self.results.items():
            status = "❌" if has_bugs else "✅"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\nTotal bugs found: {len(self.bugs_found)}")
        
        if bugs_found == 0:
            print("🎉 No bugs detected! Code looks clean.")
        else:
            print("⚠️ Bugs detected. Consider using BugFixer to automatically fix some issues.")

def main():
    """Run bug detector."""
    detector = BugDetector()
    results, bugs = detector.run_all_detections()
    
    print(f"\n💡 Found {len(bugs)} potential bugs")
    print("💡 For automatic fixing, use the BugFixer module")

if __name__ == "__main__":
    main() 