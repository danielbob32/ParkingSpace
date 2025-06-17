#!/usr/bin/env python3
"""
Code Quality Check Script
Validates Python syntax for all files in the parkingspace package.
"""

import ast
import os
from pathlib import Path

def check_code_quality():
    """Check syntax and basic code quality."""
    print("=== CODE QUALITY CHECK ===")
    
    # Get the src/parkingspace directory
    src_dir = Path(__file__).parent.parent / "src" / "parkingspace"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return False
    
    python_files = list(src_dir.glob("*.py"))
    
    if not python_files:
        print("❌ No Python files found")
        return False
    
    all_valid = True
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file to check syntax
            ast.parse(content)
            print(f"✓ {file_path.name}")
            
        except SyntaxError as e:
            print(f"❌ {file_path.name}: Syntax error at line {e.lineno}: {e.msg}")
            all_valid = False
        except Exception as e:
            print(f"❌ {file_path.name}: Error reading file: {e}")
            all_valid = False
    
    if all_valid:
        print("✓ All Python files are syntactically valid")
        return True
    else:
        print("❌ Some files have syntax errors")
        return False

if __name__ == "__main__":
    success = check_code_quality()
    exit(0 if success else 1)
