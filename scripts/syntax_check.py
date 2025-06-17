#!/usr/bin/env python3
"""
Syntax check script for all Python files in the project.
"""

import ast
import os
import sys

def main():
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    src_dir = os.path.join(project_root, 'src', 'parkingspace')
    
    if not os.path.exists(src_dir):
        print(f"❌ Source directory not found: {src_dir}")
        sys.exit(1)
    
    print("=== SYNTAX CHECK ===")
    
    for filename in os.listdir(src_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(src_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                print(f"✓ {filename}")
            except SyntaxError as e:
                print(f"❌ {filename}: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"⚠ {filename}: {e}")
    
    print("All Python files are syntactically valid")

if __name__ == "__main__":
    main()
