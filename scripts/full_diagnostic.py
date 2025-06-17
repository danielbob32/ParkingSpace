#!/usr/bin/env python3
"""
Full Diagnostic Script
Comprehensive system diagnostic for the parkingspace package.
"""

import sys
from pathlib import Path
import os

def full_diagnostic():
    """Run complete system diagnostic."""
    print("=== PARKINGSPACE DIAGNOSTIC ===")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Project root: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Import the package
        import src.parkingspace as ps
        print("✓ Package import successful")
        
        # Check version if available
        if hasattr(ps, '__version__'):
            print(f"Package version: {ps.__version__}")
        else:
            print("Package version: Not defined")
        
        # Check available exports
        if hasattr(ps, '__all__'):
            print(f"Available exports: {list(ps.__all__)}")
        else:
            # Get all public attributes
            exports = [attr for attr in dir(ps) if not attr.startswith('_')]
            print(f"Available exports: {exports}")
        
        # Test key components
        print("\n=== COMPONENT TESTS ===")
        
        # Test main module
        try:
            from src.parkingspace import main
            print("✓ Main module import successful")
        except Exception as e:
            print(f"❌ Main module import failed: {e}")
        
        # Test pipeline module
        try:
            from src.parkingspace import pipeline
            print("✓ Pipeline module import successful")
        except Exception as e:
            print(f"❌ Pipeline module import failed: {e}")
        
        # Test regions module
        try:
            from src.parkingspace import regions
            print("✓ Regions module import successful")
        except Exception as e:
            print(f"❌ Regions module import failed: {e}")
        
        # Test utils module
        try:
            from src.parkingspace import utils
            print("✓ Utils module import successful")
        except Exception as e:
            print(f"❌ Utils module import failed: {e}")
        
        # Test performance module
        try:
            from src.parkingspace import performance
            print("✓ Performance module import successful")
        except Exception as e:
            print(f"❌ Performance module import failed: {e}")
        
        print("\n✓ Full diagnostic complete")
        return True
        
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

if __name__ == "__main__":
    success = full_diagnostic()
    exit(0 if success else 1)
