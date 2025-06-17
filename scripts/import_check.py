#!/usr/bin/env python3
"""
Import check script to verify all imports work correctly.
"""

import sys
import os

def main():
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("=== IMPORT CHECK ===")
    
    try:
        import src.parkingspace
        print("✓ Main package import successful")
        
        # Test specific module imports
        from src.parkingspace import main
        print("✓ Main module import successful")
        
        from src.parkingspace import pipeline
        print("✓ Pipeline module import successful")
        
        from src.parkingspace import regions
        print("✓ Regions module import successful")
        
        from src.parkingspace import utils
        print("✓ Utils module import successful")
        
        from src.parkingspace import config
        print("✓ Config module import successful")
        
        from src.parkingspace import logger
        print("✓ Logger module import successful")
        
        from src.parkingspace import performance
        print("✓ Performance module import successful")
        
        from src.parkingspace import exceptions
        print("✓ Exceptions module import successful")
        
        print("All imports successful - System ready for development")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
