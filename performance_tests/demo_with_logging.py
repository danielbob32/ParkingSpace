#!/usr/bin/env python3
"""
Simple demo showing how to run the parking space detection with file logging enabled.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from parkingspace.logger import setup_logging
from parkingspace.main import main

def run_with_logging():
    """Run the main application with file logging enabled"""
    print("🎯 Running ParkingSpace Detection with File Logging")
    print("=" * 60)
    
    # Setup logging with file output
    log_file = "parkingspace_run.log"
    logger = setup_logging(log_file=log_file)
    
    print(f"📝 Logging to: {os.path.abspath(log_file)}")
    print("🚀 Starting detection (check the video window and log file)...")
    print("💡 Press 'q' in the video window to quit")
    print()
    
    try:
        # Run the main application
        main()
        
        print(f"\n✅ Application completed! Check the log file: {os.path.abspath(log_file)}")
        
        # Show log file contents
        if os.path.exists(log_file):
            print("\n📄 Log file contents:")
            print("-" * 40)
            with open(log_file, 'r') as f:
                content = f.read()
                print(content)
            print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        # Keep log file for inspection
        if os.path.exists(log_file):
            print(f"\n📋 Log file saved at: {os.path.abspath(log_file)}")
            print("   You can view it with: notepad parkingspace_run.log")

if __name__ == "__main__":
    run_with_logging()
