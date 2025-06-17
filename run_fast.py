#!/usr/bin/env python3
"""
Optimized ParkingSpace Detection launcher with fast startup enabled.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.parkingspace.main import main

if __name__ == "__main__":
    # Parse command line arguments
    config_file = None
    video_file = None
    
    for arg in sys.argv[1:]:
        if arg.endswith('.json'):
            config_file = arg
        elif arg.endswith(('.mp4', '.avi', '.mov')):
            video_file = arg
    
    # Run with fast startup enabled
    main(config_file, video_file, fast_mode=True)
