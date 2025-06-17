#!/usr/bin/env python3
"""
Timing analysis - shows exactly when each phase happens
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def analyze_startup_phases():
    """Analyze each phase of startup with precise timing"""
    print("🔍 ANALYZING STARTUP PHASES")
    print("=" * 50)
    
    total_start = time.time()
    
    # Phase 1: Basic imports
    print("Phase 1: Basic imports...")
    import_start = time.time()
    
    import cv2
    import numpy as np
    from src.parkingspace.config import get_config
    from src.parkingspace.logger import setup_logging, get_logger
    
    import_time = time.time() - import_start
    print(f"  ✅ Basic imports: {import_time:.3f}s")
    
    # Phase 2: OpenCV window
    print("Phase 2: OpenCV window display...")
    window_start = time.time()
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "ParkingSpace Detection", (80, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.namedWindow("Parking Spaces", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Parking Spaces", frame)
    cv2.waitKey(1)
    
    window_time = time.time() - window_start
    print(f"  ✅ OpenCV window: {window_time:.3f}s")
    
    # Phase 3: Heavy imports
    print("Phase 3: Service imports...")
    heavy_import_start = time.time()
    
    from src.parkingspace.services import ParkingSpaceService
    
    heavy_import_time = time.time() - heavy_import_start
    print(f"  ✅ Service imports: {heavy_import_time:.3f}s")
    
    # Phase 4: Config and service creation
    print("Phase 4: Configuration and service creation...")
    config_start = time.time()
    
    logger = setup_logging()
    config = get_config()
    parking_service = ParkingSpaceService(config)
    
    config_time = time.time() - config_start
    print(f"  ✅ Config & service: {config_time:.3f}s")
    
    # Phase 5: Service initialization (model loading)
    print("Phase 5: Service initialization...")
    init_start = time.time()
    
    # Update window
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Loading AI models...", (120, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Parking Spaces", frame)
    cv2.waitKey(1)
    
    parking_service.initialize()
    
    init_time = time.time() - init_start
    print(f"  ✅ Service initialization: {init_time:.3f}s")
    
    system_ready_time = time.time() - total_start
    print(f"📊 SYSTEM READY: {system_ready_time:.3f}s")
    
    # Phase 6: Video opening
    print("Phase 6: Video opening...")
    video_start = time.time()
    
    parking_service.video_service.open_video("Demo/exp1.mp4")
    ret, first_frame = parking_service.video_service.read_frame()
    
    video_time = time.time() - video_start
    print(f"  ✅ Video opened: {video_time:.3f}s")
    
    # Phase 7: First AI detection (THE BIG DELAY)
    print("Phase 7: FIRST AI DETECTION...")
    detection_start = time.time()
    
    # Update window
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Running AI detection...", (100, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Parking Spaces", frame)
    cv2.waitKey(1)
    
    detection_result = parking_service.model_service.detect_vehicles(first_frame)
    processing_result = parking_service.processing_service.process_frame(
        first_frame, detection_result.vehicle_mask
    )
    
    detection_time = time.time() - detection_start
    print(f"  ✅ First AI detection: {detection_time:.3f}s")
    print(f"      ^ THIS IS THE REAL BOTTLENECK!")
    
    # Show actual result
    cv2.imshow("Parking Spaces", processing_result.result_image)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()
    
    actual_video_time = time.time() - total_start
    print(f"🎬 ACTUAL VIDEO DISPLAY: {actual_video_time:.3f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TIMING BREAKDOWN:")
    print(f"  OpenCV window:     {window_time:.3f}s")
    print(f"  System ready:      {system_ready_time:.3f}s") 
    print(f"  First detection:   {detection_time:.3f}s ← BOTTLENECK")
    print(f"  Actual video:      {actual_video_time:.3f}s")
    
    print(f"\n💡 INSIGHT:")
    print(f"  The delay between 'Starting video processing' and actual")
    print(f"  video display is {detection_time:.3f}s due to first AI detection!")
    
    return {
        'window': window_time,
        'system_ready': system_ready_time,
        'first_detection': detection_time,
        'actual_video': actual_video_time
    }

if __name__ == "__main__":
    analyze_startup_phases()
