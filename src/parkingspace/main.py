# src/parkingspace/main.py

"""
Main application entry point for ParkingSpace Detection System.
Uses service-oriented architecture with integrated optimizations.
"""

import time
from .config import get_config
from .logger import setup_logging, get_logger
from .services import ParkingSpaceService
from .exceptions import ParkingSpaceError
from .capabilities import get_capability_detector, get_startup_optimizer


def main(config_file=None, video_file=None):
    """
    Main application entry point with integrated optimizations
    
    Args:
        config_file: Optional path to configuration file
        video_file: Optional path to video file to process
    """
    startup_start = time.time()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting ParkingSpace Detection System")
    
    try:
        # Get configuration
        config = get_config(config_file)
        if video_file:
            config.video.input_file = video_file
        
        # Get capability detection
        detector = get_capability_detector()
        capabilities = detector.detect_system_capabilities()
        _apply_capability_optimizations(config, capabilities)
        logger.info(f"⚡ Performance level: {capabilities.estimated_performance_level.upper()}")
        
        # Optimize PyTorch settings
        if capabilities:
            optimizer = get_startup_optimizer()
            optimizer.optimize_torch_settings(capabilities)
        
        logger.info(f"🖥️  Using device: {config.device}")
        
        # Initialize main service (now with integrated optimizations)
        parking_service = ParkingSpaceService(config)
        parking_service.initialize()
        
        # Log startup performance
        startup_time = time.time() - startup_start
        logger.info(f"✅ System ready in {startup_time:.3f}s")
        
        # Process video
        logger.info("🎬 Starting video processing...")
        parking_service.process_video(video_file)
        
        logger.info("🎯 ParkingSpace Detection System completed successfully")
        
    except ParkingSpaceError as e:
        logger.error(f"❌ Application error: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.info("⏹️  Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
def _apply_capability_optimizations(config, capabilities):
    """Apply system capability-based optimizations to configuration"""
    if not capabilities:
        return
        
    recommendations = capabilities.recommended_settings
    
    # Apply device optimization
    if recommendations.get("device") and hasattr(config, 'device'):
        config.device = recommendations["device"]
    
    # Apply processing optimizations
    if hasattr(config, 'processing'):
        if recommendations.get("processing_interval"):
            config.processing.interval_seconds = recommendations["processing_interval"]
    
    # Apply detection optimizations  
    if hasattr(config, 'detection'):
        if recommendations.get("image_size"):
            config.detection.image_size = recommendations["image_size"]
    
    # Apply performance optimizations
    if hasattr(config, 'performance'):
        if recommendations.get("enable_cuda_benchmark") is not None:
            config.performance.enable_cuda_benchmark = recommendations["enable_cuda_benchmark"]


if __name__ == "__main__":
    import sys
    
    # Simple command line argument parsing
    config_file = None
    video_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.json'):
            config_file = sys.argv[1]
        elif sys.argv[1].endswith(('.mp4', '.avi', '.mov')):
            video_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        if sys.argv[2].endswith('.json'):
            config_file = sys.argv[2]
        elif sys.argv[2].endswith(('.mp4', '.avi', '.mov')):
            video_file = sys.argv[2]
    
    main(config_file, video_file)
    import sys
    
    # Simple command line argument parsing
    config_file = None
    video_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.json'):
            config_file = sys.argv[1]
        elif sys.argv[1].endswith(('.mp4', '.avi', '.mov')):
            video_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        if sys.argv[2].endswith('.json'):
            config_file = sys.argv[2]
        elif sys.argv[2].endswith(('.mp4', '.avi', '.mov')):
            video_file = sys.argv[2]
    
    main(config_file, video_file)
