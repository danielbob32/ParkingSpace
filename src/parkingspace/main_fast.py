# src/parkingspace/main_fast.py

"""
Fast startup version of the main application entry point.
Implements aggressive optimizations to minimize startup time.
"""

import time
import sys
import os
from pathlib import Path

# Enable fast startup optimizations immediately
from .fast_startup import enable_fast_startup

# Start fast startup manager
fast_startup = enable_fast_startup()

# Defer heavy imports using lazy loading
from .logger import setup_logging, get_logger


def main_fast(config_file=None, video_file=None):
    """
    Fast startup main application entry point
    
    Args:
        config_file: Optional path to configuration file
        video_file: Optional path to video file to process
    """
    startup_start = time.time()
    
    # Setup minimal logging first
    logger = setup_logging()
    logger.info("🚀 Starting ParkingSpace Detection System (Fast Mode)")
    
    try:
        # Start parallel operations immediately
        fast_startup.parallel_capability_detection()
        
        # Load configuration with timing
        config = fast_startup.time_operation(
            "Configuration Loading",
            lambda: _load_config_fast(config_file)
        )
        
        # Start model loading in parallel while we do other setup
        fast_startup.parallel_model_loading(config)
        
        # Do lightweight initialization
        regions = fast_startup.time_operation(
            "Region Loading",
            lambda: _load_regions_fast(config)
        )
        
        # Get parallel results
        capabilities = fast_startup.time_operation(
            "Capability Detection (Wait)",
            lambda: fast_startup.parallel_initializer.get_result("capability_detection")
        )
        
        if capabilities:
            _apply_capability_optimizations_fast(config, capabilities)
            logger.info(f"⚡ Performance level: {capabilities.estimated_performance_level.upper()}")
        
        # Initialize services with pre-loaded model
        parking_service = fast_startup.time_operation(
            "Service Initialization",
            lambda: _initialize_services_fast(config)
        )
        
        # Log startup performance
        startup_time = time.time() - startup_start
        logger.info(f"✅ System ready in {startup_time:.3f}s (Fast Mode)")
        fast_startup.log_startup_summary()
        
        # Process video
        logger.info("🎬 Starting video processing...")
        parking_service.process_video(video_file)
        
        logger.info("🎯 ParkingSpace Detection System completed successfully")
        
    except Exception as e:
        # Import exceptions only when needed
        from .exceptions import ParkingSpaceError
        
        if isinstance(e, ParkingSpaceError):
            logger.error(f"❌ Application error: {str(e)}")
        else:
            logger.error(f"💥 Unexpected error: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.info("⏹️  Application interrupted by user")


def _load_config_fast(config_file):
    """Fast configuration loading"""
    from .config import get_config
    return get_config(config_file)


def _load_regions_fast(config):
    """Fast region loading"""
    from .regions import load_regions_from_file
    
    try:
        return load_regions_from_file(config.regions_file)
    except Exception as e:
        fast_startup.logger.warning(f"⚠️  Region loading failed: {e}")
        return None


def _apply_capability_optimizations_fast(config, capabilities):
    """Apply capability optimizations without heavy operations"""
    recommendations = capabilities.recommended_settings
    
    # Apply lightweight optimizations only
    if recommendations.get("device") and hasattr(config, 'device'):
        config.device = recommendations["device"]
    
    if hasattr(config, 'processing'):
        if recommendations.get("processing_interval"):
            config.processing.interval_seconds = recommendations["processing_interval"]
    
    if hasattr(config, 'detection'):
        if recommendations.get("image_size"):
            config.detection.image_size = recommendations["image_size"]


def _initialize_services_fast(config):
    """Fast service initialization using parallel-loaded components"""
    from .services import ParkingSpaceService
    
    # Get pre-loaded model from parallel initialization
    model = fast_startup.parallel_initializer.get_result("model_loading", timeout=10.0)
    
    # Create service
    parking_service = ParkingSpaceService(config)
    
    # If model was pre-loaded, use it
    if model:
        parking_service.model_service.model = model
        parking_service.model_service._warmup_done = True
        fast_startup.logger.info("⚡ Using pre-loaded model")
    else:
        # Fallback to normal loading
        fast_startup.logger.info("🔄 Fallback to normal model loading")
        parking_service.model_service.load_model()
    
    # Initialize other services
    parking_service.region_service.load_regions()
    
    # Initialize timing
    parking_service.last_time_processed = time.time() - config.processing.interval_seconds
    
    return parking_service


# Compatibility function
def main(config_file=None, video_file=None):
    """Main entry point - automatically uses fast startup"""
    return main_fast(config_file, video_file)


if __name__ == "__main__":
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
    
    main_fast(config_file, video_file)
