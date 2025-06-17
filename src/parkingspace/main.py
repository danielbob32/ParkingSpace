# src/parkingspace/main.py

"""
Main application entry point for ParkingSpace Detection System.
Uses service-oriented architecture for better separation of concerns.
"""

from .config import get_config
from .logger import setup_logging, get_logger
from .services import ParkingSpaceService
from .exceptions import ParkingSpaceError


def main(config_file=None, video_file=None):
    """
    Main application entry point
    
    Args:
        config_file: Optional path to configuration file
        video_file: Optional path to video file to process
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting ParkingSpace Detection System")
    
    try:
        # Get configuration
        config = get_config(config_file)
        logger.info(f"Using device: {config.device}")
        
        # Initialize main service
        parking_service = ParkingSpaceService(config)
        parking_service.initialize()
        
        # Process video
        parking_service.process_video(video_file)
        
        logger.info("ParkingSpace Detection System completed successfully")
        
    except ParkingSpaceError as e:
        logger.error(f"Application error: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


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
