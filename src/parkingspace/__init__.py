"""ParkingSpace - Parking Space Detection using YOLO"""

from .main import main
from .pipeline import process_frame
from .regions import load_regions_from_file, save_regions_to_file, get_thresholds
from .utils import get_contour_center
from .config import get_config, Config
from .logger import setup_logging, get_logger
from .performance import get_performance_monitor
from .capabilities import (
    get_capability_detector,
    get_startup_optimizer,
    SystemCapabilities,
    OptimizationProfile,
    CapabilityDetector,
    StartupOptimizer
)
from .services import (
    ParkingSpaceService,
    ModelService,
    RegionService,
    ProcessingService,
    VideoService,
    DetectionResult,
    FrameProcessingResult
)
from .exceptions import (
    ParkingSpaceError,
    ModelLoadError,
    RegionLoadError,
    ProcessingError,
    ConfigurationError
)

__version__ = "1.0.0"
__author__ = "ParkingSpace Team"

__all__ = [
    # Main entry point
    "main",
    
    # Core functionality
    "process_frame",
    "get_contour_center",
    
    # Configuration
    "get_config",
    "Config",
    
    # Logging and monitoring
    "setup_logging",
    "get_logger", 
    "get_performance_monitor",
    
    # Regions
    "load_regions_from_file",
    "save_regions_to_file",
    "get_thresholds",
    
    # Capabilities
    "get_capability_detector",
    "get_startup_optimizer",
    "SystemCapabilities",
    "OptimizationProfile",
    "CapabilityDetector", 
    "StartupOptimizer",
    
    # Services
    "ParkingSpaceService",
    "ModelService",
    "RegionService", 
    "ProcessingService",
    "VideoService",
    "DetectionResult",
    "FrameProcessingResult",
      # Exceptions
    "ParkingSpaceError",
    "ModelLoadError",
    "RegionLoadError", 
    "ProcessingError",
    "ConfigurationError",
]
