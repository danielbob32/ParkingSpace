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
from .fast_startup import (
    get_fast_startup_manager,
    LazyImporter,
    ModelCache,
    ParallelInitializer,
    FastStartupManager
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

__version__ = "0.1.0"
__all__ = [
    "main",
    "process_frame", 
    "load_regions_from_file",
    "save_regions_to_file",
    "get_thresholds",
    "get_contour_center",
    "get_config",
    "Config",
    "setup_logging",
    "get_logger",
    "get_performance_monitor",    "get_capability_detector",
    "get_startup_optimizer",
    "SystemCapabilities",
    "OptimizationProfile",
    "CapabilityDetector",
    "StartupOptimizer",
    "get_fast_startup_manager",
    "LazyImporter",
    "ModelCache",
    "ParallelInitializer",
    "FastStartupManager",
    "ParkingSpaceService",
    "ModelService",
    "RegionService", 
    "ProcessingService",
    "VideoService",
    "DetectionResult",
    "FrameProcessingResult",
    "ParkingSpaceError",
    "ModelLoadError",
    "RegionLoadError",
    "ProcessingError",
    "ConfigurationError"
]
