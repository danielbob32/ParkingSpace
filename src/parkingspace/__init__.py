"""ParkingSpace - Parking Space Detection using YOLO"""

from .main import main
from .pipeline import process_frame
from .regions import load_regions_from_file, save_regions_to_file, get_thresholds
from .utils import get_contour_center
from .config import get_config
from .logger import setup_logging, get_logger
from .performance import get_performance_monitor
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
    "setup_logging",
    "get_logger",
    "get_performance_monitor",
    "ParkingSpaceError",
    "ModelLoadError",
    "RegionLoadError",
    "ProcessingError",
    "ConfigurationError"
]
