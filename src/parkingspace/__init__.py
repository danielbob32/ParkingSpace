"""ParkingSpace - Parking Space Detection using YOLO"""

from .main import main
from .pipeline import process_frame
from .regions import load_regions_from_file, save_regions_to_file, get_thresholds

__version__ = "0.1.0"
__all__ = [
    "main",
    "process_frame", 
    "load_regions_from_file",
    "save_regions_to_file",
    "get_thresholds"
]
