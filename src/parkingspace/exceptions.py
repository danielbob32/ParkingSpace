# src/parkingspace/exceptions.py

class ParkingSpaceError(Exception):
    """Base exception for ParkingSpace application"""
    pass

class ModelLoadError(ParkingSpaceError):
    """Raised when YOLO model fails to load"""
    pass

class RegionLoadError(ParkingSpaceError):
    """Raised when regions file fails to load"""
    pass

class ProcessingError(ParkingSpaceError):
    """Raised when frame processing fails"""
    pass

class ConfigurationError(ParkingSpaceError):
    """Raised when configuration is invalid"""
    pass
