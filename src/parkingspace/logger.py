# src/parkingspace/logger.py

import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the application
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('parkingspace')
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = 'parkingspace') -> logging.Logger:
    """Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
