# src/parkingspace/config.py

import os
import logging
from typing import Dict, Any

class Config:
    """Configuration management for ParkingSpace application"""
    
    def __init__(self):
        self.device = 'cuda' if self._is_cuda_available() else 'cpu'
        self.model_path = self._get_model_path()
        self.regions_file = 'regions.json'
        self.log_level = logging.INFO
        
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_model_path(self) -> str:
        """Get the path to the YOLO model"""
        possible_paths = ['yolo11x-seg.pt', 'yolo12n.pt']
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return 'yolo11x-seg.pt'  # Default fallback
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration parameters"""
        return {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.7,
            'max_detections': 300,
            'classes': [2, 3, 5, 7]  # car, motorcycle, bus, truck
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration"""
        return {
            'blur_kernel_size': (5, 5),
            'morphology_kernel_size': (3, 3),
            'contour_min_area': 100
        }

def get_config() -> Config:
    """Get application configuration instance"""
    return Config()
