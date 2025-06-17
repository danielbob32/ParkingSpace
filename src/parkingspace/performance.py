# src/parkingspace/performance.py

import time
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from .logger import get_logger

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    frame_processing_time: float
    detection_time: float
    memory_usage: float
    cpu_usage: float
    gpu_memory: Optional[float] = None

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = None
        
    def start_frame_processing(self):
        """Start timing frame processing"""
        self.start_time = time.time()
        
    def end_frame_processing(self, detection_time: float = 0.0) -> PerformanceMetrics:
        """End timing and record performance metrics
        
        Args:
            detection_time: Time spent on detection (seconds)
            
        Returns:
            Performance metrics for this frame
        """
        if self.start_time is None:
            raise ValueError("Must call start_frame_processing() first")
            
        processing_time = time.time() - self.start_time
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        # Get GPU memory if available
        gpu_memory = None
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if reserved > 0:
                    gpu_memory = allocated / reserved * 100
                else:
                    gpu_memory = 0.0
        except (ImportError, AttributeError):
            pass
            
        metrics = PerformanceMetrics(
            frame_processing_time=processing_time,
            detection_time=detection_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            gpu_memory=gpu_memory
        )
        
        self.metrics_history.append(metrics)
        self.start_time = None
        
        return metrics
    
    def get_average_metrics(self, last_n: int = 100) -> Optional[PerformanceMetrics]:
        """Get average performance metrics over last N frames
        
        Args:
            last_n: Number of recent frames to average
            
        Returns:
            Average performance metrics or None if no data
        """
        if not self.metrics_history:
            return None
            
        recent_metrics = self.metrics_history[-last_n:]
        
        avg_processing_time = sum(m.frame_processing_time for m in recent_metrics) / len(recent_metrics)
        avg_detection_time = sum(m.detection_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        
        avg_gpu = None
        gpu_metrics = [m.gpu_memory for m in recent_metrics if m.gpu_memory is not None]
        if gpu_metrics:
            avg_gpu = sum(gpu_metrics) / len(gpu_metrics)
        
        return PerformanceMetrics(
            frame_processing_time=avg_processing_time,
            detection_time=avg_detection_time,
            memory_usage=avg_memory,
            cpu_usage=avg_cpu,
            gpu_memory=avg_gpu
        )
    
    def log_performance_report(self, last_n: int = 100):
        """Log performance report
        
        Args:
            last_n: Number of recent frames to include in report
        """
        avg_metrics = self.get_average_metrics(last_n)
        if avg_metrics is None:
            self.logger.info("No performance data available")
            return
            
        self.logger.info("Performance Report (last %d frames):", last_n)
        self.logger.info("  Average frame processing time: %.3f seconds", avg_metrics.frame_processing_time)
        self.logger.info("  Average detection time: %.3f seconds", avg_metrics.detection_time)
        self.logger.info("  Average memory usage: %.1f%%", avg_metrics.memory_usage)
        self.logger.info("  Average CPU usage: %.1f%%", avg_metrics.cpu_usage)
        
        if avg_metrics.gpu_memory is not None:
            self.logger.info("  Average GPU memory usage: %.1f%%", avg_metrics.gpu_memory)
        
        fps = 1.0 / avg_metrics.frame_processing_time if avg_metrics.frame_processing_time > 0 else 0
        self.logger.info("  Estimated FPS: %.1f", fps)

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
