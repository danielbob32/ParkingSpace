# src/parkingspace/capabilities.py

"""
System capability detection and optimization for ParkingSpace Detection System.
Provides hardware detection, performance profiling, and automatic optimization.
"""

import os
import sys
import time
import psutil
import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .logger import get_logger


@dataclass
class SystemCapabilities:
    """System capabilities and hardware information"""
    # Hardware
    cpu_count: int
    cpu_brand: str
    total_memory_gb: float
    
    # GPU
    has_cuda: bool
    cuda_devices: List[str]
    gpu_memory_gb: Optional[float]
    
    # System
    platform: str
    python_version: str
    
    # Performance
    estimated_performance_level: str  # "low", "medium", "high", "ultra"
    recommended_settings: Dict[str, any]


@dataclass
class OptimizationProfile:
    """Optimization profile based on system capabilities"""
    # Model settings
    model_size: str  # "nano", "small", "medium", "large", "xlarge"
    image_size: Tuple[int, int]
    batch_size: int
    
    # Processing settings
    processing_interval: float
    enable_multithreading: bool
    use_half_precision: bool
    
    # Memory settings
    cache_frames: bool
    preload_model: bool
    
    # Display settings
    display_fps: int
    display_scale: float


class CapabilityDetector:
    """Detects system capabilities and recommends optimizations"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def detect_system_capabilities(self) -> SystemCapabilities:
        """Detect and analyze system capabilities"""
        self.logger.info("🔍 Detecting system capabilities...")
        
        # CPU Information
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_brand = self._get_cpu_brand()
        
        # Memory Information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        # GPU Information
        has_cuda, cuda_devices, gpu_memory_gb = self._detect_gpu()
        
        # System Information
        platform_info = f"{platform.system()} {platform.release()}"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Performance Level Estimation
        performance_level = self._estimate_performance_level(
            cpu_count, total_memory_gb, has_cuda, gpu_memory_gb
        )
        
        # Recommended Settings
        recommended_settings = self._generate_recommendations(
            performance_level, has_cuda, total_memory_gb, cpu_count
        )
        
        capabilities = SystemCapabilities(
            cpu_count=cpu_count,
            cpu_brand=cpu_brand,
            total_memory_gb=total_memory_gb,
            has_cuda=has_cuda,
            cuda_devices=cuda_devices,
            gpu_memory_gb=gpu_memory_gb,
            platform=platform_info,
            python_version=python_version,
            estimated_performance_level=performance_level,
            recommended_settings=recommended_settings
        )
        
        self._log_capabilities(capabilities)
        return capabilities
    
    def _get_cpu_brand(self) -> str:
        """Get CPU brand information"""
        try:
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'], 
                    capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':', 1)[1].strip()
        except:
            pass
        return "Unknown CPU"
    
    def _detect_gpu(self) -> Tuple[bool, List[str], Optional[float]]:
        """Detect GPU capabilities"""
        cuda_devices = []
        gpu_memory_gb = None
        has_cuda = False
        
        try:
            import torch
            if torch.cuda.is_available():
                has_cuda = True
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    cuda_devices.append(f"GPU {i}: {device_name}")
                    
                    # Get memory info for the first device
                    if i == 0:
                        memory_bytes = torch.cuda.get_device_properties(i).total_memory
                        gpu_memory_gb = memory_bytes / (1024**3)
                        
                self.logger.info(f"✅ CUDA available with {device_count} device(s)")
            else:
                self.logger.info("❌ CUDA not available")
                
        except ImportError:
            self.logger.warning("🔶 PyTorch not available - GPU detection skipped")
        except Exception as e:
            self.logger.warning(f"🔶 GPU detection failed: {e}")
            
        return has_cuda, cuda_devices, gpu_memory_gb
    
    def _estimate_performance_level(self, cpu_count: int, memory_gb: float, 
                                  has_cuda: bool, gpu_memory_gb: Optional[float]) -> str:
        """Estimate system performance level"""
        score = 0
        
        # CPU score
        if cpu_count >= 8:
            score += 3
        elif cpu_count >= 4:
            score += 2
        else:
            score += 1
            
        # Memory score
        if memory_gb >= 16:
            score += 3
        elif memory_gb >= 8:
            score += 2
        else:
            score += 1
            
        # GPU score
        if has_cuda and gpu_memory_gb:
            if gpu_memory_gb >= 8:
                score += 4
            elif gpu_memory_gb >= 4:
                score += 3
            elif gpu_memory_gb >= 2:
                score += 2
            else:
                score += 1
        
        # Determine performance level
        if score >= 9:
            return "ultra"
        elif score >= 7:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, performance_level: str, has_cuda: bool, 
                                memory_gb: float, cpu_count: int) -> Dict[str, any]:
        """Generate optimization recommendations based on capabilities"""
        recommendations = {
            "device": "cuda" if has_cuda else "cpu",
            "enable_cuda_benchmark": has_cuda,
            "use_half_precision": has_cuda and performance_level in ["high", "ultra"],
            "enable_multithreading": cpu_count >= 4,
            "worker_threads": min(cpu_count, 4) if cpu_count >= 4 else 1,
        }
        
        # Performance-based settings
        if performance_level == "ultra":
            recommendations.update({
                "model_size": "xlarge",
                "image_size": (1088, 1920),
                "batch_size": 4,
                "processing_interval": 1.0,
                "cache_frames": True,
                "display_fps": 30,
                "display_scale": 1.0,
            })
        elif performance_level == "high":
            recommendations.update({
                "model_size": "large",
                "image_size": (896, 1600),
                "batch_size": 2,
                "processing_interval": 2.0,
                "cache_frames": True,
                "display_fps": 25,
                "display_scale": 1.0,
            })
        elif performance_level == "medium":
            recommendations.update({
                "model_size": "medium",
                "image_size": (640, 1120),
                "batch_size": 1,
                "processing_interval": 3.0,
                "cache_frames": False,
                "display_fps": 20,
                "display_scale": 0.8,
            })
        else:  # low
            recommendations.update({
                "model_size": "small",
                "image_size": (480, 864),
                "batch_size": 1,
                "processing_interval": 5.0,
                "cache_frames": False,
                "display_fps": 15,
                "display_scale": 0.6,
            })
            
        return recommendations
    
    def _log_capabilities(self, capabilities: SystemCapabilities):
        """Log detected capabilities"""
        self.logger.info("🖥️  System Capabilities Detected:")
        self.logger.info(f"  • Platform: {capabilities.platform}")
        self.logger.info(f"  • Python: {capabilities.python_version}")
        self.logger.info(f"  • CPU: {capabilities.cpu_brand} ({capabilities.cpu_count} cores)")
        self.logger.info(f"  • Memory: {capabilities.total_memory_gb:.1f} GB")
        
        if capabilities.has_cuda:
            self.logger.info(f"  • GPU: ✅ CUDA Available ({capabilities.gpu_memory_gb:.1f} GB VRAM)")
            for device in capabilities.cuda_devices:
                self.logger.info(f"    - {device}")
        else:
            self.logger.info("  • GPU: ❌ CUDA Not Available")
            
        self.logger.info(f"  • Performance Level: {capabilities.estimated_performance_level.upper()}")
        
        # Log key recommendations
        rec = capabilities.recommended_settings
        self.logger.info("⚙️  Optimization Recommendations:")
        self.logger.info(f"  • Device: {rec['device']}")
        self.logger.info(f"  • Model Size: {rec['model_size']}")
        self.logger.info(f"  • Image Size: {rec['image_size']}")
        self.logger.info(f"  • Processing Interval: {rec['processing_interval']}s")
        if rec.get('use_half_precision'):
            self.logger.info("  • Half Precision: ✅ Enabled (FP16)")
        if rec.get('enable_multithreading'):
            self.logger.info(f"  • Multithreading: ✅ {rec['worker_threads']} threads")


class StartupOptimizer:
    """Optimizes application startup time"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.startup_times = {}
        
    def profile_startup(self, func, name: str):
        """Profile a startup function and log timing"""
        start_time = time.time()
        result = func()
        elapsed = time.time() - start_time
        
        self.startup_times[name] = elapsed
        self.logger.info(f"⏱️  {name}: {elapsed:.3f}s")
        
        return result
        
    def preload_imports(self):
        """Preload heavy imports to reduce startup time"""
        self.logger.info("📦 Preloading imports...")
        
        import_start = time.time()
        
        # Preload heavy libraries
        try:
            import torch
            import torchvision
            import ultralytics
            import cv2
            import numpy as np
            
            # Warm up torch if CUDA is available
            if torch.cuda.is_available():
                dummy_tensor = torch.randn(1, 3, 224, 224, device='cuda')
                _ = dummy_tensor * 2  # Simple operation to warm up GPU
                del dummy_tensor
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.warning(f"Import preloading failed: {e}")
            
        elapsed = time.time() - import_start
        self.logger.info(f"📦 Import preloading completed: {elapsed:.3f}s")
        
    def optimize_torch_settings(self, capabilities: SystemCapabilities):
        """Optimize PyTorch settings based on capabilities"""
        try:
            import torch
            
            # Set number of threads for CPU operations
            if capabilities.cpu_count >= 4:
                num_threads = min(capabilities.cpu_count, 8)
                torch.set_num_threads(num_threads)
                self.logger.info(f"🔧 PyTorch threads: {num_threads}")
            
            # Enable optimizations
            if capabilities.has_cuda:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.logger.info("🚀 CUDA optimizations enabled")
                
            # Memory management
            if capabilities.has_cuda and capabilities.gpu_memory_gb and capabilities.gpu_memory_gb < 4:
                # Conservative memory settings for low VRAM
                torch.cuda.empty_cache()
                self.logger.info("💾 Conservative GPU memory settings applied")
                
        except Exception as e:
            self.logger.warning(f"PyTorch optimization failed: {e}")
    
    def log_startup_summary(self):
        """Log startup performance summary"""
        if not self.startup_times:
            return
            
        total_time = sum(self.startup_times.values())
        self.logger.info("🚀 Startup Performance Summary:")
        self.logger.info(f"  • Total startup time: {total_time:.3f}s")
        
        for name, time_taken in sorted(self.startup_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_taken / total_time) * 100
            self.logger.info(f"  • {name}: {time_taken:.3f}s ({percentage:.1f}%)")


# Global instances
_capability_detector = None
_startup_optimizer = None


def get_capability_detector() -> CapabilityDetector:
    """Get global capability detector instance"""
    global _capability_detector
    if _capability_detector is None:
        _capability_detector = CapabilityDetector()
    return _capability_detector


def get_startup_optimizer() -> StartupOptimizer:
    """Get global startup optimizer instance"""
    global _startup_optimizer
    if _startup_optimizer is None:
        _startup_optimizer = StartupOptimizer()
    return _startup_optimizer
