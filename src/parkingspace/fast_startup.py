# src/parkingspace/fast_startup.py

"""
Fast startup optimizations for ParkingSpace Detection System.
Implements lazy loading, caching, and parallel initialization strategies.
"""

import os
import sys
import time
import threading
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logger import get_logger


class LazyImporter:
    """Lazy import system to delay heavy imports until needed"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._imported_modules = {}
        self._import_times = {}
    
    def lazy_import(self, module_name: str, attribute: Optional[str] = None):
        """Lazy import a module or attribute"""
        key = f"{module_name}.{attribute}" if attribute else module_name
        
        if key not in self._imported_modules:
            start_time = time.time()
            
            try:
                module = __import__(module_name, fromlist=[attribute] if attribute else [])
                if attribute:
                    self._imported_modules[key] = getattr(module, attribute)
                else:
                    self._imported_modules[key] = module
                    
                import_time = time.time() - start_time
                self._import_times[key] = import_time
                
                if import_time > 0.1:  # Only log slow imports
                    self.logger.debug(f"📦 Lazy imported {key}: {import_time:.3f}s")
                    
            except Exception as e:
                self.logger.warning(f"❌ Failed to lazy import {key}: {e}")
                return None
        
        return self._imported_modules.get(key)
    
    def get_import_summary(self) -> Dict[str, float]:
        """Get summary of import times"""
        return self._import_times.copy()


class ModelCache:
    """Cache system for model states and configurations"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
    
    def _get_cache_key(self, model_path: str, device: str, image_size: tuple) -> str:
        """Generate cache key for model configuration"""
        # Include file modification time for cache invalidation
        try:
            mtime = os.path.getmtime(model_path)
            cache_data = f"{model_path}_{device}_{image_size}_{mtime}"
            return hashlib.md5(cache_data.encode()).hexdigest()
        except:
            return hashlib.md5(f"{model_path}_{device}_{image_size}".encode()).hexdigest()
    
    def get_model_cache_path(self, model_path: str, device: str, image_size: tuple) -> Path:
        """Get path to cached model state"""
        cache_key = self._get_cache_key(model_path, device, image_size)
        return self.cache_dir / f"model_{cache_key}.cache"
    
    def save_model_state(self, model_path: str, device: str, image_size: tuple, 
                        model_state: Dict[str, Any]):
        """Save model state to cache"""
        try:
            cache_path = self.get_model_cache_path(model_path, device, image_size)
            with open(cache_path, 'wb') as f:
                pickle.dump(model_state, f)
            self.logger.info(f"💾 Model state cached: {cache_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to cache model state: {e}")
    
    def load_model_state(self, model_path: str, device: str, image_size: tuple) -> Optional[Dict[str, Any]]:
        """Load model state from cache"""
        try:
            cache_path = self.get_model_cache_path(model_path, device, image_size)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    model_state = pickle.load(f)
                self.logger.info(f"⚡ Model state loaded from cache: {cache_path.name}")
                return model_state
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load cached model state: {e}")
        return None
    
    def clear_cache(self):
        """Clear all cached model states"""
        try:
            for cache_file in self.cache_dir.glob("model_*.cache"):
                cache_file.unlink()
            self.logger.info("🧹 Model cache cleared")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to clear cache: {e}")


class ParallelInitializer:
    """Parallel initialization system for faster startup"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.logger = get_logger(__name__)
        self._futures = {}
        self._results = {}
        
    def submit_task(self, name: str, func: Callable, *args, **kwargs):
        """Submit a task for parallel execution"""
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        future = executor.submit(func, *args, **kwargs)
        self._futures[name] = (future, executor)
        self.logger.debug(f"🚀 Submitted parallel task: {name}")
    
    def get_result(self, name: str, timeout: float = 30.0):
        """Get result from parallel task"""
        if name in self._results:
            return self._results[name]
            
        if name in self._futures:
            future, executor = self._futures[name]
            try:
                start_time = time.time()
                result = future.result(timeout=timeout)
                elapsed = time.time() - start_time
                
                self._results[name] = result
                self.logger.debug(f"✅ Parallel task completed: {name} ({elapsed:.3f}s)")
                
                executor.shutdown(wait=False)
                del self._futures[name]
                
                return result
            except Exception as e:
                self.logger.error(f"❌ Parallel task failed: {name} - {e}")
                executor.shutdown(wait=False)
                del self._futures[name]
                return None
        
        return None
    
    def wait_all(self, timeout: float = 30.0):
        """Wait for all parallel tasks to complete"""
        completed_tasks = []
        
        for name, (future, executor) in self._futures.items():
            try:
                result = future.result(timeout=timeout)
                self._results[name] = result
                completed_tasks.append(name)
                executor.shutdown(wait=False)
            except Exception as e:
                self.logger.error(f"❌ Parallel task failed: {name} - {e}")
                executor.shutdown(wait=False)
        
        for name in completed_tasks:
            del self._futures[name]


class FastStartupManager:
    """Main fast startup manager that coordinates all optimizations"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.lazy_importer = LazyImporter()
        self.model_cache = ModelCache()
        self.parallel_initializer = ParallelInitializer()
        self.startup_times = {}
        
    def time_operation(self, name: str, func: Callable, *args, **kwargs):
        """Time an operation and log the result"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        self.startup_times[name] = elapsed
        self.logger.info(f"⏱️  {name}: {elapsed:.3f}s")
        
        return result
    
    def optimize_imports(self):
        """Optimize heavy imports using lazy loading"""
        self.logger.info("📦 Optimizing imports with lazy loading...")
        
        # Pre-import only essential modules
        essential_imports = [
            'os', 'sys', 'time', 'logging', 'json', 'pathlib'
        ]
        
        for module in essential_imports:
            self.lazy_importer.lazy_import(module)
        
        # Defer heavy imports
        self.logger.info("🔄 Heavy imports will be loaded on demand")
    
    def parallel_model_loading(self, config):
        """Start model loading in parallel"""
        def load_model_async():
            # Import heavy modules only when needed
            torch = self.lazy_importer.lazy_import('torch')
            YOLO = self.lazy_importer.lazy_import('ultralytics', 'YOLO')
            
            if torch and YOLO:
                model = YOLO(config.model_path)
                model.to(config.device)
                return model
            return None
        
        self.parallel_initializer.submit_task("model_loading", load_model_async)
        self.logger.info("🚀 Model loading started in parallel")
    
    def parallel_capability_detection(self):
        """Start capability detection in parallel"""
        def detect_capabilities_async():
            from .capabilities import get_capability_detector
            detector = get_capability_detector()
            return detector.detect_system_capabilities()
        
        self.parallel_initializer.submit_task("capability_detection", detect_capabilities_async)
        self.logger.info("🔍 Capability detection started in parallel")
    
    def get_startup_summary(self) -> Dict[str, float]:
        """Get startup timing summary"""
        total_time = sum(self.startup_times.values())
        summary = {
            'total_time': total_time,
            'operations': self.startup_times.copy()
        }
        
        # Add import times
        import_times = self.lazy_importer.get_import_summary()
        if import_times:
            summary['import_times'] = import_times
            summary['total_import_time'] = sum(import_times.values())
        
        return summary
    
    def log_startup_summary(self):
        """Log detailed startup performance summary"""
        summary = self.get_startup_summary()
        total_time = summary['total_time']
        
        self.logger.info("🏁 Fast Startup Summary:")
        self.logger.info(f"  • Total startup time: {total_time:.3f}s")
        
        # Show operations sorted by time
        operations = sorted(summary['operations'].items(), key=lambda x: x[1], reverse=True)
        for name, time_taken in operations:
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            self.logger.info(f"  • {name}: {time_taken:.3f}s ({percentage:.1f}%)")
        
        # Show import times if available
        if 'import_times' in summary:
            import_total = summary['total_import_time']
            self.logger.info(f"  • Total import time: {import_total:.3f}s")


# Global instance
_fast_startup_manager = None


def get_fast_startup_manager() -> FastStartupManager:
    """Get global fast startup manager instance"""
    global _fast_startup_manager
    if _fast_startup_manager is None:
        _fast_startup_manager = FastStartupManager()
    return _fast_startup_manager


def enable_fast_startup():
    """Enable fast startup optimizations"""
    manager = get_fast_startup_manager()
    manager.optimize_imports()
    return manager
