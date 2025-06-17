# src/parkingspace/services.py

"""
Service layer for ParkingSpace application.
Implements business logic and coordinates between different components.
"""

import time
import cv2
import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from ultralytics import YOLO

from .config import Config
from .pipeline import process_frame
from .regions import load_regions_from_file, get_thresholds
from .logger import get_logger
from .performance import get_performance_monitor
from .exceptions import ModelLoadError, RegionLoadError, ProcessingError


@dataclass
class DetectionResult:
    """Result from vehicle detection"""
    vehicle_mask: np.ndarray
    detection_time: float
    vehicle_count: int


@dataclass
class FrameProcessingResult:
    """Result from frame processing"""
    result_image: np.ndarray
    empty_spaces: int
    parking_space_boxes: np.ndarray
    processing_time: float


class ModelService:
    """Service for managing YOLO model operations with optimizations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.model: Optional[YOLO] = None
        self._warmup_done = False
        
    def load_model(self) -> None:
        """Load and initialize the YOLO model with optimizations"""
        load_start = time.time()
        
        try:
            self.logger.info(f"📥 Loading YOLO model: {self.config.model_path}")
            
            # Load model
            self.model = YOLO(self.config.model_path)
            self.model.to(self.config.device)
            
            # Apply optimizations
            self._apply_model_optimizations()
            
            # Warm up model for better performance
            self._warmup_model()
            
            load_time = time.time() - load_start
            self.logger.info(f"✅ YOLO model loaded and optimized in {load_time:.3f}s")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
    
    def load_model_fast(self, pre_loaded_model=None) -> None:
        """Fast model loading with optional pre-loaded model"""
        if pre_loaded_model:
            self.logger.info("⚡ Using pre-loaded model")
            self.model = pre_loaded_model
            self._apply_model_optimizations()
            # Skip warmup since model is already loaded
            self._warmup_done = True
            return
        
        # Fallback to normal loading
        self.load_model()
    
    def preload_model_async(self):
        """Preload model asynchronously for faster startup"""
        import threading
        
        def load_async():
            try:
                self.logger.info("🚀 Pre-loading model in background...")
                self.load_model()
            except Exception as e:
                self.logger.warning(f"⚠️  Background model loading failed: {e}")
        
        thread = threading.Thread(target=load_async)
        thread.daemon = True
        thread.start()
        return thread
    
    def _apply_model_optimizations(self):
        """Apply various optimizations to the model"""
        if self.config.device == 'cuda':
            # Enable CUDA optimizations
            if self.config.performance.enable_cuda_benchmark:
                torch.backends.cudnn.benchmark = True
                self.logger.info("🚀 CUDA benchmarking enabled")
            
            # Try to use half precision if supported
            try:
                if hasattr(self.model.model, 'half'):
                    self.model.model.half()
                    self.logger.info("🔄 Half precision (FP16) enabled")
            except Exception as e:
                self.logger.warning(f"⚠️  Half precision failed: {e}")
        
        # Set model to evaluation mode for inference
        if hasattr(self.model.model, 'eval'):
            self.model.model.eval()
    
    def _warmup_model(self):
        """Warm up the model with a dummy inference for better performance"""
        if self._warmup_done:
            return
            
        try:
            self.logger.info("🔥 Warming up model...")
            warmup_start = time.time()
            
            # Create dummy input matching expected input size
            h, w = self.config.detection.image_size
            dummy_input = torch.randn(1, 3, h, w, device=self.config.device)
            
            # Convert to numpy for YOLO (it expects numpy arrays)
            dummy_array = (dummy_input.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)[0]
            
            # Run dummy inference
            with torch.no_grad():
                _ = self.model(dummy_array, verbose=False)
            
            # Clear GPU cache
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
            
            warmup_time = time.time() - warmup_start
            self.logger.info(f"🔥 Model warmup completed in {warmup_time:.3f}s")
            self._warmup_done = True
            
        except Exception as e:
            self.logger.warning(f"⚠️  Model warmup failed: {e}")
    
    def detect_vehicles(self, frame: np.ndarray) -> DetectionResult:
        """Detect vehicles in the frame with optimizations"""
        if self.model is None:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
            
        detection_start = time.time()
        
        # Run YOLO detection with optimized settings
        result = self.model(
            frame,
            conf=self.config.detection.confidence_threshold,
            classes=self.config.detection.classes,
            imgsz=self.config.detection.image_size,
            verbose=False,  # Reduce output verbosity
            device=self.config.device
        )
        
        detection_time = time.time() - detection_start
        
        # Process results efficiently
        r = result[0]
        masks = r.masks
        class_ids = r.boxes.cls if r.boxes is not None else []
        
        # Pre-allocate mask for better performance
        frame_h, frame_w = frame.shape[:2]
        vehicle_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        vehicle_count = 0
        
        if masks is not None and len(masks.data) > 0:
            # Process masks in batch for better performance
            for mask, cls in zip(masks.data, class_ids):
                class_id = int(cls)
                if class_id in self.config.detection.classes:
                    # Convert mask efficiently
                    binary_mask = mask.cpu().numpy().astype(np.uint8)
                    
                    # Resize mask if needed
                    if binary_mask.shape != (frame_h, frame_w):
                        binary_mask = cv2.resize(
                            binary_mask, 
                            (frame_w, frame_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # Combine masks using bitwise OR for better performance
                    vehicle_mask = cv2.bitwise_or(vehicle_mask, binary_mask)
                    vehicle_count += 1
        
        return DetectionResult(
            vehicle_mask=vehicle_mask,
            detection_time=detection_time,
            vehicle_count=vehicle_count
        )


class RegionService:
    """Service for managing parking regions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.regions_loaded = False
        self.regions = None
        self.thresholds = None
        
    def load_regions(self) -> None:
        """Load parking regions from file"""
        try:
            self.regions = load_regions_from_file(self.config.regions_file)
            self.thresholds = get_thresholds()
            self.regions_loaded = True
            self.logger.info("Regions and thresholds loaded successfully")
            
        except FileNotFoundError:
            raise RegionLoadError(f"Regions file not found: {self.config.regions_file}")
        except Exception as e:
            raise RegionLoadError(f"Failed to load regions: {str(e)}")
    
    def get_regions(self) -> Tuple:
        """Get loaded regions"""
        if not self.regions_loaded:
            raise RegionLoadError("Regions not loaded. Call load_regions() first.")
        return self.regions
    
    def get_thresholds(self) -> dict:
        """Get loaded thresholds"""
        if not self.regions_loaded:
            raise RegionLoadError("Regions not loaded. Call load_regions() first.")
        return self.thresholds


class ProcessingService:
    """Service for processing frames and detecting parking spaces"""
    
    def __init__(self, config: Config, region_service: RegionService):
        self.config = config
        self.region_service = region_service
        self.logger = get_logger(__name__)
        
    def process_frame(self, frame: np.ndarray, vehicle_mask: np.ndarray) -> FrameProcessingResult:
        """Process frame to detect parking spaces"""
        processing_start = time.time()
        
        try:
            regions = self.region_service.get_regions()
            thresholds = self.region_service.get_thresholds()
            
            # Process frame to find empty spaces
            parking_space_boxes, number_of_empty_spaces = process_frame(
                frame, 
                vehicle_mask, 
                self.config.video.probability_map, 
                thresholds,
                *regions
            )
            
            # Create result image
            result_image = cv2.addWeighted(frame, 0.7, parking_space_boxes, 0.3, 0)
            
            # Add text overlay
            cv2.putText(
                result_image, 
                f"Empty Parking Spaces: {number_of_empty_spaces}",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            processing_time = time.time() - processing_start
            
            return FrameProcessingResult(
                result_image=result_image,
                empty_spaces=number_of_empty_spaces,
                parking_space_boxes=parking_space_boxes,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - processing_start
            raise ProcessingError(f"Error processing frame: {str(e)}")


class VideoService:
    """Service for managing video input and output"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.cap: Optional[cv2.VideoCapture] = None
        
    def open_video(self, video_path: Optional[str] = None) -> None:
        """Open video file for processing"""
        video_file = video_path or self.config.video.input_file
        
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise ProcessingError(f"Cannot open video file: {video_file}")
            
        self.logger.info(f"Video opened: {video_file}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video"""
        if self.cap is None:
            raise ProcessingError("Video not opened. Call open_video() first.")
            
        return self.cap.read()
    
    def release(self) -> None:
        """Release video resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
    
    def show_frame(self, frame: np.ndarray, window_name: str = "Parking Spaces") -> bool:
        """Show frame and check for quit key"""
        if self.config.video.show_display:
            cv2.imshow(window_name, frame)
            return cv2.waitKey(1) & 0xFF == ord(self.config.video.quit_key)
        return False


class ParkingSpaceService:
    """Main service that coordinates all parking space detection operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.performance_monitor = get_performance_monitor()
        
        # Initialize services
        self.model_service = ModelService(config)
        self.region_service = RegionService(config)
        self.processing_service = ProcessingService(config, self.region_service)
        self.video_service = VideoService(config)
        
        self.frame_count = 0
        self.last_time_processed = 0.0
        self.last_processed_result = None  # Store last processing result
        
    def initialize(self) -> None:
        """Initialize all services"""
        self.logger.info("Initializing ParkingSpace Detection System")
        
        # Load model and regions
        self.model_service.load_model()
        self.region_service.load_regions()
        
        # Initialize timing        self.last_time_processed = time.time() - self.config.processing.interval_seconds
        
        self.logger.info("System initialization complete")
    
    def process_video(self, video_path: Optional[str] = None) -> None:
        """Process video file for parking space detection"""
        self.video_service.open_video(video_path)
        
        try:
            while True:
                ret, frame = self.video_service.read_frame()
                if not ret:
                    self.logger.info("End of video reached")
                    break
                  # Check if it's time to process frame
                current_time = time.time()
                should_process = current_time - self.last_time_processed >= self.config.processing.interval_seconds
                
                display_frame = frame  # Default to original frame
                
                if should_process:
                    # Process frame and get result
                    quit_requested = self._process_single_frame(frame, current_time)
                    if quit_requested:
                        self.logger.info("User requested quit")
                        break
                    # Use the processed result if available
                    if self.last_processed_result is not None:
                        display_frame = self.last_processed_result.result_image
                else:
                    # Show previous processed result or original frame
                    if self.last_processed_result is not None:
                        display_frame = self.last_processed_result.result_image
                    
                    # Check for quit key on current display
                    if self.video_service.show_frame(display_frame):
                        self.logger.info("User requested quit")
                        break
        finally:
            self.video_service.release()
            self._log_final_report()
    
    def _process_single_frame(self, frame: np.ndarray, current_time: float) -> None:
        """Process a single frame"""
        self.performance_monitor.start_frame_processing()
        self.last_time_processed = current_time
        self.frame_count += 1
        
        try:
            # Detect vehicles
            detection_result = self.model_service.detect_vehicles(frame)
              # Process parking spaces
            processing_result = self.processing_service.process_frame(
                frame, detection_result.vehicle_mask
            )
            
            # Store the result for display in non-processing frames
            self.last_processed_result = processing_result
            
            # Record performance metrics
            metrics = self.performance_monitor.end_frame_processing(detection_result.detection_time)
            
            # Log performance periodically
            if self.frame_count % self.config.performance.log_interval == 0:
                self.logger.info(
                    f"Frame {self.frame_count}: "
                    f"Processing time: {metrics.frame_processing_time:.3f}s, "
                    f"Detection time: {metrics.detection_time:.3f}s, "
                    f"Empty spaces: {processing_result.empty_spaces}, "
                    f"Vehicles detected: {detection_result.vehicle_count}"
                )
            
            # Show result (return True if quit key pressed)
            return self.video_service.show_frame(processing_result.result_image)
            
        except Exception as e:
            self.logger.error(f"Error processing frame {self.frame_count}: {str(e)}")
            self.performance_monitor.end_frame_processing()  # End timing even on error
            return False
    
    def _log_final_report(self) -> None:
        """Log final performance report"""
        self.performance_monitor.log_performance_report()
        self.logger.info(f"Processed {self.frame_count} frames successfully")
