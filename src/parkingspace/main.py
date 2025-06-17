# src/parkingspace/main.py

import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import from modules
from .pipeline import process_frame
from .regions import load_regions_from_file, get_thresholds
from .config import get_config
from .logger import setup_logging, get_logger
from .performance import get_performance_monitor
from .exceptions import ModelLoadError, RegionLoadError, ProcessingError

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting ParkingSpace Detection System")
    
    # Get configuration
    config = get_config()
    logger.info(f"Using device: {config.device}")
    
    # Get performance monitor
    perf_monitor = get_performance_monitor()
    
    try:
        # Set CUDA benchmarking for performance
        if config.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA benchmarking enabled")

        # Load regions
        try:
            (
                upper_level_l,
                upper_level_m,
                upper_level_r,
                close_perp,
                far_side,
                close_side,
                far_perp,
                small_park,
                ignore_regions
            ) = load_regions_from_file(config.regions_file)
            logger.info("Regions loaded successfully")
        except FileNotFoundError:
            raise RegionLoadError(f"Regions file not found: {config.regions_file}")
        except Exception as e:
            raise RegionLoadError(f"Failed to load regions: {str(e)}")

        # Get thresholds
        thresholds = get_thresholds()
        logger.info("Thresholds loaded successfully")

        # Initialize YOLO model
        try:
            model = YOLO(config.model_path)
            model.to(config.device)
            logger.info(f"YOLO model loaded: {config.model_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")        # Path to the video file
        video_file = 'Demo/exp3.mp4'
        prob_map_path = 'Demo/probability_map.png'

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ProcessingError(f"Cannot open video file: {video_file}")

        logger.info(f"Processing video: {video_file}")
        interval = 3.0  # seconds
        last_time_processed = time.time() - interval
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            current_time = time.time()
            if current_time - last_time_processed >= interval:
                perf_monitor.start_frame_processing()
                last_time_processed = current_time
                frame_count += 1

                try:
                    # YOLO detection
                    detection_start = time.time()
                    detection_config = config.get_detection_config()
                    result = model(
                        frame, 
                        conf=detection_config['confidence_threshold'], 
                        classes=detection_config['classes'], 
                        imgsz=(1088, 1920)
                    )
                    detection_time = time.time() - detection_start
                    
                    r = result[0]
                    masks = r.masks
                    class_ids = r.boxes.cls if r.boxes is not None else []

                    # Combine masks
                    vehicle_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    if masks is not None:
                        for mask, cls in zip(masks.data, class_ids):
                            class_id = int(cls)
                            if class_id in detection_config['classes']:
                                binary_mask = mask.cpu().numpy().astype(np.uint8)
                                if binary_mask.shape != (frame.shape[0], frame.shape[1]):
                                    binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]),
                                                             interpolation=cv2.INTER_NEAREST)
                                vehicle_mask = cv2.bitwise_or(vehicle_mask, binary_mask)

                    # Process frame to find empty spaces
                    parking_space_boxes, number_of_empty_spaces = process_frame(
                        frame, 
                        vehicle_mask, 
                        prob_map_path, 
                        thresholds,
                        upper_level_l, 
                        upper_level_m, 
                        upper_level_r,
                        close_perp, 
                        far_side, 
                        close_side, 
                        far_perp, 
                        small_park,
                        ignore_regions
                    )

                    # Overlay bounding boxes onto original
                    result_image = cv2.addWeighted(frame, 0.7, parking_space_boxes, 0.3, 0)

                    # Show count of empty spaces
                    cv2.putText(result_image, f"Empty Parking Spaces: {number_of_empty_spaces}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    cv2.imshow('Parking Spaces', result_image)
                    
                    # Record performance metrics
                    metrics = perf_monitor.end_frame_processing(detection_time)
                    
                    # Log performance every 10 frames
                    if frame_count % 10 == 0:
                        logger.info(f"Frame {frame_count}: Processing time: {metrics.frame_processing_time:.3f}s, "
                                  f"Detection time: {metrics.detection_time:.3f}s, "
                                  f"Empty spaces: {number_of_empty_spaces}")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    perf_monitor.end_frame_processing()  # End timing even on error
                    continue
                    
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # Final performance report
        perf_monitor.log_performance_report()
        logger.info(f"Processed {frame_count} frames successfully")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
