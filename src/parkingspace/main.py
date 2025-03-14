# src/parkingspace/main.py

import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import from your new modules
from .pipeline import process_frame
from .regions import load_regions_from_file, get_thresholds

def main():
    # Set CUDA benchmarking for performance
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the regions from file
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
    ) = load_regions_from_file('regions.json')  # or your path

    # Get thresholds
    thresholds = get_thresholds()

    # Initialize YOLO model
    model = YOLO('yolo11x-seg.pt')  # or your model path
    model.to(device)

    # Path to the video file
    video_file = 'Demo/exp3.mp4'  # your video path
    prob_map_path = 'Demo/probability_map.png'  # your probability map path

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    interval = 3.0  # seconds
    last_time_processed = time.time() - interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_time_processed >= interval:
            last_time_processed = current_time

            # YOLO detection
            result = model(frame, conf=0.25, classes=[2, 3, 7], imgsz=(1088, 1920))
            r = result[0]
            masks = r.masks
            class_ids = r.boxes.cls if r.boxes is not None else []

            # Combine masks
            vehicle_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            if masks is not None:
                for mask, cls in zip(masks.data, class_ids):
                    class_id = int(cls)
                    if class_id in [2, 3, 7]:  # car, motorcycle, truck
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Extra check to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
