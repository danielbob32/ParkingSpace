# %% Parking Space Detection with Segmentation Masks

import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import torch
import random

# Set CUDA benchmarking for performance optimization
torch.backends.cudnn.benchmark = True

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Function to get the center of the contour
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def verify_nearby_vehicle(contour, vehicle_mask, aspect_ratio, 
                          search_radius=50, aspect_ratio_tolerance=0.4):
    """
    Returns:
        List of bounding boxes (x, y, w, h) for any 'similar' vehicles
        near the given contour, or an empty list if none found.
    """

    x, y, w, h = cv2.boundingRect(contour)

    # Define the search region around the contour
    search_x1 = max(x - search_radius, 0)
    search_y1 = max(y - search_radius, 0)
    search_x2 = min(x + w + search_radius, vehicle_mask.shape[1])
    search_y2 = min(y + h + search_radius, vehicle_mask.shape[0])

    # Extract the region of interest from the vehicle mask
    roi_vehicle_mask = vehicle_mask[search_y1:search_y2, search_x1:search_x2]

    # Find vehicle contours in the search region
    nearby_contours, _ = cv2.findContours(roi_vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matching_vehicles = []
    for nearby_contour in nearby_contours:
        nx, ny, nw, nh = cv2.boundingRect(nearby_contour)

        # Recompute bounding box in the coordinate space of the full image
        full_x = search_x1 + nx
        full_y = search_y1 + ny

        # Calculate aspect ratio for the found vehicle
        if nw != 0:
            nearby_aspect_ratio = nh / float(nw)
        else:
            nearby_aspect_ratio = 0

        # Check aspect ratio similarity
        ratio_difference = abs(nearby_aspect_ratio - aspect_ratio)
        if ratio_difference <= aspect_ratio_tolerance:
            # Found a "similar" vehicle in the search region
            matching_vehicles.append((full_x, full_y, nw, nh))

    return matching_vehicles


def process_frame(frame, vehicle_mask, prob_map_path, thresholds):
    # 1) Load probability map
    prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)
    if prob_map is None:
        raise Exception("Failed to load probability map")

    # Resize probability map if needed
    if prob_map.shape != frame.shape[:2]:
        prob_map = cv2.resize(prob_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2) Invert vehicle_mask to find empty parking regions
    vehicle_mask_scaled = (vehicle_mask * 255).astype(np.uint8)
    mask_img_inv = cv2.bitwise_not(vehicle_mask_scaled)

    # 3) Combine probability map with inverted mask
    prob_map_combined = cv2.bitwise_and(prob_map, prob_map, mask=mask_img_inv)

    # 4) Binarize + morphological ops
    _, binary_map = cv2.threshold(prob_map_combined, 50, 255, cv2.THRESH_BINARY)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_map, kernel, iterations=6)

    # 5) Find potential parking-space contours
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We'll collect: (contour, vehicle_bboxes_for_that_contour)
    final_contours_info = []

    for contour in contours:
        center = get_contour_center(contour)
        if center is None:
            continue

        # Skip if in an ignore region
        if any(cv2.pointPolygonTest(region, center, False) >= 0 for region in ignore_regions):
            continue

        # Basic geometry: area, solidity
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area else 0

        # Determine region thresholds
        region_thresholds = None
        if cv2.pointPolygonTest(upper_level_l, center, False) >= 0:
            region_thresholds = thresholds['upper_level_l']
        elif cv2.pointPolygonTest(upper_level_m, center, False) >= 0:
            region_thresholds = thresholds['upper_level_m']
        elif cv2.pointPolygonTest(upper_level_r, center, False) >= 0:
            region_thresholds = thresholds['upper_level_r']
        elif cv2.pointPolygonTest(close_perp, center, False) >= 0:
            region_thresholds = thresholds['close_perp']
        elif cv2.pointPolygonTest(far_side, center, False) >= 0:
            region_thresholds = thresholds['far_side']
        elif cv2.pointPolygonTest(close_side, center, False) >= 0:
            region_thresholds = thresholds['close_side']
        elif cv2.pointPolygonTest(far_perp, center, False) >= 0:
            region_thresholds = thresholds['far_perp']
        elif cv2.pointPolygonTest(small_park, center, False) >= 0:
            region_thresholds = thresholds['small_park']
        else:
            # Not in a defined region
            continue

        # Check area and solidity
        if area < region_thresholds['min_area']:
            continue
        if solidity < region_thresholds['min_solidity']:
            continue

        # Bounding rect dimension check
        x, y, w, h = cv2.boundingRect(contour)
        if not (region_thresholds['min_width'] <= w <= region_thresholds['max_width'] and
                region_thresholds['min_height'] <= h <= region_thresholds['max_height']):
            continue

        # Aspect ratio check
        aspect_ratio = max(w, h) / float(min(w, h) if min(w, h) else 1)
        if aspect_ratio > region_thresholds['max_aspect_ratio']:
            continue

        # 6) Now find *all* vehicles near this contour
        vehicle_bboxes = verify_nearby_vehicle(
            contour, 
            vehicle_mask_scaled, 
            aspect_ratio, 
            search_radius=50, 
            aspect_ratio_tolerance=0.4
        )

        final_contours_info.append((contour, vehicle_bboxes))

    # -------------------------------------------------------------------
    # 7) Draw results on a copy of the original frame
    # -------------------------------------------------------------------
    labeled_image_bgr = frame.copy()
    total_spaces = 0
    avg_width_space = 200

    for i, (contour, vehicle_bboxes) in enumerate(final_contours_info):
        # If there are NO vehicles nearby => spot is red
        if len(vehicle_bboxes) == 0:
            color = (0, 0, 255)  # BGR for red
        else:
            # If there's at least one vehicle => random color
            # (You can also do something stable, e.g., random.seed(i))
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            color = (b, g, r)

        # Get bounding rect for the spot
        x, y, w, h = cv2.boundingRect(contour)

        # Subdivide if extra wide
        if w > avg_width_space:
            num_spaces = int(w / avg_width_space)
            space_width = w / num_spaces
            for j in range(num_spaces):
                sx = int(x + j * space_width)
                cv2.rectangle(labeled_image_bgr, (sx, y), (sx + int(space_width), y + h), color, 2)
                cv2.putText(labeled_image_bgr, str(total_spaces + 1),
                            (sx + int(space_width)//2, y + h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                total_spaces += 1
        else:
            # Single bounding box
            cv2.rectangle(labeled_image_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(labeled_image_bgr, str(total_spaces + 1),
                        (x + w//2, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            total_spaces += 1

        # 8) Draw *all* vehicles for this spot in the same color
        for (vx, vy, vw, vh) in vehicle_bboxes:
            cv2.rectangle(labeled_image_bgr, (vx, vy), (vx + vw, vy + vh), color, 2)

    return labeled_image_bgr, total_spaces
    
# Function to load the regions from the pre-made regions.json file
def load_regions_from_file(file_path='regions.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return (
        np.array(data['upper_level_l']),
        np.array(data['upper_level_m']),
        np.array(data['upper_level_r']),
        np.array(data['close_perp']),
        np.array(data['far_side']),
        np.array(data['close_side']),
        np.array(data['far_perp']),
        np.array(data['small_park']),
        [np.array(region) for region in data['ignore_regions']]
    )

# The thresholds for each region
thresholds = {
    'upper_level_l': {
        'min_area': 2000,
        'max_aspect_ratio': 16,
        'min_solidity': 0.7,
        'min_width': 60,
        'max_width': 500,
        'min_height': 40,
        'max_height': 300
    },
    'upper_level_m': {
        'min_area': 2000,
        'max_aspect_ratio': 16,
        'min_solidity': 0.7,
        'min_width': 120,
        'max_width': 1050,
        'min_height': 50,
        'max_height': 300
    },
    'upper_level_r': {
        'min_area': 2000,
        'max_aspect_ratio': 16,
        'min_solidity': 0.7,
        'min_width': 170,
        'max_width': 500,
        'min_height': 100,
        'max_height': 150
    },
    'close_perp': {
        'min_area': 10,
        'max_aspect_ratio': 5,
        'min_solidity': 0.6,
        'min_width': 10,
        'max_width': 200,
        'min_height': 10,
        'max_height': 200
    },
    'far_side': {
        'min_area': 100,
        'max_aspect_ratio': 5,
        'min_solidity': 0.7,
        'min_width': 30,
        'max_width': 200,
        'min_height': 30,
        'max_height': 200
    },
    'close_side': {
        'min_area': 100,
        'max_aspect_ratio': 5,
        'min_solidity': 0.6,
        'min_width': 30,
        'max_width': 200,
        'min_height': 30,
        'max_height': 200
    },
    'far_perp': {
        'min_area': 100,
        'max_aspect_ratio': 5,
        'min_solidity': 0.7,
        'min_width': 30,
        'max_width': 200,
        'min_height': 30,
        'max_height': 200
    },
    'small_park': {
        'min_area': 200,
        'max_aspect_ratio': 2,
        'min_solidity': 0.9,
        'min_width': 30,
        'max_width': 200,
        'min_height': 30,
        'max_height': 200
    }
}

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
) = load_regions_from_file()

# Initialize the YOLO model with CUDA support
model = YOLO('yolo11x-seg.pt')  # Load the model
model.to(device)

# Path to the video file
video_file = 'Demo/exp3.mp4'  # Change to your video path
prob_map_path = 'Demo/probability_map.png'  # Change to your probability map path

# Open the video file
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# Set the interval in seconds
interval = 3.0  # Process one frame every second

# Initialize a variable to keep track of the last processed time
last_time_processed = time.time() - interval

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_time_processed >= interval:
        # Update the last processed time
        last_time_processed = current_time

        # Process the frame using the model
        result = model(frame, conf=0.25, classes=[2, 3, 7], imgsz=(1088, 1920))
        r = result[0]
        masks = r.masks
        class_ids = r.boxes.cls

        # Combine the masks for the vehicle classes
        vehicle_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        if masks is not None:
            for mask, cls in zip(masks.data, class_ids):
                class_id = int(cls)
                if class_id in [2, 3, 7]:  # Vehicle classes: car, motorcycle, truck
                    # Convert mask to binary NumPy array
                    binary_mask = mask.cpu().numpy().astype(np.uint8)
                    # Resize the mask to the original frame size if necessary
                    if binary_mask.shape != (frame.shape[0], frame.shape[1]):
                        binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Combine masks of all detected vehicles
                    vehicle_mask = cv2.bitwise_or(vehicle_mask, binary_mask)


        # Process to find empty spaces
        parking_space_boxes, number_of_empty_spaces = process_frame(frame, vehicle_mask, prob_map_path, thresholds)

        # Overlay the parking space boxes on the original frame
        result_image = cv2.addWeighted(frame, 0.7, parking_space_boxes, 0.3, 0)

        # Display the number of empty parking spaces
        cv2.putText(result_image, f"Empty Parking Spaces: {number_of_empty_spaces}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Parking Spaces', result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
