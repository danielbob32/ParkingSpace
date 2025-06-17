# %% pre-run for all
##%% import libraries

import numpy as np
import cv2
import os
from ultralytics import YOLO
import time
import shutil
import re
import json

## %% utility functions
# Function to extract video and frame number from the file name for sorting
def extract_video_frame(file_name):
    match = re.match(r'video_(\d+)_frame(\d+)\.png', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None
# Function to get the center of a contour
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

#Function to save and load regions
def save_regions_to_file(upper_level_l, upper_level_m, upper_level_r, close_perp,far_side,close_side,far_perp,small_park, ignore_regions, file_path='regions.json'):
    data = {'upper_level_l': upper_level_l.tolist(),'upper_level_m': upper_level_m.tolist(),'upper_level_r': upper_level_r.tolist(), 'close_perp': close_perp.tolist(), 'far_side': far_side.tolist(), 'close_side':
             close_side.tolist(), 'far_perp': far_perp.tolist(),'small_park': small_park.tolist(), 'ignore_regions': [region.tolist() for region in ignore_regions]}
    with open(file_path, 'w') as file:
        json.dump(data, file)

# Function to load regions from a file
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

# %% parking space processing
## %% Process image
upper_level_l,upper_level_m,upper_level_r, close_perp,far_side,close_side,far_perp,small_park, ignore_regions = load_regions_from_file()
prob_map_path = 'Assets/ProbMap/probability_map.png'
hsv_mask_file = 'Assets/MaskOutputs/video_2_frame14400_mask.png'
processed_image_path = 'Assets/ParkingSpaces/processed_image.png'
segmented_im_path = 'Assets/SegmentedImages/video_2_frame14400.png'
# Load the probability map
prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)

# Invert the mask image
mask_img = cv2.imread(hsv_mask_file, cv2.IMREAD_GRAYSCALE)
mask_img_inv = cv2.bitwise_not(mask_img)

# Combine the probability map with the inverted mask
prob_map_combined = cv2.bitwise_and(prob_map, prob_map, mask=mask_img_inv)

# Apply thresholding to create a binary map
_, binary_map = cv2.threshold(prob_map_combined, 80, 255, cv2.THRESH_BINARY)

# Define kernel size for morphological operations to separate close contours
kernel_size = 5 # This may need to be adjusted based on your specific contours
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Perform erosion to separate contours that are very close together
eroded_image = cv2.erode(binary_map, kernel, iterations=6)


# find contours on the eroded image
contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an index for labeling the contours
contour_index = 0

# Define a color for the text, e.g., white
text_color = (255, 255, 255)


# Define thresholds for each region
thresholds = {
        'upper_level_l': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 60,  
            'max_width': 500, 
            'min_height': 0, 
            'max_height': 300 
        },
        'upper_level_m': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 100,  
            'max_width': 1050, 
            'min_height': 50, 
            'max_height': 300 
        },
        'upper_level_r': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 110,  
            'max_width': 500, 
            'min_height':50, 
            'max_height': 150 
        },
        'close_perp': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.6,
            'min_width': 10, 
            'max_width': 100,
            'min_height': 10,
            'max_height': 100 
        },
        'far_side': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.7,
            'min_width': 70, 
            'max_width': 200, 
            'min_height': 30, 
            'max_height': 200 
        },
        'close_side': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.6,
            'min_width': 30, 
            'max_width': 200, #
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

final_contours = []
for contour in contours:
    center = get_contour_center(contour)
    if center is None:
        continue

    # Skip contours in ignore regions
    if any(cv2.pointPolygonTest(region, center, False) >= 0 for region in ignore_regions):
        continue

    area = cv2.contourArea(contour)

    # Check which region the contour center point is in
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

    # Calculate solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Apply thresholds based on the region
    if area < region_thresholds['min_area'] and solidity < region_thresholds['min_solidity']:
        continue

    # Calculate bounding rect and aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    

    # Check if contour dimensions are within the specified range for the region
    if not (region_thresholds['min_width'] <= w <= region_thresholds['max_width'] and
            region_thresholds['min_height'] <= h <= region_thresholds['max_height']):
        continue  # Skip contour if it doesn't meet size constraints

    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else max(w, h)

    if aspect_ratio > region_thresholds['max_aspect_ratio']:
        continue

    # Calculate solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    if solidity < region_thresholds['min_solidity']:
        continue

    # If the contour passed all checks, it's a valid parking space
    final_contours.append(contour)
# Draw final contours on a new image
result_image = np.zeros_like(prob_map, dtype=np.uint8)
cv2.drawContours(result_image, final_contours, -1, (255), thickness=cv2.FILLED)

# Optionally, convert to BGR if you want to save in color
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

# Save the result
cv2.imwrite(processed_image_path, result_image_bgr)

max_width_single_space = 100
# Define the gap you want to leave between split contours
split_gap = 10

# Create a copy of the result_image for labeling
labeled_image = np.zeros_like(result_image, dtype=np.uint8)
labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored labels

# Define font scale and thickness for labels
font_scale = 0.5
font_thickness = 2

# Define parking space size parameters
min_width_single_space = 60  # Minimum width to be considered a single space
avg_width_space = 175  # Average width of a parking space
min_sum_split = 200  # Minimum sum width to consider splitting into two

for i, contour in enumerate(final_contours):
    x, y, w, h = cv2.boundingRect(contour)

    if w > avg_width_space:
        num_spaces = w // avg_width_space

        # Check if the remaining width is enough to consider another space
        if w % avg_width_space >= min_width_single_space:
            num_spaces += 1

        space_width = w / num_spaces

        for j in range(num_spaces):
            space_x = x + j * space_width

            # Adjust the width of the last space if necessary
            last_space_width = space_width if j < num_spaces - 1 else w - space_x + x

            # Draw and label each divided space
            cv2.rectangle(labeled_image_bgr, (int(space_x), y), (int(space_x + last_space_width), y + h), (0, 255, 0), 2)
            label = f"{i + 1}-{j + 1}"
            cv2.putText(labeled_image_bgr, label, (int(space_x + last_space_width // 2), y + h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    else:
        # Handle single parking space
        cv2.rectangle(labeled_image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = str(i + 1)
        cv2.putText(labeled_image_bgr, label, (x + w // 2, y + h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)


# Save or display the labeled_image as needed
cv2.imwrite(processed_image_path, labeled_image_bgr)
# or display it
cv2.imshow('Labeled Image', labeled_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

## %% in development - parking spaces divider
max_width_single_space = 100
# Define the gap you want to leave between split contours
split_gap = 10

# Create a copy of the result_image for labeling
labeled_image = np.zeros_like(result_image, dtype=np.uint8)
labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored labels

# Define font scale and thickness for labels
font_scale = 0.5
font_thickness = 2

# Define parking space size parameters
min_width_single_space = 60  # Minimum width to be considered a single space
avg_width_space = 175  # Average width of a parking space


for i, contour in enumerate(final_contours):
    x, y, w, h = cv2.boundingRect(contour)

    if w > avg_width_space:
        num_spaces = w // avg_width_space

        # Check if the remaining width is enough to consider another space
        if w % avg_width_space >= min_width_single_space:
            num_spaces += 1

        space_width = w / num_spaces

        for j in range(num_spaces):
            space_x = x + j * space_width

            # Adjust the width of the last space if necessary
            last_space_width = space_width if j < num_spaces - 1 else w - space_x + x

            # Draw and label each divided space
            cv2.rectangle(labeled_image_bgr, (int(space_x), y), (int(space_x + last_space_width), y + h), (0, 255, 0), 2)
            label = f"{i + 1}-{j + 1}"
            cv2.putText(labeled_image_bgr, label, (int(space_x + last_space_width // 2), y + h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    else:
        # Handle single parking space
        cv2.rectangle(labeled_image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = str(i + 1)
        cv2.putText(labeled_image_bgr, label, (x + w // 2, y + h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)


# Save or display the labeled_image as needed
cv2.imwrite(processed_image_path, labeled_image_bgr)
# or display it
cv2.imshow('Labeled Image', labeled_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

## %% Color the result image contours in green
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
result_image_bgr[np.where((result_image_bgr == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

#load segmented image
segmented_im = cv2.imread(segmented_im_path, cv2.IMREAD_COLOR)
# Overlay the result image on top of the segmented image
overlay_image = cv2.addWeighted(segmented_im, 0.7, labeled_image_bgr, 0.3, 0)

# Display the overlay image
cv2.imshow('Overlay Image', overlay_image)
cv2.imwrite(processed_image_path, overlay_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the HSV mask image
hsv_mask = cv2.imread(hsv_mask_file, cv2.IMREAD_COLOR)

# Overlay the result image on top of the HSV mask
overlay_image = cv2.addWeighted(hsv_mask, 0.7, result_image_bgr, 0.3, 0)

# Display the overlay image
cv2.imshow('Overlay Image', overlay_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



# %% Pre-Processing (snippets, segmentation, mask, probability map, draw areas, width and heigh analysis)
## %% Snapshots from video
input_folder = 'Assets/VideoFiles'
output_folder = 'Assets/SnipShots'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

snapshot_interval = 1800  # Define the interval for snapshots
small_jump = 100  # Define a smaller interval for the set function

# Iterate over each video in the input folder and its subfolder
for root, dirs, files in os.walk(input_folder):
    for i, video_file in enumerate(files):
        input_video = os.path.join(root, video_file)
        cap = cv2.VideoCapture(input_video)

        # Check if the video opened successfully
        if not cap.isOpened():
            print(f'Error opening video file {video_file}')
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        while current_frame < total_frames:
            # Skip to a frame slightly before the one we want
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - small_jump)

            # Read frames one at a time until we reach the one we want
            for _ in range(small_jump):
                ret, frame = cap.read()
                if not ret:
                    break

            # If read was successful, save the frame
            if ret:
                frame_path = os.path.join(output_folder, f'video_{i}_frame{current_frame}.png')
                cv2.imwrite(frame_path, frame)

            # Skip to the next interval
            current_frame += snapshot_interval

        cap.release()

cv2.destroyAllWindows()

## %% segmentation

model = YOLO('yolov8s-seg.pt')

input_folder = 'Assets/SnipShots'
output_folder = 'Assets/SegmentedImages'
base_segmentation_output_dir = 'runs/segment'

# Record the start time
start_time = time.time()

# Function to extract video and frame number from the file name for sorting
def extract_video_frame(file_name):
    # This will match video_X_frameY.png and extract X, Y as integers
    match = re.match(r'video_(\d+)_frame(\d+)\.png', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

# Get all files and filter out any that don't match the expected pattern
file_names = [f for f in os.listdir(input_folder) if extract_video_frame(f) is not None]

# Sort files by video number, then by frame number
file_names.sort(key=extract_video_frame)

# Process each file in the sorted order
for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    print(f"Processing file: {file_name}")  # For debugging

    # Run the model and wait for a second for the file system to update (if necessary)
    # Replace the following line with your actual YOLO model call
    model(source=file_path, show=False, conf=0.05, save=True, classes=2, show_labels=False, show_conf=False, show_boxes=False)

    # Assuming the model saves the output with the same filename in a predict directory
    # Find the latest 'predict' directory
    latest_predict_dir = max([os.path.join(base_segmentation_output_dir, d) for d in os.listdir(base_segmentation_output_dir)], 
                             key=os.path.getctime)
    
    segmented_file_path = os.path.join(latest_predict_dir, file_name)
    output_path = os.path.join(output_folder, file_name)

    # Move the segmented file to the output directory
    if os.path.exists(segmented_file_path):
        shutil.move(segmented_file_path, output_path)
    else:
        print(f"WARNING: Segmented file not found for {file_name}")  # For debugging

# Record the end time
end_time = time.time()
total_time = end_time - start_time

print("Segmentation process completed.")
print(f"Total running time: {total_time} seconds")
## %% make HSV mask

# Directories
segmented_image_dir = 'Assets/SegmentedImages'
binary_mask_dir = 'Assets/MaskOutputs'

# Ensure binary mask directory exists
if not os.path.exists(binary_mask_dir):
    os.makedirs(binary_mask_dir)

# Define the color range for the orange masks in the HSV color space
hsv_lower_range = np.array([5, 100, 100], dtype=np.uint8)
hsv_upper_range = np.array([15, 255, 255], dtype=np.uint8)

# Morphological operations kernel
kernel = np.ones((3, 3), np.uint8)

# Function to extract video and frame number from the file name for sorting
def extract_video_frame(file_name):
    match = re.match(r'video_(\d+)_frame(\d+)\.png', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return 0, 0  # Default to 0 if no match

# Get all files and filter out any that don't match the expected pattern
file_names = [f for f in os.listdir(segmented_image_dir) if f.lower().endswith('.png') and extract_video_frame(f)]

# Sort files by video number, then by frame number
file_names.sort(key=extract_video_frame)

# Process each file in the sorted order
for image_file in file_names:
    start_time = time.time()

    image_path = os.path.join(segmented_image_dir, image_file)
    
    # Load and convert the image to HSV
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create the binary mask
    hsv_mask = cv2.inRange(image_hsv, hsv_lower_range, hsv_upper_range)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('HSV Mask', hsv_mask)    
    # Save the binary mask
    binary_mask_path = os.path.join(binary_mask_dir, f'{os.path.splitext(image_file)[0]}_mask.png')
    cv2.imwrite(binary_mask_path, hsv_mask)

    end_time = time.time()
    print(f'Processed {image_file} in {end_time - start_time:.2f} seconds.')

print("All images processed and saved.")

## %% make probability map

segmented_image_dir = 'Assets/MaskOutputs'  # Path to the folder with segmented mask images
output_directory = 'Assets/ProbMap'  # Path to save the output probability map
output_path = os.path.join(output_directory, 'probability_map.png')  # Full output file path

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize an empty array to accumulate the probabilities
accumulator = None

# Count the number of images processed
image_count = 0

# Iterate over the segmented mask images
for file_name in os.listdir(segmented_image_dir):
    if not file_name.lower().endswith(('.png')):
        continue  # Skip non-image files
    
    file_path = os.path.join(segmented_image_dir, file_name)
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not load image {file_name}")
        continue  # Skip any files that couldn't be loaded

    print(f"Processing image {file_name}")

    # Initialize the accumulator with the shape of the first mask
    if accumulator is None:
        accumulator = np.zeros_like(mask, dtype=np.float32)

    # Assume the mask is already binary with cars as white (255) and the rest as black (0)
    binary_mask = mask.astype(np.float32) / 255  # Normalize the mask to [0.0, 1.0]
    
    # Accumulate the probabilities
    accumulator += binary_mask
    image_count += 1

# Normalize the accumulator to get the probability map
if image_count > 0:
    probability_map = accumulator / image_count
    # Convert the probability map to a grayscale image in [0, 255]
    probability_map_image = np.uint8(probability_map * 255)
    # Save the probability map image
    result = cv2.imwrite(output_path, probability_map_image)
    if result:
        print(f"Probability map saved successfully to {output_path}")
    else:
        print(f"Failed to save the probability map. Check the output path: {output_path}")
else:
    print("No images were processed. Check the input directory.")

## %% Draw areas and save
# This section should be executed once to define and save regions
coords = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', img)
        print(f"Point chosen: ({x}, {y})")

img = cv2.imread('Assets/ProbMap/probability_map.png', cv2.IMREAD_COLOR)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

upper_level_l = np.array(coords[:8], dtype=np.int32)
upper_level_m = np.array(coords[8:16], dtype=np.int32)
upper_level_r = np.array(coords[16:24], dtype=np.int32)
close_perp = np.array(coords[24:32], dtype=np.int32)
far_side = np.array(coords[32:40], dtype=np.int32)
close_side = np.array(coords[40:48], dtype=np.int32)
far_perp = np.array(coords[48:56], dtype=np.int32)
small_park = np.array(coords[56:64], dtype=np.int32)
ignore_regions = [np.array(coords[64:], dtype=np.int32)]
save_regions_to_file(upper_level_l,upper_level_m,upper_level_r, close_perp,far_side,close_side,far_perp,small_park, ignore_regions)
print("Regions saved successfully.")

## %% plot regions on parkingspaces.png
# Load regions from file
upper_level_l,upper_level_m,upper_level_r,close_perp,far_side,close_side,far_perp,small_park, ignore_regions = load_regions_from_file()

# Load the image with parking spaces
parking_spaces_path = 'Assets/ProbMap/probability_map.png'
parking_spaces = cv2.imread(parking_spaces_path, cv2.IMREAD_COLOR)

# Draw the regions on the image
cv2.polylines(parking_spaces, [upper_level_l], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [upper_level_m], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [upper_level_r], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [close_perp], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [far_side], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [close_side], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [far_perp], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(parking_spaces, [small_park], isClosed=True, color=(0, 255, 0), thickness=2)
for region in ignore_regions:
    cv2.polylines(parking_spaces, [region], isClosed=True, color=(0, 0, 255), thickness=2)

# Display the image with regions
cv2.imshow('Parking Spaces with Regions', parking_spaces)
cv2.waitKey(0)
cv2.destroyAllWindows()

## %% calc width and heigh of image
# Initialize a list to store points
points = []

# Define the event callback function
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        points.append((x, y))

        if len(points) >= 2:
            cv2.line(image, points[-2], points[-1], (255, 0, 0), 2)
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            print(f"Width: {width}px, Height: {height}px")
            
            # Reset the points list so that the next pair can be measured
            points = []

        cv2.imshow('image', image)

# Read your image
image_path = 'Assets/ProbMap/probability_map.png'
image = cv2.imread(image_path)
cv2.imshow('image', image)

# Set the mouse callback function to 'click_event'
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()



