#%%

import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
import subprocess
from ultralytics import YOLO
import time
import shutil
import re

# %% Snipshops from video
input_folder = 'Assets/VideoFiles'
output_folder = 'Assets/SnipShots'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

snapshot_interval = 1800  # Define the interval for snapshots
small_jump = 100  # Define a smaller interval for the set function

# Iterate over each video in the input folder and its subfolders
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

# %% segmetation
# Assuming the YOLO model is correctly defined and imported
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
# %% make HSV mask

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

    # Save the binary mask
    binary_mask_path = os.path.join(binary_mask_dir, f'{os.path.splitext(image_file)[0]}_mask.png')
    cv2.imwrite(binary_mask_path, hsv_mask)

    end_time = time.time()
    print(f'Processed {image_file} in {end_time - start_time:.2f} seconds.')

print("All images processed and saved.")

# %% make probability map

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

# %% detect parking spaces
prob_map_path = 'Assets/ProbMap/probability_map.png'
output_dir = 'Assets/ParkingSpaces'
output_path = os.path.join(output_dir, 'parking_spaces.png')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load the probability map
prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)

# Threshold the probability map to get the parking spaces
_, parking_spaces = cv2.threshold(prob_map, 128, 255, cv2.THRESH_BINARY)

# Save the parking spaces image
result = cv2.imwrite(output_path, parking_spaces)
if result:
    print(f"Parking spaces detected and saved successfully to {output_path}")
else:
    print(f"Failed to save the parking spaces image. Check the output path: {output_path}")


# %% detect cars

