#%%

import numpy as np
import cv2
import os
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
hsv_mask_file = 'Assets/MaskOutputs/video_0_frame0_mask.png'
output_dir = 'Assets/ParkingSpaces'
output_path = os.path.join(output_dir, 'parking_spaces.png')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the probability map
prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)
mask_img = cv2.imread(hsv_mask_file, cv2.IMREAD_GRAYSCALE)

# Invert the mask image
mask_img = cv2.bitwise_not(mask_img)

# Combine the probability map with the inverted mask
prob_map = cv2.bitwise_and(prob_map, prob_map, mask=mask_img)

# Threshold the probability map
_, thresholded = cv2.threshold(prob_map, 128, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# set a threshold to remove small contours
min_area = 8000

# Filter out small contours based on the area
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw the contours on a blank image
contour_img = np.zeros_like(prob_map)
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Dilate the contours to make them more visible
kernel = np.ones((3, 3), np.uint8)
contour_img = cv2.dilate(contour_img, kernel, iterations=1)

# Convert the contour image to a 3-channel BGR image
contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)

# Split the channels
b, g, r = cv2.split(contour_img)

# Perform the operation on each channel
b[contour_img[:,:,0] == 255] = 0
g[contour_img[:,:,1] == 255] = 255
r[contour_img[:,:,2] == 255] = 0

# Merge the channels
contour_img = cv2.merge((b, g, r))

# Save the output image
result = cv2.imwrite(output_path, contour_img)

if result:
    print(f"Parking spaces detected and saved successfully to {output_path}")
else:
    print(f"Failed to save the parking spaces. Check the output path: {output_path}")







# %% draw areas
# Initialize a list to store clicked points
coords = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        # Draws a circle where you click
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', img)
        print(f"Point chosen: ({x}, {y})")  # Prints to the Jupyter output cell

# Load your image
img = cv2.imread(prob_map_path, cv2.IMREAD_COLOR)

# Display the image in a window
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% define areas   
# Use the points from the coords list to define your regions
# Here's an example if you chose 12 points: 4 for near region, 4 for far region, and 4 for ignore region

near_region = np.array(coords[:4], dtype=np.int32)
far_region = np.array(coords[4:8], dtype=np.int32)
ignore_regions = [np.array(coords[8:], dtype=np.int32)]  # Assuming the last 4 points are for ignore region

# Print out the regions for confirmation
print("Near Region:", near_region)
print("Far Region:", far_region)
print("Ignore Regions:", ignore_regions)

# Draw the near region in blue
cv2.polylines(img, [near_region], True, (255, 0, 0), 3)

# Draw the far region in green
cv2.polylines(img, [far_region], True, (0, 255, 0), 3)

# Draw each ignore region in red
for region in ignore_regions:
    cv2.polylines(img, [region], True, (0, 0, 255), 3)

# Show the image with the regions drawn on it
cv2.imshow('Defined Regions', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save this image to check the regions
cv2.imwrite('defined_regions.png', img)
# %% process image
import matplotlib.pyplot as plt

# Define utility functions
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

# Thresholds for 'near' and 'far' regions and solidity
threshold_near = 15000
threshold_far = 1500
threshold_solidity = 0.95 # Contours with a solidity less than this will be filtered out

# Load the original probability map and mask image
prob_map_path = 'Assets/ProbMap/probability_map.png'
hsv_mask_file = 'Assets/MaskOutputs/video_0_frame0_mask.png'
prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)
mask_img = cv2.imread(hsv_mask_file, cv2.IMREAD_GRAYSCALE)

# Invert the mask image and combine it with the probability map
mask_img_inv = cv2.bitwise_not(mask_img)
prob_map_combined = cv2.bitwise_and(prob_map, prob_map, mask=mask_img_inv)

# Threshold the combined map to binary
_, binary_map = cv2.threshold(prob_map_combined, 80, 255, cv2.THRESH_BINARY)

# Additional parameters for morphological operations
# If the current kernel size is too strong, you can try a smaller one:
kernel_erode = np.ones((3 ,3), np.uint8)  # Try reducing the size of the kernel

# Perform erosion to separate connected objects
eroded_image = cv2.erode(binary_map, kernel_erode, iterations=1)

# Find contours on the eroded image
contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

final_contours = []
for contour in contours:
    center = get_contour_center(contour)
    if center is None:
        continue

    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Check if the contour is in 'near' or 'far' region
    in_near_region = cv2.pointPolygonTest(near_region, center, False) >= 0
    in_far_region = cv2.pointPolygonTest(far_region, center, False) >= 0

    # Check if the contour is within any ignore regions
    in_ignore_region = any(cv2.pointPolygonTest(region, center, False) >= 0 for region in ignore_regions)
    
    # Apply area and solidity filters based on the region
    if in_near_region and not in_ignore_region:
        if area >= threshold_near and solidity >= threshold_solidity:
            final_contours.append(contour)
    elif in_far_region and not in_ignore_region:
        if area >= threshold_far:  # Solidity check might be relaxed for far region
            final_contours.append(contour)

# Draw the final contours on the final_img
cv2.drawContours(contour_img, final_contours, -1, (255), thickness=cv2.FILLED)

# Optionally dilate the contours on the contour_img to make them more visible
kernel = np.ones((3, 3), np.uint8)
contour_img_dilated = cv2.dilate(contour_img, kernel, iterations=1)

# Convert the contour_img to a 3-channel BGR image for coloring
contour_img_bgr = cv2.cvtColor(contour_img_dilated, cv2.COLOR_GRAY2BGR)

# Color the contour areas in green on the contour_img_bgr
b, g, r = cv2.split(contour_img_bgr)
g[contour_img_bgr[:,:,0] == 255] = 255
contour_img_colored = cv2.merge((b, g, r))
empty_spaces = len(final_contours)  # final_contours are your filtered parking space contours
output_path = 'Assets/ParkingSpaces/parking_spaces.png'
# Save the result
if cv2.imwrite(output_path, contour_img_colored):
    print(f"Parking spaces detected and saved successfully to {output_path}")
else:
    print(f"Failed to save the parking spaces. Check the output path: {output_path}")



# %% overlay on original image 

# Load the original image where you want to overlay the parking spots
original_image_path = 'Assets/SegmentedImages/video_0_frame0.png'  # Path to your base image
original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

# Load the processed image with parking spaces in green
processed_image_path = 'Assets/ParkingSpaces/parking_spaces.png'  # Path to your processed image
processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)

# Find contours in the processed image
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the number of empty spaces
number_of_empty_spaces = len(contours)

# Draw the contours on the original image
for contour in contours:
    cv2.drawContours(original_image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)

# Create a window to display the result
window_title = f"Empty Parking Spaces: {number_of_empty_spaces}"
cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
cv2.imshow(window_title, original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the original image with parking spots overlaid
overlay_image_path = 'Assets/SegmentedImages/video_0_frame0_with_overlay.png'  # Path to save overlay image
cv2.imwrite(overlay_image_path, original_image)# %%

# %%
