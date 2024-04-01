#%%
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import cv2

#%% Segmenatation of raw images to create masks

from ultralytics import YOLO
import os
import shutil
import time

# Initialize the model
model = YOLO('yolov8x-seg.pt')

# Directories
image_dir = 'Assets/Baseline Images'
output_directory = 'Assets/Segmented Images'
base_segmentation_output_dir = 'runs/segment'  # Base directory where YOLO saves results

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to find the most recent directory in a given base directory
def find_latest_directory(base_path):
    dir_paths = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not dir_paths:
        return None
    latest_dir = max(dir_paths, key=os.path.getmtime)
    return latest_dir

# Process each image
for image_file in os.listdir(image_dir):
    if image_file.lower().endswith('.jpg'):
        image_path = os.path.join(image_dir, image_file)

        # Run the model (results are saved to a directory)
        model(source=image_path, show=False, conf=0.05, save=True, classes=2,
              show_labels=False, show_conf=False, show_boxes=False)

        # Wait a bit to ensure the filesystem is updated
        time.sleep(1)

        # Find the latest 'predict' directory
        latest_predict_dir = find_latest_directory(base_segmentation_output_dir)
        if latest_predict_dir:
            # Assuming the first file in the latest predict directory is the processed image
            for segmented_file in os.listdir(latest_predict_dir):
                segmented_file_path = os.path.join(latest_predict_dir, segmented_file)
                output_path = os.path.join(output_directory, segmented_file)

                # Move or copy the segmented file to the output directory
                shutil.move(segmented_file_path, output_path)
                break  # Only process the first file for each image

print("Segmentation process completed.")

#%% Produce binary mask hsv

import cv2
import numpy as np
import os

# Directories
segmented_image_dir = 'Assets/Segmented Images'  
binary_mask_dir = 'Assets/Mask Outputs'  

# Ensure binary mask directory exists
os.makedirs(binary_mask_dir, exist_ok=True)

# Define the color range for the orange masks in the HSV color space
hsv_lower_range = np.array([5, 100, 100], dtype=np.uint8)
hsv_upper_range = np.array([15, 255, 255], dtype=np.uint8)

# Morphological operations kernel
kernel = np.ones((3,3), np.uint8)

# Process each segmented image
for image_file in os.listdir(segmented_image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
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

print("HSV mask generation process completed.")

#%% Make probability map from masks

import cv2
import numpy as np
import os

segmented_image_dir = 'Assets/Mask Outputs'  # Path to the folder with segmented mask images
output_directory = 'Assets/Grey Scale'  # Path to save the output probability map
output_path = os.path.join(output_directory, 'probability_map.png')  # Full output file path

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize an empty array to accumulate the probabilities
accumulator = None

# Count the number of images processed
image_count = 0

# Iterate over the segmented mask images
for file_name in os.listdir(segmented_image_dir):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
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







#%%
#*********************** Old code for reference, will be deleted as progressed ***************************#
    
#%% Make Snips

# Input and output folders
video_source = "Assets/test.mp4"  
output_video = "looped_video.mp4"

# Define 94 loops, needed to make 1000 frames out of 10 sec video
loops_required = 94  

# Create a text file with the filenames repeated
concat_file = "concat_list.txt"
with open(concat_file, 'w') as f:
    for _ in range(loops_required):
        f.write(f"file '{video_source}'\n")

# Loop the video using ffmpeg
ffmpeg_loop_command = [
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', concat_file,
    '-c', 'copy',
    '-y',  # Overwrite the output file if it already exists
    output_video
]

subprocess.run(ffmpeg_loop_command)

# Output settings
frames_per_second = 1
output_folder = "extracted_frames"

# Create the directory for output images if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Build the ffmpeg command for extracting frames
ffmpeg_extract_command = [
    'ffmpeg',
    '-i', output_video,
    '-vf', f"fps={frames_per_second}",
    os.path.join(output_folder, "frame_%04d.jpg"),
    '-y'  # Overwrite the output file if it already exists
]

# Run the ffmpeg command
subprocess.run(ffmpeg_extract_command)

#%% Detection - single image
model = YOLO('yolov8x.pt')
results = model(source='Assets/Baseline Images/1.jpg', show=False, conf=0.05, save=True, classes=2, save_txt=True, save_conf=True, line_width=1)



#%% Segmantation for masks - single image
model = YOLO('yolov8x-seg.pt')

results = model(source='Assets/Baseline Images/frame_0001.jpg', show=False, conf=0.05, save=True, classes=2,show_labels=False,show_conf=False, show_boxes=False)

if hasattr(results[0], 'masks') and results[0].masks is not None:
    # Prepare the text content
    mask_texts = []
    for i, mask in enumerate(results[0].masks):
        mask_text = f"Car #{i+1} Mask Coordinates:\n{mask}\n"
        mask_texts.append(mask_text)
    
    # Write the mask data to a text file
    with open('car_masks.txt', 'w') as f:
        for mask_text in mask_texts:
            f.write(mask_text)

    print("Masks data saved to car_masks.txt")
else:
    print("No masks data found in the results.")

#%% Segmenation of serverial images

def process_images(model, image_dir, output_dir):
    # List all jpg files in the image directory
    images = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]
    
    # Process each image
    for image_path in images:
        results = model(source=image_path, show=False, conf=0.05, save=True, classes=2, save_txt=True, save_conf=True, line_width=1)

        # Check for the 'masks' attribute in the results
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            mask_texts = [f"Car #{i+1} Mask Coordinates:\n{mask}\n" for i, mask in enumerate(results[0].masks)]
            
            # Construct a filename for the mask data
            mask_filename = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_masks.txt")
            
            # Write the mask data to a text file
            with open(mask_filename, 'w') as f:
                f.writelines(mask_texts)
            
            print(f"Masks data saved to {mask_filename}")
        else:
            print(f"No masks data found for {image_path}")

# model init
model = YOLO('yolov8x-seg.pt')

# Choose path's
image_dir = 'Assets/Baseline Images'
output_dir = 'Assets/Mask Outputs'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process the images
process_images(model, image_dir, output_dir)
#%% Count Cars - stream (change frame to live feed)

model = YOLO('yolov8x.pt')

#Total parking spots 
total_parking_spots = 48

total_parking_spots = 10  # Replace '10' with the actual number of parking spots

# Run inference on the source as a stream
results = model(source='rtsp', stream=True, show=True, conf=0.15, classes=2, line_width=1)

for r in results:
    # Convert boxes to numpy array
    boxes_np = r.boxes.numpy()

    # Count cars (assuming class 2 is 'car')
    car_count = (r.boxes.cls == 2).sum()

    # Calculate vacant parking spots
    vacant_spots = total_parking_spots - car_count

 

#%% Draw Plots from masks for a single car
# Coordinates for the shape
coordinates = np.array([
   [        519,         516],
       [        507,         528],
       [        504,         528],
       [        495,         537],
       [        492,         537],
       [        483,         546],
       [        483,         561],
       [        480,         564],
       [        480,         597],
       [        483,         600],
       [        483,         603],
       [        486,         606],
       [        489,         606],
       [        492,         609],
       [        528,         609],
       [        531,         606],
       [        540,         606],
       [        543,         603],
       [        549,         603],
       [        552,         600],
       [        564,         600],
       [        567,         597],
       [        615,         597],
       [        627,         585],
       [        630,         585],
       [        633,         582],
       [        633,         579],
       [        636,         576],
       [        636,         573],
       [        642,         567],
       [        642,         555],
       [        645,         552],
       [        645,         528],
       [        642,         525],
       [        642,         522],
       [        639,         519],
       [        636,         519],
       [        633,         516]
], dtype=np.float32)

# Plotting the shape
plt.figure(figsize=(8, 6))
plt.plot(coordinates[:, 0], -coordinates[:, 1], 'o-')  # Inverting y-axis for visual purposes
plt.title("Shape Drawn from Coordinates")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('output_shape.png')

#%% Creating prob map for a single car blue

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import json

def extract_coordinates(text):
    coord_pairs = re.findall(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', text)
    return np.array([tuple(map(float, pair)) for pair in coord_pairs])

directory = 'Assets/Mask Outputs/Car_Mask_Files_normalized'

# Filename for Car #1
car_file = 'car_#1_masks.txt'

if car_file in os.listdir(directory):
    with open(os.path.join(directory, car_file), 'r') as file:
        file_content = file.read()
        json_objects = file_content.split('\n')
        for json_object in json_objects:
            if json_object:
                coords = json.loads(json_object)
                if coords:  # Ensure that there are coordinates to process
                    coords = np.array(coords)

                    # Normalize the coordinates
                    coords[:, 0] /= np.max(coords[:, 0])
                    coords[:, 1] /= np.max(coords[:, 1])

                    # Plot the filled polygon
                    plt.fill(coords[:, 0], coords[:, 1], color='blue', alpha=0.5)

        plt.title("Car #1 Occupancy Visualization", fontsize=15, fontweight='bold')
        plt.xlabel('Normalized X Coordinate', fontsize=12)
        plt.ylabel('Normalized Y Coordinate', fontsize=12)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
else:
    print("Mask file for Car #1 not found.")
# %% Organize the masks files per car
import os
import re
import numpy as np

# Function to extract coordinates for all cars in the text
def extract_all_car_coordinates(text):
    car_coords_dict = {}
    # Match each car's coordinates, expecting the format: "Car #1 Mask Coordinates:\n ... \n\n"
    car_sections = re.split(r'(Car #\d+ Mask Coordinates:)', text)
    for i in range(1, len(car_sections), 2):  # Increment by 2 to skip the actual coordinates and only get headers
        car_number = car_sections[i].split()[1]  # Car number is the second word in the header
        coords_text = car_sections[i+1].strip()
        coords_list = extract_coordinates(coords_text)
        if coords_list.size > 0:
            if car_number not in car_coords_dict:
                car_coords_dict[car_number] = []
            car_coords_dict[car_number].append(coords_list)
    return car_coords_dict

# Function to extract coordinates from a string
def extract_coordinates(coord_string):
    coord_pairs = re.findall(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', coord_string)
    return np.array(coord_pairs, dtype=float)

# Path to the directory containing the mask files
input_directory = 'Assets/Mask Outputs'
output_directory = os.path.join(input_directory, 'Car_Mask_Files')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file and compile coordinates for each car
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('_masks.txt'):
        filepath = os.path.join(input_directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
            car_coords_dict = extract_all_car_coordinates(text)
            for car_number, coords_list in car_coords_dict.items():
                car_filepath = os.path.join(output_directory, f'car_{car_number}_masks.txt')
                # Append coordinates to each car's file
                with open(car_filepath, 'a') as car_file:
                    for coords in coords_list:
                        car_file.write(f'{coords.tolist()}\n')

print("Completed processing all mask files.")

# %% Plot the masks for all cars (makes distirbution on edges)
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the directory containing the car mask files
input_directory = 'Assets/Mask Outputs/Car_Mask_Files'

# Initialize a grid for the probability map
grid_size = (100, 100)  # Adjust the size of the grid as needed
probability_grid = np.zeros(grid_size)

# Function to process a single file and extract coordinates
def process_file(filepath):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            # Strip unwanted characters
            line = line.strip().strip('[],\n')
            if line:  # Check if line is not empty
                # Find all floating-point numbers in the line
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                # We expect two numbers per line for x and y coordinates
                if len(numbers) == 2:
                    x, y = map(float, numbers)
                    coords.append((x, y))
    return coords

# Process all files and accumulate the coordinate counts on the grid
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('_masks.txt'):
        coords = process_file(os.path.join(input_directory, filename))
        # Debug: Print sample coordinates from each file
        print(f"{filename} sample coords: {coords[:5]}")
        for x, y in coords:
            grid_x = min(int(x * grid_size[0]), grid_size[0] - 1)
            grid_y = min(int(y * grid_size[1]), grid_size[1] - 1)
            probability_grid[grid_x, grid_y] += 1

# Debug: Print max, min, and total counts
print(f"Max count: {np.max(probability_grid)}, Min count: {np.min(probability_grid)}, Total count: {total_counts}")

# Normalize the grid to get the probability map
total_counts = np.sum(probability_grid)
if total_counts > 0:  # Prevent division by zero
    probability_grid /= total_counts

# Plot the probability map
plt.imshow(probability_grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Combined Probability Map for All Cars')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.gca().invert_yaxis()  # Inverting the y-axis so that origin is at the top-left
plt.show()

# %% Plot the masks for all cards v.2 (makes distirbution on edges also)
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define the path to the directory containing the car mask files
input_directory = 'Assets/Mask Outputs/Car_Mask_Files'  # Replace with your directory path

# Initialize a grid for the probability map
grid_size = (100, 100)
probability_grid = np.zeros(grid_size)

# Function to process a single file and extract coordinates
def process_file(filepath):
    coords = []
    with open(filepath, 'r') as f:
        content = f.read()
        # Look for all coordinate pairs within brackets
        matches = re.findall(r'\[\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)\s*\]', content)
        for match in matches:
            x, y = map(float, match[:2])
            coords.append((x, y))
    return coords

# Collect all coordinates from all files
all_coords = []
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('_masks.txt'):
        file_path = os.path.join(input_directory, filename)
        coords = process_file(file_path)
        print(f"File: {filename}, Number of coordinates: {len(coords)}")  # Debugging output
        all_coords.extend(coords)

# Convert the collected coordinates to a NumPy array
all_coords = np.array(all_coords)

# Normalize the coordinates using standard score normalization
if all_coords.size > 0:
    mean_value = np.mean(all_coords, axis=0)
    std_value = np.std(all_coords, axis=0) + 1e-8  # Add a small constant to prevent division by zero
    all_coords = (all_coords - mean_value) / std_value
# Increment the grid cells based on the coordinates
for x, y in all_coords:
    grid_x = min(int(x * (grid_size[0] - 1)), grid_size[0] - 1)
    grid_y = min(int(y * (grid_size[1] - 1)), grid_size[1] - 1)
    probability_grid[grid_y, grid_x] += 1  # Swap x and y here

# Normalize the grid to convert counts to probabilities
probability_grid /= np.sum(probability_grid)

# Use logarithmic normalization to better visualize the data
log_norm = LogNorm(vmin=probability_grid[probability_grid > 0].min(), vmax=probability_grid.max())

# Plot the raw coordinates
plt.figure(figsize=(10, 8))
plt.scatter(all_coords[:, 0], all_coords[:, 1], alpha=0.1)
plt.title('Scatter Plot of All Coordinates')
plt.xlabel('Normalized X Coordinate')
plt.ylabel('Normalized Y Coordinate')
plt.grid(True)
plt.show()

# Save the probability grid to a text file
np.savetxt('probability_grid.txt', probability_grid)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(probability_grid, cmap='hot', interpolation='nearest', norm=log_norm)
plt.title('Heatmap of All Coordinates')
plt.xlabel('Normalized X Coordinate')
plt.ylabel('Normalized Y Coordinate')
plt.colorbar(label='Log Probability')
plt.show()
# %% extract the xyn instead of xy
import os
import re
import numpy as np

# Function to extract coordinates for all cars in the text
def extract_all_car_coordinates(text):
    car_coords_dict = {}
    # Match each car's coordinates, expecting the format: "Car #1 Mask Coordinates:\n ... \n\n"
    car_sections = re.split(r'(Car #\d+ Mask Coordinates:)', text)
    for i in range(1, len(car_sections), 2):  # Increment by 2 to skip the actual coordinates and only get headers
        car_number = car_sections[i].split()[1]  # Car number is the second word in the header
        coords_text = car_sections[i+1].strip()
        coords_list = extract_xyn(coords_text)
        if coords_list.size > 0:
            if car_number not in car_coords_dict:
                car_coords_dict[car_number] = []
            car_coords_dict[car_number].append(coords_list)
    return car_coords_dict

# Function to extract xyn coordinates from a string
def extract_xyn(coord_string):
    # Match the xyn line and extract the array
    match = re.search(r'xyn: \[array\(\[\[(.*?)\]\]', coord_string, re.DOTALL)
    if match:
        xyn_string = match.group(1)
        # Split the string into coordinate triples and convert to float
        coord_triples = re.findall(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', xyn_string)
        return np.array(coord_triples, dtype=float)
    else:
        print(f"No match found in the following coordinate string:\n{coord_string}")
        return np.array([], dtype=float)


# Path to the directory containing the mask files
input_directory = 'Assets/Mask Outputs'
output_directory = os.path.join(input_directory, 'Car_Mask_Files_normalized')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file and compile coordinates for each car
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith('_masks.txt'):
        filepath = os.path.join(input_directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
            car_coords_dict = extract_all_car_coordinates(text)
            if not car_coords_dict:
                print(f"No coordinates extracted from file: {filename}")
            for car_number, coords_list in car_coords_dict.items():
                car_filepath = os.path.join(output_directory, f'car_{car_number}_masks.txt')
                # Append coordinates to each car's file
                with open(car_filepath, 'a') as car_file:
                    for coords in coords_list:
                        car_file.write(f'{coords.tolist()}\n')

print("Completed processing all mask files.")

# %% Plot the masks for all cars v.3 from xyn (works on 3 cars spereatly)
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Path to the directory containing the mask files
directory = 'Assets/Mask Outputs/Car_Mask_Files_normalized'

# Define the car numbers to process
car_numbers = [1, 2, 3]

# Loop through each car number
for car_number in car_numbers:
    # Initialize an empty list to store coordinates
    all_coordinates = []

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_masks.txt') and f"car_#{car_number}" in filename:
            print(f"Processing file {filename} for car #{car_number}")  # Debugging line
            with open(os.path.join(directory, filename), 'r') as file:
                file_content = file.read()
                json_objects = file_content.split('\n')  # Split the file content by newline character
                for json_object in json_objects:
                    if json_object:  # Check if the json object is not empty
                        coords = json.loads(json_object)  # Load coordinates from json object
                        centroid = np.mean(coords, axis=0)
                        all_coordinates.append(centroid)
    # Continue with the rest of the code for each car
    all_coordinates = np.array(all_coordinates)

    # Normalize coordinates to range [0, 1] if not already
    max_x = np.max(all_coordinates[:, 0])
    max_y = np.max(all_coordinates[:, 1])
    all_coordinates[:, 0] /= max_x
    all_coordinates[:, 1] /= max_y

    # Define the grid size
    grid_size = (100, 100)  # for example, adjust as needed

    # Create a grid to accumulate counts
    grid = np.zeros(grid_size)

    # Iterate over each coordinate and increment the grid cells
    for x, y in all_coordinates:
        grid_x = int(x * (grid_size[0] - 1))  # Scale to grid size
        grid_y = int(y * (grid_size[1] - 1))  # Scale to grid size
        grid[grid_x, grid_y] += 1

    # Normalize the grid to convert counts to probabilities
    probability_map = grid / len(all_coordinates)

    # Plot the probability map
    plt.imshow(probability_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Probability Map of Car #{car_number}")
    plt.show()
# %% Plot the masks for all cars v.4 from xyn (trying to combine all the plots to single one)
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Path to the directory containing the mask files
directory = 'Assets/Mask Outputs/Car_Mask_Files_normalized'

# Define the car numbers to process
car_numbers = [1, 2, 3]

# Initialize an empty list to store coordinates for all cars
all_coordinates = []

# Loop through each car number
for car_number in car_numbers:
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_masks.txt') and f"car_#{car_number}" in filename:
            print(f"Processing file {filename} for car #{car_number}")  # Debugging line
            with open(os.path.join(directory, filename), 'r') as file:
                file_content = file.read()
                json_objects = file_content.split('\n')  # Split the file content by newline character
                for json_object in json_objects:
                    if json_object:  # Check if the json object is not empty
                        coords = json.loads(json_object)  # Load coordinates from json object
                        centroid = np.mean(coords, axis=0)
                        all_coordinates.append(centroid)

# Continue with the rest of the code for all cars
all_coordinates = np.array(all_coordinates)

# Normalize coordinates to range [0, 1] if not already
max_x = np.max(all_coordinates[:, 0])
max_y = np.max(all_coordinates[:, 1])
all_coordinates[:, 0] /= max_x
all_coordinates[:, 1] /= max_y

# Define the grid size
grid_size = (100, 100)  # for example, adjust as needed

# Create a grid to accumulate counts
grid = np.zeros(grid_size)

# Iterate over each coordinate and increment the grid cells
for x, y in all_coordinates:
    grid_x = int(x * (grid_size[0] - 1))  # Scale to grid size
    grid_y = int(y * (grid_size[1] - 1))  # Scale to grid size
    grid[grid_x, grid_y] += 1

# Normalize the grid to convert counts to probabilities
probability_map = grid / len(all_coordinates)

# Plot the probability map
plt.imshow(probability_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Combined Probability Map for All Cars")
plt.show()
# %% Plot the masks for all cars v.5 from xyn (all cars in folder)
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Path to the directory containing the mask files
directory = 'Assets/Mask Outputs/Car_Mask_Files_normalized'

all_coordinates = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('_masks.txt'):
        # Extract the car number from the filename
        car_number = filename.split('_')[1]  # Assuming the filename format is "car_#_masks.txt"
        print(f"Processing file {filename} for car #{car_number}")  # Debugging line
        with open(os.path.join(directory, filename), 'r') as file:
            file_content = file.read()
            json_objects = file_content.split('\n')  # Split the file content by newline character
            for json_object in json_objects:
                if json_object:  # Check if the json object is not empty
                    coords = json.loads(json_object)  # Load coordinates from json object
                    centroid = np.mean(coords, axis=0)
                    all_coordinates.append(centroid)

# Continue with the rest of the code for all cars
all_coordinates = np.array(all_coordinates)

# Normalize coordinates to range [0, 1] if not already
max_x = np.max(all_coordinates[:, 0])
max_y = np.max(all_coordinates[:, 1])
all_coordinates[:, 0] /= max_x
all_coordinates[:, 1] /= max_y

# Define the grid size
grid_size = (100, 100)  # for example, adjust as needed

# Create a grid to accumulate counts
grid = np.zeros(grid_size)

# Iterate over each coordinate and increment the grid cells
for x, y in all_coordinates:
    grid_x = int(x * (grid_size[0] - 1))  # Scale to grid size
    grid_y = int(y * (grid_size[1] - 1))  # Scale to grid size
    grid[grid_x, grid_y] += 1

# Normalize the grid to convert counts to probabilities
probability_map = grid / len(all_coordinates)

# Plot the probability map
plt.imshow(probability_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Combined Probability Map for All Cars")
plt.show()
# %% final
import os
import json
import numpy as np
import matplotlib.pyplot as plt

directory =  'Assets/Mask Outputs/Car_Mask_Files_normalized' # replace with your directory
all_coordinates = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('_masks.txt'):
        print(f"Processing file {filename}")  # Debugging line
        with open(os.path.join(directory, filename), 'r') as file:
            file_content = file.read()
            json_objects = file_content.split('\n')  # Split the file content by newline character
            for json_object in json_objects:
                if json_object:  # Check if the json object is not empty
                    coords = json.loads(json_object)  # Load coordinates from json object
                    centroid = np.mean(coords, axis=0)
                    all_coordinates.append(centroid)

# Continue with the rest of the code for all cars
all_coordinates = np.array(all_coordinates)

# Normalize coordinates to range [0, 1] if not already
max_x = np.max(all_coordinates[:, 0])
max_y = np.max(all_coordinates[:, 1])
all_coordinates[:, 0] /= max_x
all_coordinates[:, 1] /= max_y

# Define the grid size
grid_size = (100, 100)  # for example, adjust as needed

# Create a grid to accumulate counts
grid = np.zeros(grid_size)

# Iterate over each coordinate and increment the grid cells
for x, y in all_coordinates:
    grid_x = int(x * (grid_size[0] - 1))  # Scale to grid size
    grid_y = int(y * (grid_size[1] - 1))  # Scale to grid size
    grid[grid_x, grid_y] += 1

# Normalize the grid to convert counts to probabilities
probability_map = grid / len(all_coordinates)

# Define the minimum and maximum values for the colormap
z_min, z_max = np.min(probability_map), np.max(probability_map)

# Define the minimum and maximum values for the x and y coordinates
x_min, x_max = 0, 1  # Since the coordinates were normalized to [0, 1]
y_min, y_max = 0, 1  # Since the coordinates were normalized to [0, 1]

# Plot the probability map
plt.imshow(probability_map, cmap='hot', vmin=z_min, vmax=z_max, extent=[x_min, x_max, y_min, y_max], interpolation='nearest', origin='lower')
plt.colorbar()
plt.title("Combined Probability Map for All Cars")
plt.show()
# %% creatuing prob map for a single car 'hot'
import os
import numpy as np
import matplotlib.pyplot as plt
import json

def read_json_object(json_object):
    coords = json.loads(json_object)
    return np.array(coords)

directory = 'Assets/Mask Outputs/Car_Mask_Files_normalized'
car_file = 'car_#1_masks.txt'
full_path = os.path.join(directory, car_file)

if os.path.exists(full_path):
    with open(full_path, 'r') as file:
        file_content = file.readlines()

    # Create a grid to accumulate the heat values
    grid_size = (100, 100)
    heat_map = np.zeros(grid_size)

    for json_object in file_content:
        if json_object.strip():  # Ensure the JSON object is not empty
            coords = read_json_object(json_object)
            if coords.size > 0:
                # Normalize the coordinates
                coords[:, 0] /= np.max(coords[:, 0])
                coords[:, 1] /= np.max(coords[:, 1])

                # Populate the heat map
                for coord in coords:
                    x, y = int(coord[0] * (grid_size[0] - 1)), int(coord[1] * (grid_size[1] - 1))
                    heat_map[y, x] += 1  # Increment the cell corresponding to this coordinate

    # Normalize the heat map
    heat_map /= np.max(heat_map)

    # Generate a smoothed heat map using Gaussian filter, if necessary
    # from scipy.ndimage.filters import gaussian_filter
    # heat_map_smoothed = gaussian_filter(heat_map, sigma=2)

    # Create the x and y coordinates for the grid
    x = np.linspace(0, 1, grid_size[0])
    y = np.linspace(0, 1, grid_size[1])
    X, Y = np.meshgrid(x, y)

    # Plot filled contour
    plt.contourf(X, Y, heat_map, levels=100, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.title(f"Heat Map of Car #1's Location Over Time", fontsize=14, fontweight='bold')
    plt.xlabel('Normalized X Coordinate', fontsize=10)
    plt.ylabel('Normalized Y Coordinate', fontsize=10)
    
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio of the plot square
    plt.show()

else:
    print(f"File not found: {car_file}")
# %% Mask for a single image w.o RGB
    
import cv2
import numpy as np

# Load the image from file
image_path = 'runs/segment/predict4/frame_0001.jpg'  # Make sure to provide the correct path to your image file
image = cv2.imread(image_path)

# Convert the image to the HSV color space
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for the orange masks in the HSV color space
# These values are approximations and may need to be fine-tuned
hsv_lower_range = np.array([5, 100, 100], dtype=np.uint8)
hsv_upper_range = np.array([15, 255, 255], dtype=np.uint8)

# Create a mask that detects all the pixels within the HSV range
hsv_mask = cv2.inRange(image_hsv, hsv_lower_range, hsv_upper_range)

# Apply morphological operations to clean up the mask
# These operations help to remove noise and small dots
kernel = np.ones((3,3), np.uint8)
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

# Save the resulting binary mask to a file
binary_mask_path = 'path_to_save_binary_mask.png'  # Provide the path where you want to save the mask
cv2.imwrite(binary_mask_path, hsv_mask)

# The saved image will have the car masks in white and the background in black

