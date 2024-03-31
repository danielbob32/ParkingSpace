#%%
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import numpy as np

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

results = model(source='Assets/Baseline Images/1.jpg', show=False, conf=0.05, save=True, classes=2, save_txt=True, save_conf=True, line_width=1)

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

#%% Creating prob map for a single car

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Function to extract coordinates
def extract_coordinates(text):
    # Find all coordinate pairs in the text
    coord_pairs = re.findall(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', text)
    # Convert strings to floats and make a numpy array
    coords_list = [tuple(map(float, pair)) for pair in coord_pairs]
    return np.array(coords_list)

# Path to the directory containing the mask files
directory = 'Assets/Mask Outputs'

# Initialize an empty list to store coordinates
all_coordinates = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('_masks.txt'):
        with open(os.path.join(directory, filename), 'r') as file:
            text = file.read()
            # Assuming the first car in each file is the one we're tracking
            if 'Car #1 Mask Coordinates' in text:
                index_start = text.find('Car #1 Mask Coordinates')
                index_end = text.find('Car #2 Mask Coordinates', index_start)
                car_text = text[index_start:index_end]
                coords = extract_coordinates(car_text)
                # Store the centroid of the car for simplicity
                centroid = np.mean(coords, axis=0)
                all_coordinates.append(centroid)

# Rest of the code for creating the grid and plotting remains the same
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
plt.title("Probability Map of Car #1")
plt.show()

# %%
