#%%
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np

#%% Make Snips
import subprocess
import os

# Define the video source and the output video name
video_source = "Assets/test.mp4"  # Change this to your video file path
output_video = "looped_video.mp4"

# Calculate how many times to loop the video to get close to 1000 images
loops_required = 94  # As your video is approximately 10 seconds long

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

# Define the rate at which you want to extract frames (frames per second)
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

# Assuming the results object has an attribute 'masks'
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
import os

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

# Initialize your model
model = YOLO('yolov8x-seg.pt')

# Define your directories
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

# Run inference on the source as a stream
results = model(source='rtsp', stream=True, show=True, conf=0.15, classes=2, line_width=1)

for r in results:
    # Convert boxes to numpy array
    boxes_np = r.boxes.numpy()

    # Count cars (assuming class 2 is 'car')
    car_count = (r.boxes.cls == 2).sum()

    # Calculate vacant parking spots
    vacant_spots = total_parking_spots - car_count

    # Since we are not using OpenCV, we can print the count to the console
    #print(f"Vacant Spots": {vacant_spots})


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

#%% Create Probabilty map
import torch
import numpy as np
import os
import re

def create_probability_map(mask_dir, output_file, shape):
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.txt')]
    sum_masks = np.zeros(shape, dtype=float)
    count = 0

    for mask_file in mask_files:
        with open(mask_file, 'r') as f:
            data = f.read()

        # Extract tensor data using a regular expression
        tensor_data = re.findall(r'data: tensor\((\[.*?\])\)', data, re.DOTALL)
        if tensor_data:
            for tensor_string in tensor_data:
                # Convert string representation of tensor to an actual tensor
                tensor = torch.tensor(eval(tensor_string))
                # Convert tensor to numpy array and add to sum_masks
                sum_masks += tensor.numpy()

        count += 1

    # Calculate the probability (average)
    probability_map = sum_masks / count

    # Convert the probability map to a format suitable for saving as an image
    probability_map_image = (probability_map * 255).astype(np.uint8)
    
    # Save the probability map
    cv2.imwrite(output_file, probability_map_image)

# Use the function
mask_directory = 'Assets/Mask Outputs'  # Replace with your masks directory
output_probability_map = 'probability_map.jpg'  # Output file path
image_shape = (384, 640)  # Replace with the shape of your masks
create_probability_map(mask_directory, output_probability_map, image_shape)
# %%
