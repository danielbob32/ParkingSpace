#%%
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    print(f"Vacant Spots": {vacant_spots})


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