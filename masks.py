#%%
import torch
from ultralytics import YOLO

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
# %%
