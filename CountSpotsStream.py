#%%
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8x.pt')

def count_cars(frame):
    # Initialize car count
    car_count = 0
    
    # Perform object detection using the YOLO model
    results = model(source=frame, conf=0.05, classes=2, save_conf=True, line_width=1)
    
    # Iterate through each result in the results list
    for result in results:
        # Access the 'boxes' attribute from each 'result'
        # Then iterate through each detection
        for detection in result.boxes:
            # Check if the class ID is 2 (for cars)
            if int(detection.cls) == 2:
                car_count += 1
    
    return car_count