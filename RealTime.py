# %% pre-run
import cv2
import numpy as np
import json
from ultralytics import YOLO
import time

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def process_frame(frame,prob_map_path,thresholds):

    # Define the color range for the orange masks in the HSV color space
    hsv_lower_range = np.array([5, 100, 100], dtype=np.uint8)
    hsv_upper_range = np.array([15, 255, 255], dtype=np.uint8)

    # Morphological operations kernel
    kernel = np.ones((3, 3), np.uint8)

     # Load and convert the image to HSV
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the binary mask
    hsv_mask = cv2.inRange(image_hsv, hsv_lower_range, hsv_upper_range)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

    # Load the probability map
    prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)
    if prob_map is None:
        raise Exception("Failed to load probability map")
    
   # Combine the probability map with the inverted mask
    mask_img_inv = cv2.bitwise_not(hsv_mask)
    prob_map_combined = cv2.bitwise_and(prob_map, prob_map, mask=mask_img_inv)

    # Apply thresholding to create a binary map
    _, binary_map = cv2.threshold(prob_map_combined, 80, 255, cv2.THRESH_BINARY)

    # Define kernel size for morphological operations to separate close contours
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform erosion to separate contours that are very close together
    eroded_image = cv2.erode(binary_map, kernel, iterations=6)  

    # Now find contours on the eroded image
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Define a color for the text, e.g., white
    text_color = (255, 255, 255)  

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

    #  BGR to save in color
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    # Save the result
    cv2.imwrite(processed_image_path, result_image_bgr)


    # Create a copy of the result_image for labeling
    labeled_image = np.zeros_like(result_image, dtype=np.uint8)
    labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored labels

    # Define font scale and thickness for labels
    font_scale = 0.5
    font_thickness = 2

    # Define parking space size parameters
    min_width_single_space = 60  # Minimum width to be considered a single space
    avg_width_space = 200  # Average width of a parking space that can be splited
 

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

    
#display the labeled_image as needed
    #cv2.imshow('Labeled Image', labeled_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    return labeled_image_bgr,final_contours






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




# Essentials for model
model = YOLO('yolov8x-seg.pt')
processed_image_path = 'Assets/ParkingSpaces/processed_image.png'
prob_map_path = 'Assets/ProbMap/probability_map.png'
rtsp_url =  'rtsp url here'
cap = cv2.VideoCapture(rtsp_url)
upper_level_l,upper_level_m,upper_level_r, close_perp,far_side,close_side,far_perp,small_park, ignore_regions = load_regions_from_file()

# The thresholds for each region
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


# %% main function for real-time processing
# Set the interval in seconds
interval = 1.0  # Process one frame every second

# Initialize a variable to keep track of the last processed time
last_time_processed = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_time_processed >= interval:
        # Update the last processed time
        last_time_processed = current_time

        # Process the frame using the model
        result = model(frame, stream=True)
        for r in result:
            segmented_frame = r.plot()

            # Debug: Visualize the segmentation results by combining with the original frame
            segmented_overlay = cv2.addWeighted(frame, 0.7, segmented_frame, 0.3, 0)

            # Process to find empty spaces
            final_image, final_contours = process_frame(segmented_frame, prob_map_path, thresholds)
            add_frame = cv2.addWeighted(segmented_overlay, 0.7, final_image, 0.3, 0)  # Use segmented_overlay instead of frame

            # Display the number of empty parking spaces
            number_of_empty_spaces = len(final_contours)
            cv2.putText(add_frame, f"Empty Parking Spaces: {number_of_empty_spaces}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Final Image', add_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                break
    else:
        # Optionally, delay the loop to save CPU resources
        time.sleep(max(0, interval - (current_time - last_time_processed)))

    # Break loop from the outer loop as well with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


