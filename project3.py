# James Ocampo
# U69643093
# Machine Learning was used for only Coin Classification, please see train.py for the training process

import cv2
import numpy as np
import joblib
import os

# ------------ Constants I May Change ------------
# Step 1: Read, scale down, make it grayscale, and blur the image to prepare for processing
RESIZE = 0.155
BLUR_SIZE = (5, 5)
# Step 2: Color the coins white and the background black with Otsu Thresholding
# Step 3: Morph filters remove remaining noise within the coin and random white spots of noise)
MORPHS = 2
# Step 4: Find boundaries between close coins using watershed 
BG_DILATIONS = 3
DIST_MASK_SIZE = 5
# Step 5: More Morphological filtering to better separate the coins by bridging the borders of close coins
# Step 6: Set up BFS for labeling the new mask
MIN_COIN = 310
MAX_DIME = 468
MAX_PENNY = 570
MAX_NICKLE = 732
MAX_QUARTER = 950
# Step 7: Do BFS with optimal step size
# Step 8: print the information from the identified coins
# ------------------------------------------------

# Kernel used for morphological operations
small_kernel = np.ones((3, 3), np.uint8)

# Function to predict the class of a coin
def predict_features(features, scaler, pca, clf):
    features = np.array(features, dtype=np.float64).reshape(1, -1)  # Reshape features
    features_scaled = scaler.transform(features)                    # Scale features
    features_pca = pca.transform(features_scaled)                   # Apply PCA
    return clf.predict(features_pca)[0]                             # Predict class

# Function to perform BFS on the image
def bfs(start_x, start_y, ID, new_mask, reduced_image, marked, coin_dict, height, width, scaler, pca, clf):
    queue = [(start_x, start_y)]                                    # Initialize queue
    count, total_rows, total_columns = 0, 0, 0                      # Initialize counters
    total_blue, total_green, total_red = 0, 0, 0                    # Initialize color counters
    left_extreme, right_extreme = start_y, start_y                  # Initialize extreme values
    top_extreme, bottom_extreme = start_x, start_x                  # Initialize extreme values

    while queue:                                                            # While queue is not empty
        x, y = queue.pop(0)                                                 # Pop the first element
        if 0 <= x < height and 0 <= y < width and new_mask[x, y] != 255:    # Check if pixel is within bounds and not white
            if (x,y) not in marked:                                         # Check if pixel is not marked   
                blue, green, red = reduced_image[x, y]                      # Get pixel color
                # Update color counters (int for overflow)
                total_red += int(red)                                       # Update color counters (int for overflow)
                total_green += int(green)                                   # Update color counters (int for overflow)
                total_blue += int(blue)                                     # Update color counters (int for overflow)
                # Keep track of extremes for radius calculation
                left_extreme = min(left_extreme, y)                         # Update extreme values
                right_extreme = max(right_extreme, y)                       # Update extreme values
                top_extreme = min(top_extreme, x)                           # Update extreme values
                bottom_extreme = max(bottom_extreme, x)                     # Update extreme values
                # Update Pixel counter
                count += 1
                # Update total rows and columns for center calculation
                total_rows += y
                total_columns += x
                marked[(x,y)] = True                                        # Mark the pixel
                queue.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])      # Add adjacent pixels to queue

    # Calculate center and average color values (eliminate false coins)
    if count < MIN_COIN or count > MAX_QUARTER:
        return

    # Calculate center
    center_x = round(total_rows / count)
    center_y = round(total_columns / count)
    avg_blue = round(total_blue / count)
    # Calculate average color values
    avg_green = round(total_green / count)
    avg_red = round(total_red / count)
    # Calculate average radius
    up = center_x - top_extreme
    down = bottom_extreme - center_x
    left = center_y - left_extreme
    right = right_extreme - center_y
    avg_radius = round((up + down + left + right) / 4)
    # Predict the class of the coin
    features = np.array([[count, avg_radius, avg_blue, avg_green, avg_red]], dtype=np.float64)
    classification = predict_features(features, scaler, pca, clf)
    # Update coin dictionary
    coin_dict[ID] = [count, center_x, center_y, avg_radius, avg_blue, avg_green, avg_red, classification]
    
    # This was used to collect training data for train.py
    # print(ID, count, avg_radius, avg_blue, avg_green, avg_red)

# This function processes the image and prints the results
def process_image(image_path):
    # Load models (Machine Learning)
    scaler = joblib.load('scaler.joblib')       # Load scaler (Machine Learning)
    pca = joblib.load('pca.joblib')             # Load PCA (Machine Learning)
    clf = joblib.load('classifier.joblib')      # Load classifier (Machine Learning)
    
    # --------------------- EXACT SAME FROM PROJECT 1 --------------------- 
    # Read and process image
    image = cv2.imread(image_path)
    reduced_image = cv2.resize(image, (0, 0), fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_LINEAR)     # Resize image to reduce processing time and help with noise
    grayscale_image = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2GRAY)                                   # Convert to grayscale
    blurred_image = cv2.GaussianBlur(grayscale_image, BLUR_SIZE, 0)                                     # Blur image to reduce noise further

    # Otsu Thresholding
    ret, otsu_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Morphological operations
    dilated = cv2.dilate(otsu_image, small_kernel, iterations=2)    # Dilate to remove noise
    eroded = cv2.erode(dilated, small_kernel, iterations=2)         # Erode to remove noise                                       

    # Watershed segmentation
    definite_bg = cv2.dilate(eroded, small_kernel, iterations=BG_DILATIONS)                        # Dilate to remove noise
    distance_to_black = cv2.distanceTransform(eroded, cv2.DIST_L2, DIST_MASK_SIZE)                 # Distance transform
    thresh, definite_fg = cv2.threshold(distance_to_black, 0.5 * distance_to_black.max(), 255, 0)  # Threshold

    definite_fg = np.uint8(definite_fg)                             # Convert to uint8
    boundary = cv2.subtract(definite_bg, definite_fg)               # Subtract to get boundary

    ret, markers = cv2.connectedComponents(definite_fg)             # Connected components
    markers = markers + 1                                           # Add 1 to markers
    markers[boundary == 255] = 0                                    # Set boundary to 0
    markers = cv2.watershed(reduced_image, markers)                 # Watershed

    # Border processing
    borders = markers.astype(np.uint8)                              # Convert to uint8
    borders = cv2.dilate(borders, small_kernel, iterations=3)       # Dilate to remove noise
    borders = cv2.erode(borders, small_kernel, iterations=2)        # Erode to remove noise
    borders = cv2.dilate(borders, small_kernel, iterations=3)       # Dilate to remove noise
    borders = cv2.erode(borders, small_kernel, iterations=1)        # Erode to remove noise
    borders = cv2.dilate(borders, small_kernel, iterations=1)       # Dilate to remove noise
    new_mask = cv2.erode(borders, small_kernel, iterations=3)       # Erode to remove noise

    # Coin detection
    height, width = new_mask.shape  # Get image dimensions
    ID = -1                         # Initialize ID
    marked = {}                     # Initialize marked pixels
    coin_dict = {}                  # Initialize coin dictionary

    for x in range(0, height, 20):                                  # Loop through image
        for y in range(0, width, 20):                               # Loop through image
            if new_mask[x, y] != 255 and (x,y) not in marked:       # Check if pixel is not white and not marked
                ID += 1                                             # Increment ID
                bfs(x, y, ID, new_mask, reduced_image, marked, coin_dict, height, width, scaler, pca, clf)  # Perform BFS

    # Print results
    print(len(coin_dict))
    for coin in coin_dict:
        count, center_x, center_y, avg_radius, avg_blue, avg_green, avg_red, classification = coin_dict[coin]   # Get coin info
        resized_x = int(center_x / RESIZE)  # Resize x coordinate to account for initial resizing
        resized_y = int(center_y / RESIZE)  # Resize y coordinate to account for initial resizing
        print(resized_x, resized_y, classification) # Print coin info

def main():
    image_file = input()        # Get image file
    process_image(image_file)   # Process image

if __name__ == "__main__":
    main()