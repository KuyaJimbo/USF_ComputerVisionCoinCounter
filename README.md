# Coin Detection and Classification

This project processes images of US coins to detect their positions and classify them by type (quarters, dimes, nickels, and pennies). It uses computer vision techniques and machine learning to identify and classify coins based on their visual properties.

## Overview

The system works by:
1. Processing an input image to segment and identify individual coins
2. Extracting features from each coin (size, color, radius)
3. Using a machine learning model to classify each coin into the correct denomination
4. Outputting the coordinates and value of each detected coin

## Features

- **Image Processing Pipeline**: Uses OpenCV for image processing, thresholding, and segmentation
- **Machine Learning Classification**: Applies PCA and Logistic Regression to classify coins based on extracted features
- **Watershed Algorithm**: Implements watershed algorithm to separate touching coins
- **BFS Traversal**: Uses breadth-first search to identify and label individual coins

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- joblib

## Project Structure

- **project3.py**: Main script that processes images and identifies coins
- **train.py**: Script used to train the machine learning models for coin classification
- **scaler.joblib**: Saved StandardScaler model for feature normalization
- **pca.joblib**: Saved PCA model for dimensionality reduction
- **classifier.joblib**: Saved LogisticRegression model for coin classification

## How It Works

### Image Processing Pipeline

1. **Image Preprocessing**:
   - Resize the image for faster processing
   - Convert to grayscale
   - Apply Gaussian blur to reduce noise

2. **Segmentation**:
   - Apply Otsu thresholding to separate coins from background
   - Use morphological operations to clean up the image
   - Implement watershed algorithm to separate touching coins

3. **Coin Detection**:
   - Perform BFS to identify connected components (coins)
   - Extract features (size, radius, color) for each coin
   - Filter out components that are too small or too large to be coins

### Machine Learning Classification

The classification system uses:
- **Feature Extraction**: Size (pixel count), radius, and RGB color values
- **Feature Processing**: StandardScaler for normalization and PCA for dimensionality reduction
- **Classification Model**: Logistic Regression to classify coins as quarters (25¢), dimes (10¢), nickels (5¢), or pennies (1¢)

## Usage

```bash
python project3.py
```

When prompted, enter the path to the image file you want to process.

## Output Format

The program outputs:
1. The total number of coins detected
2. For each coin: X-coordinate, Y-coordinate, and value (in cents)

## Machine Learning Implementation

Machine learning is used specifically for coin classification. The training process (in train.py) involves:
1. Collecting training data with features from various coins
2. Standardizing features using StandardScaler
3. Reducing dimensionality with PCA
4. Training a LogisticRegression classifier
5. Saving the models using joblib for later use in the main program

## Performance Considerations

- The system balances accuracy with processing speed
- Various constants in the code can be tuned to optimize for different images
- Processing time is constrained to under 20 seconds per image
