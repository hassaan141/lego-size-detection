import cv2 as cv
import numpy as np
import math


def measure(image_path):
    # Read the image
    frame = cv.imread(image_path)
    if frame is None:
        print("Error: Could not read the image.")
        return

        # Resize frame for consistency
    frame = cv.resize(frame, (480, 480))

    # Convert to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define HSV range for red color (adjust as needed)
    red_lower = np.array([0, 150, 150])
    red_upper = np.array([10, 255, 255])
    red_lower_2 = np.array([160, 150, 150])
    red_upper_2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv.inRange(hsv, red_lower, red_upper)
    mask2 = cv.inRange(hsv, red_lower_2, red_upper_2)
    mask = cv.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)

        # Draw the bounding box
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print width and height
        print(f"Detected Brick - Width: {w} pixels, Height: {h} pixels")

    # Display results
    cv.imshow('Detected Red Brick', frame)
    cv.imshow('Red Mask', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Run the function with an example image
image_path = 'images/straight-red.jpg'  # Replace with your actual image path
measure(image_path)
