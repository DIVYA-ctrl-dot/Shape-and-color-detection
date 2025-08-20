# Shape-and-color-detection
Shape and color detection is a fundamental aspect of computer vision, enabling machines to understand and interpret visual information from images and videos. This project focuses on detecting shapes and colors using OpenCV.

import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color ranges
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'green': [(40, 100, 100), (80, 255, 255)],
    'blue': [(110, 100, 100), (130, 255, 255)]
}

# Detect colors
for color, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
            cv2.putText(img, color, (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Detect shapes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(approx) == 4:
            cv2.putText(img, 'Rectangle', (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(approx) > 5:
            cv2.putText(img, 'Circle', (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the output
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

