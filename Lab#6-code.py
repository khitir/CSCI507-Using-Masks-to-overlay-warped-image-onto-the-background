# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:17:43 2024

@author: zezva
"""

import cv2
import numpy as np

# Read the images
I1 = cv2.imread("C:\\Users\\zezva\Desktop\\Lab#6\\US_Cellular.jpg")  # US Cellular field image
I2 = cv2.imread("C:\\Users\\zezva\\Desktop\\Lab#6\\Mines_Logo.jpg")   # Mines Logo image

# Read the points from pts.dat
points = []
with open("C:\\Users\\zezva\\Desktop\\Lab#6\\pts.dat", 'r') as file:
    for line in file:
        points.append(list(map(float, line.strip().split())))

points = np.array(points, dtype=np.float32)

# Define the points
stadium_points = points[:4]  # Points from the stadium image
logo_points = points[4:]     # Corresponding points from the logo image

# Compute the perspective transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(logo_points, stadium_points)

# Warp the logo image to fit into the stadium image
warped_logo = cv2.warpPerspective(I2, transformation_matrix, (I1.shape[1], I1.shape[0]))

# Create a mask for the logo
mask = np.zeros_like(I1)
cv2.fillPoly(mask, [stadium_points.astype(np.int32)], (255, 255, 255))

# Invert the mask to get the background mask
background_mask = cv2.bitwise_not(mask)

# Combine the warped logo with the stadium image
final_image = cv2.bitwise_and(I1, background_mask) + cv2.bitwise_and(warped_logo, mask)

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image
cv2.imwrite("C:\\Users\\zezva\\Desktop\\Lab#6\\final_image.jpg", final_image)
