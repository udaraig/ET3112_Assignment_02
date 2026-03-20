import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Step A: Load the cropped image
# Make sure to replace 'cropped_image.jpg' with the actual file name of Fig 1c.
# We load it in grayscale (0) because Canny requires a single-channel image.
img = cv.imread(r"C:\Users\UDARA\OneDrive\Desktop\ET3112_Assignment_02\img3.jpg.jpeg", 0) 

# Step B: Apply the Canny Edge Detector
# The assignment strictly requires minVal=550 and maxVal=690[cite: 8, 14].
edges = cv.Canny(img, 550, 690) 

# Step C: Plot the Original and Edge Images
# Using matplotlib's imshow function as requested.
plt.figure(figsize=(10, 5))

# Plotting the original cropped image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Cropped Image')

# Plotting the edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detector Output')

# Display the plots side-by-side
plt.show()

# Step D: Extract Feature Coordinates
# We find all pixels in the 'edges' array that are not zero (meaning they are edges).
# Note: The assignment PDF has a small typo here ("edges! [01")[cite: 15], 
# the correct Python syntax is "edges != 0".
indices = np.where(edges != 0) 

# Assign the extracted positions to x and y coordinates[cite: 13].
x = indices[1] # [cite: 18]
y = indices[0] # [cite: 19]

print(f"Successfully extracted {len(x)} edge coordinates.")