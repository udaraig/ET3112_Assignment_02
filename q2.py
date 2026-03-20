import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# --- Re-run Q1 logic to get x and y ---
img = cv.imread('img3.jpg.jpeg', 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1]
y = indices[0]

# ==========================================
# Question 2: Scatter Plot [cite: 29]
# ==========================================

plt.figure(figsize=(8, 6))

# Plot the extracted x and y coordinates
plt.scatter(x, y, s=5, color='blue')

# Important: Images have the origin (0,0) at the top-left. 
# We invert the y-axis so the plot visually matches the crop field image.
plt.gca().invert_yaxis() 

plt.title('Scatter Plot of Extracted Edge Features (Q2)')
plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')

# Display the plot
plt.show()