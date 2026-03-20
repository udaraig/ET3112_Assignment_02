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
# Question 3: Least-Squares-Fit Line
# ==========================================

# 1. Calculate the least-squares line (y = mx + c) using all points
# np.polyfit calculates the slope (m) and y-intercept (c)
m, c = np.polyfit(x, y, 1)

# 2. Generate x and y values to draw the red line on the graph
x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + c

# 3. Plot the scatter points AND the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=5, color='blue', label='Extracted Edges')
plt.plot(x_line, y_line, color='red', linewidth=2, label='Least-Squares Fit')

# Invert the y-axis so it visually matches the original crop field image
plt.gca().invert_yaxis() 
plt.title('Scatter Plot with Ordinary Least-Squares Fit (Q3)')
plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')
plt.legend()

# Display the final plot
plt.show()