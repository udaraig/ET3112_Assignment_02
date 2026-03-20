import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Re-run Q1 logic to get x and y ---
img = cv.imread(r"C:\Users\UDARA\OneDrive\Desktop\ET3112_Assignment_02\img3.jpg.jpeg", 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1]
y = indices[0]

# ==========================================
# Question 6: Total Least-Squares-Fit Line
# ==========================================

# 1. Prepare data for PCA (combine x and y into a 2D array of points)
points = np.column_stack((x, y))

# 2. Fit PCA to find the Total Least-Squares line
pca = PCA(n_components=2)
pca.fit(points)

# The first principal component gives the line's primary direction
direction_vector = pca.components_[0]

# Calculate the slope (m = dy/dx)
m_tls = direction_vector[1] / direction_vector[0]

# The TLS line always passes exactly through the mean (average) of all points
mean_x, mean_y = pca.mean_

# Calculate the intercept (c = y - mx)
c_tls = mean_y - m_tls * mean_x

# 3. Generate x and y values to draw the green line on the graph
x_line = np.linspace(min(x), max(x), 100)
y_line_tls = m_tls * x_line + c_tls

# 4. Plot the scatter points AND the TLS fitted line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=5, color='blue', label='Extracted Edges')
plt.plot(x_line, y_line_tls, color='green', linewidth=2, label='Total Least-Squares Fit')

# Invert the y-axis so it visually matches the original crop field image
plt.gca().invert_yaxis() 
plt.title('Scatter Plot with Total Least-Squares Fit (Q6)')
plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')
plt.legend()

# Display the final plot
plt.show()