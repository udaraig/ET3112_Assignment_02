import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

# --- Re-run logic to get coordinates ---
img = cv.imread(r"C:\Users\UDARA\OneDrive\Desktop\ET3112_Assignment_02\img3.jpg.jpeg", 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1]
y = indices[0]

# --- Total Least-Squares (TLS) Calculation using PCA ---
points = np.column_stack((x, y))
pca = PCA(n_components=2)
pca.fit(points)

# Get the slope (dy/dx) from the first principal component
direction_vector = pca.components_[0]
m_tls = direction_vector[1] / direction_vector[0]

# --- Question 7: Calculate Angle ---
angle_radians_tls = np.arctan(m_tls)
angle_degrees_tls = np.degrees(angle_radians_tls)

print(f"TLS Calculated Slope (m_tls): {m_tls:.4f}")
print(f"TLS Estimated Crop Field Angle: {angle_degrees_tls:.2f} degrees")
