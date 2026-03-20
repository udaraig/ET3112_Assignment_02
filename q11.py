import cv2 as cv
import numpy as np
from sklearn import linear_model

# 1. Re-run Question 1 logic to get the edge coordinates
img = cv.imread('img3.jpg.jpeg', 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1].reshape(-1, 1) # RANSAC needs a 2D array for X
y = indices[0]

# 2. Apply the RANSAC algorithm (Your Proposed Algorithm)
ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)

# 3. Get the robust slope (m) from the final model
m_ransac = ransac.estimator_.coef_[0]

# 4. Calculate the angle in degrees
angle_radians_ransac = np.arctan(m_ransac)
angle_degrees_ransac = np.degrees(angle_radians_ransac)

# 5. Print the final answer for Question 11
print("-" * 30)
print(f"RANSAC Slope (m): {m_ransac:.4f}")
print(f"FINAL ESTIMATED ANGLE: {angle_degrees_ransac:.2f} degrees")
print("-" * 30)
