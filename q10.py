import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# --- Re-run logic to get coordinates ---
img = cv.imread(r"c:\Users\UDARA\OneDrive\Desktop\ET3112_Assignment_02\img3.jpg.jpeg", 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1].reshape(-1, 1) # RANSAC needs 2D array for X
y = indices[0]

# ==========================================
# Question 10: RANSAC (Proposed Algorithm)
# ==========================================

# 1. Initialize and fit the RANSAC model
ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)

# 2. Identify Inliers vs Outliers (for plotting)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 3. Predict points for the line
line_x = np.arange(x.min(), x.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_x)

# 4. Plotting the results
plt.figure(figsize=(8, 6))

# Plot Outliers in light gray and Inliers in blue
plt.scatter(x[outlier_mask], y[outlier_mask], color='lightgray', s=5, label='Outliers (Noise)')
plt.scatter(x[inlier_mask], y[inlier_mask], color='blue', s=5, label='Inliers (Crop Row)')

# Plot the RANSAC line in gold/yellow
plt.plot(line_x, line_y_ransac, color='gold', linewidth=3, label='RANSAC Robust Fit')

plt.gca().invert_yaxis() 
plt.title('Question 10: Robust Estimation using RANSAC')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.show()

# Print the robust angle
m_ransac = ransac.estimator_.coef_[0]
angle_ransac = np.degrees(np.arctan(m_ransac))
print(f"RANSAC Estimated Angle: {angle_ransac:.2f} degrees")