import cv2 as cv
import numpy as np

# --- Re-run Q1 logic to get x and y ---
img = cv.imread(r"C:\Users\UDARA\OneDrive\Desktop\ET3112_Assignment_02\img3.jpg.jpeg", 0) 
edges = cv.Canny(img, 550, 690) 
indices = np.where(edges != 0) 
x = indices[1]
y = indices[0]

# --- Re-run Q3 logic to get the slope (m) ---
m, c = np.polyfit(x, y, 1)

# ==========================================
# Question 4: Estimated Angle
# ==========================================

# Calculate the angle in radians using inverse tangent
angle_radians = np.arctan(m)

# Convert the angle from radians to degrees for easier reading
angle_degrees = np.degrees(angle_radians)

print(f"Calculated Slope (m): {m:.4f}")
print(f"Estimated Crop Field Angle: {angle_degrees:.2f} degrees")