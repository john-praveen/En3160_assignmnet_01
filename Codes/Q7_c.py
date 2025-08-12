import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv.imread('images/einstein.png', cv.IMREAD_GRAYSCALE)

# 1D Sobel kernels (separable form)
kx = np.array([[1, 0, -1]], dtype=np.float32)    # horizontal derivative
ky = np.array([[1], [2], [1]], dtype=np.float32) # vertical smoothing

# For Sobel Y, the order is reversed:
kx_y = np.array([[1], [0], [-1]], dtype=np.float32) # vertical derivative
ky_y = np.array([[1, 2, 1]], dtype=np.float32)      # horizontal smoothing

# --- Sobel X ---
temp_x = cv.filter2D(img, cv.CV_32F, kx)     # horizontal derivative
sobel_x = cv.filter2D(temp_x, cv.CV_32F, ky) # vertical smoothing

# --- Sobel Y ---
temp_y = cv.filter2D(img, cv.CV_32F, ky_y)   # horizontal smoothing
sobel_y = cv.filter2D(temp_y, cv.CV_32F, kx_y) # vertical derivative



# Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel X (Separable)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Y (Separable)")
plt.axis('off')

plt.tight_layout()
plt.show()
