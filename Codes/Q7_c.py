import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve_1d_vertical(img, kernel):
    """1D vertical convolution (correlation, no kernel flip) with zero-padding."""
    k = len(kernel)
    pad = k // 2
    padded = np.pad(img, ((pad, pad), (0, 0)), mode='constant', constant_values=0)
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0
            for ki in range(k):
                val += kernel[ki] * padded[i + ki, j]
            output[i, j] = val
    return output

def convolve_1d_horizontal(img, kernel):
    """1D horizontal convolution (correlation, no kernel flip) with zero-padding."""
    k = len(kernel)
    pad = k // 2
    padded = np.pad(img, ((0, 0), (pad, pad)), mode='constant', constant_values=0)
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0
            for kj in range(k):
                val += kernel[kj] * padded[i, j + kj]
            output[i, j] = val
    return output

# Load grayscale image
img = cv2.imread('images/einstein.png', cv2.IMREAD_GRAYSCALE)

# Sobel X separable kernels (matches your 2D kernel)
vertical_kernel_x = np.array([1, 2, 1], dtype=np.float32)
horizontal_kernel_x = np.array([1, 0, -1], dtype=np.float32)

# Sobel Y separable kernels
vertical_kernel_y = np.array([1, 0, -1], dtype=np.float32)
horizontal_kernel_y = np.array([1, 2, 1], dtype=np.float32)

# Compute Sobel X
intermediate_x = convolve_1d_vertical(img, vertical_kernel_x)
sobel_x = convolve_1d_horizontal(intermediate_x, horizontal_kernel_x)

# Compute Sobel Y
intermediate_y = convolve_1d_vertical(img, vertical_kernel_y)
sobel_y = convolve_1d_horizontal(intermediate_y, horizontal_kernel_y)

# Compute gradient magnitude
grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize for display (0-255)
def normalize_img(img):
    img_norm = (img - img.min()) / (img.max() - img.min()) * 255
    return img_norm.astype(np.uint8)

sobel_x_norm = normalize_img(sobel_x)
sobel_y_norm = normalize_img(sobel_y)
grad_mag_norm = normalize_img(grad_mag)

# Display results
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.title('Sobel X')
plt.imshow(sobel_x_norm, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Sobel Y')
plt.imshow(sobel_y_norm, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Gradient Magnitude')
plt.imshow(grad_mag_norm, cmap='gray')
plt.axis('off')

plt.show()
