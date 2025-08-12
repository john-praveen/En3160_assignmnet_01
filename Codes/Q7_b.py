import cv2
import numpy as np

def correlate2d(image, kernel):
    """
    Perform 2D correlation (no kernel flip) between an image and kernel using loops.

    Args:
      image: 2D numpy array (grayscale image)
      kernel: 2D numpy array (filter kernel)

    Returns:
      correlated_image: 2D numpy array same size as input image (with zero-padding)
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad image with zeros
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float32)

    # **Do NOT flip the kernel to mimic OpenCV filter2D**

    for i in range(img_h):
        for j in range(img_w):
            roi = padded_img[i:i+k_h, j:j+k_w]
            value = (roi * kernel).sum()  # correlation, no flipping
            output[i, j] = value

    return output

# Use your kernels exactly
sobel_x = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

sobel_y = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

# Load image
img = cv2.imread('images/einstein.png', cv2.IMREAD_GRAYSCALE)

# Apply manual Sobel filtering (correlation)
grad_x = correlate2d(img, sobel_x)
grad_y = correlate2d(img, sobel_y)

# Compute gradient magnitude
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag = np.clip((grad_mag / grad_mag.max()) * 255, 0, 255).astype(np.uint8)

# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Gradient X')
plt.imshow(grad_x, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Gradient Y')
plt.imshow(grad_y, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Gradient Magnitude')
plt.imshow(grad_mag, cmap='gray')
plt.axis('off')

plt.show()
