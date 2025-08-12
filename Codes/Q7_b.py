import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/einstein.png', cv.IMREAD_GRAYSCALE)

def filter(image, filter):
    [rows, columns] = np.shape(image) # Get rows and columns of the image
    filtered_image = np.zeros(shape=(rows, columns)) # Create empty image
    
    for i in range(rows - 2):
        for j in range(columns - 2): # Process 2D convolution
            value = np.sum(np.multiply(filter, image[i:i + 3, j:j + 3])) 
            filtered_image[i + 1, j + 1] = value
    
    return filtered_image
  
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  
  # Apply the Sobel filter in the X direction
sobel_x_filtered = filter(img, sobel_x)

# Apply the Sobel filter in the Y direction
sobel_y_filtered = filter(img, sobel_y)

# Create the figure for plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].imshow(sobel_x_filtered, cmap='gray')
ax[0].set_title('Sobel X (Using custom function)')
ax[0].axis("off")
ax[1].imshow(sobel_y_filtered, cmap='gray')
ax[1].set_title('Sobel Y (Using custom function)')
ax[1].axis("off")

plt.tight_layout()
plt.show()