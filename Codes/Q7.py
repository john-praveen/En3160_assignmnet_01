import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv.imread('images/einstein.png', cv.IMREAD_GRAYSCALE)

# (a) Using filter2D with full 2D Sobel kernel (Gx)
sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
im_x = cv.filter2D(img, cv.CV_64F, sobel_x)
im_y = cv.filter2D(img, cv.CV_64F, sobel_y)
fig, ax = plt.subplots(1,2, sharex='all', sharey='all', figsize=(18,9))


ax[0].imshow(im_x, cmap='gray')
ax[0].set_title('Sobel X')
ax[0].set_xticks([]), ax[0].set_yticks([])
ax[1].imshow(im_y, cmap='gray')
ax[1].set_title('Sobel Y')
ax[1].set_xticks([]), ax[1].set_yticks([])
plt.show()
