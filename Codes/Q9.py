import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('images/daisy.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize mask, bgdModel and fgdModel required by grabCut
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Rectangle around the foreground (manually set or computed)
rect = (30, 30, img.shape[1]-60, img.shape[0]-60)

# Apply GrabCut with rectangle mode
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask such that probable foreground and foreground pixels are 1, rest 0
grabcut_mask = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Extract foreground and background images using mask
foreground = img_rgb * grabcut_mask[:, :, np.newaxis]
background = img_rgb * (1 - grabcut_mask[:, :, np.newaxis])

# Display results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title('GrabCut Mask')
plt.imshow(grabcut_mask, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Foreground')
plt.imshow(foreground)
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Background')
plt.imshow(background)
plt.axis('off')
plt.show()


# Blur the background image
background_blurred = cv2.GaussianBlur(background, (21, 21), sigmaX=0)

# Combine blurred background with sharp foreground
enhanced_img = background_blurred + foreground

# Display original and enhanced images
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Enhanced Image (Blurred Background)')
plt.imshow(enhanced_img.astype(np.uint8))
plt.axis('off')
plt.show()
