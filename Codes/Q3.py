import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"Images\highlights_and_shadows.jpg")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(img_lab)

# Normalize L channel and apply gamma correction
L_norm = L / 255.0
gamma = 0.5
L_gamma = np.power(L_norm, gamma)
L_corrected = np.uint8(L_gamma * 255)

# Merge corrected L channel back with a and b
img_lab_corrected = cv2.merge([L_corrected, a, b])
img_corrected = cv2.cvtColor(img_lab_corrected, cv2.COLOR_LAB2BGR)

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_corrected_rgb = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)

# Plot images and histograms
plt.figure(figsize=(16,8))

# Original Image
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Corrected Image
plt.subplot(2,2,2)
plt.imshow(img_corrected_rgb)
plt.title(f"Gamma Corrected Image (gamma={gamma})")
plt.axis('off')

# Histogram of original L channel
plt.subplot(2,2,3)
plt.hist(L.ravel(), bins=256, range=(0,255), color='blue')
plt.title('Original L channel Histogram')

# Histogram of corrected L channel
plt.subplot(2,2,4)
plt.hist(L_corrected.ravel(), bins=256, range=(0,255), color='red')
plt.title(f'Gamma Corrected L channel Histogram (gamma={gamma})')

plt.tight_layout()
plt.show()
