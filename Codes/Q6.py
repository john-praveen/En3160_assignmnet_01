import cv2
import numpy as np
import matplotlib.pyplot as plt

# (a) Open image and split into HSV planes
img = cv2.imread(r'Images\jeniffer.jpg')  

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Display Hue, Saturation, and Value channels in grayscale
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title('Hue'); plt.imshow(h, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title('Saturation'); plt.imshow(s, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title('Value'); plt.imshow(v, cmap='gray'); plt.axis('off')
plt.show()

# (b) Select appropriate plane for foreground mask
# Usually, the 'Value' or 'Saturation' plane helps isolate foreground by thresholding.
# Let's try thresholding on 'Value' channel using Otsu's method

_, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure()
plt.title('Foreground Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# (c) Obtain foreground only using bitwise_and
foreground = cv2.bitwise_and(img, img, mask=mask)

# Compute histogram of foreground in grayscale (for simplicity, convert foreground to grayscale)
foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([foreground_gray], [0], mask, [256], [0,256])

plt.figure()
plt.title('Histogram of Foreground')
plt.plot(hist)
plt.xlim([0,256])
plt.show()

# (d) Obtain cumulative sum of the histogram
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]  # normalize to [0,1]

# (e) Histogram equalize foreground pixels using formula:
# new_pixel = round(cdf(old_pixel) * (L - 1)) where L=256

# Create lookup table for equalization based on foreground histogram CDF
lut = np.floor(cdf_normalized * 255).astype('uint8')

# Apply LUT to foreground grayscale image (only pixels in foreground mask)
equalized_foreground_gray = cv2.LUT(foreground_gray, lut)

# Convert equalized grayscale back to BGR to merge later
equalized_foreground = cv2.cvtColor(equalized_foreground_gray, cv2.COLOR_GRAY2BGR)

# (f) Extract background and combine with equalized foreground

# Background mask is inverse of foreground mask
background_mask = cv2.bitwise_not(mask)
background = cv2.bitwise_and(img, img, mask=background_mask)

# Combine background and equalized foreground
result = cv2.add(background, equalized_foreground)

# Show all required images
plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2,3,2)
plt.title('Hue')
plt.imshow(h, cmap='gray')
plt.axis('off')

plt.subplot(2,3,3)
plt.title('Saturation')
plt.imshow(s, cmap='gray')
plt.axis('off')

plt.subplot(2,3,4)
plt.title('Value')
plt.imshow(v, cmap='gray')
plt.axis('off')

plt.subplot(2,3,5)
plt.title('Foreground Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2,3,6)
plt.title('Histogram Equalized Foreground')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
