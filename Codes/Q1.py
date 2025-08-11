import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"Images\emma.jpg"  
img = cv2.imread(image_path)

def pixelVal(pix, r1, s1, r2, s2, s3, s4):
    if (0 <= pix and pix < r1):
        return (s1 / r1) * pix
    elif pix == r1:
        return s2
    elif (r1 < pix and pix < r2):
        return ((s3 - s1) / (r2 - r1)) * (pix - r1) + s2
    elif pix == r2:
        return s4
    else:
        return ((255 - s4) / (255 - r2)) * (pix - r2) + s4

r1, s1 = 50, 50
r2, s2 = 150, 100
s3, s4 = 255, 150

pixelVal_vec = np.vectorize(pixelVal)
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2, s3, s4)

# Convert to uint8 before saving
contrast_stretched = np.clip(contrast_stretched, 0, 255).astype(np.uint8)
cv2.imwrite('contrast_stretch.jpg', contrast_stretched)

# Read for plotting
img = cv2.imread('contrast_stretch.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()
