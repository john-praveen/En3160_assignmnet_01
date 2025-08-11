import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread(r"Images\brain_proton_density_slice.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found! Check the file path.")

def piecewise_transform(pix, r1, s1, r2, s2):
    if pix < r1:
        return (s1 / r1) * pix
    elif r1 <= pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

vec_transform = np.vectorize(piecewise_transform)

# White matter enhancement
r1_w, s1_w = 150, 30
r2_w, s2_w = 190, 200
white_matter_img = vec_transform(img, r1_w, s1_w, r2_w, s2_w)
white_matter_img = np.clip(white_matter_img, 0, 255).astype(np.uint8)

# Gray matter enhancement
r1_g, s1_g = 80, 20
r2_g, s2_g = 120, 220
gray_matter_img = vec_transform(img, r1_g, s1_g, r2_g, s2_g)
gray_matter_img = np.clip(gray_matter_img, 0, 255).astype(np.uint8)

# Plot transformation curves
x = np.arange(0, 256)
white_curve = [piecewise_transform(i, r1_w, s1_w, r2_w, s2_w) for i in x]
gray_curve = [piecewise_transform(i, r1_g, s1_g, r2_g, s2_g) for i in x]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, white_curve, label="White Matter Enhancement", color='blue')
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, gray_curve, label="Gray Matter Enhancement", color='red')
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.legend()
plt.grid(True)
plt.show()

# Show images
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(1, 3, 2), plt.imshow(white_matter_img, cmap='gray'), plt.title("White Matter Enhanced"), plt.axis('off')
plt.subplot(1, 3, 3), plt.imshow(gray_matter_img, cmap='gray'), plt.title("Gray Matter Enhanced"), plt.axis('off')
plt.show()
