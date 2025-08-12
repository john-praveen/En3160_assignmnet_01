import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread(r"Images\brain_proton_density_slice.png", cv2.IMREAD_GRAYSCALE)


mu = 150
sigma = 20
x = np.linspace(0, 255, 256)
gaussian= 255 * np.exp(-((x - mu)**2) / (2 * sigma**2))

gaussian = np.clip(gaussian, 0, 255)

print(gaussian.shape)

plt.figure(figsize=(5, 5))
plt.plot(gaussian)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()

white_matter = gaussian[img]

plt.figure(figsize=(5, 5))
plt.imshow(white_matter, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

mu = 200
sigma = 20
x = np.linspace(0, 255, 256)
gaussian= 255 * np.exp(-((x - mu)**2) / (2 * sigma**2))

gaussian = np.clip(gaussian, 0, 255)
grey_matter = gaussian[img]

plt.figure(figsize=(5, 5))
plt.imshow(grey_matter, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()