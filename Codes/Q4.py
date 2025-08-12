import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('Images/spider.png')  # replace with your path
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img_hsv)

# Parameters for intensity transform
sigma = 70
a = 0.7# You can tune this to get visually pleasing output

# Define the intensity transform function
def vibrance_transform(x, a, sigma=70):
    # x is input saturation intensity (0-255)
    # We'll apply vectorized operation for entire S channel
    term = a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2))
    return np.minimum(x + term, 255)

# Apply transformation to saturation channel
S_float = S.astype(np.float32)
S_vibrance = vibrance_transform(S_float, a, sigma)
S_vibrance = np.clip(S_vibrance, 0, 255).astype(np.uint8)

# Merge back channels
img_hsv_vibrance = cv2.merge([H, S_vibrance, V])
img_vibrance = cv2.cvtColor(img_hsv_vibrance, cv2.COLOR_HSV2BGR)

# Plot intensity transformation function for visualization
x_vals = np.arange(0, 256)
y_vals = vibrance_transform(x_vals, a, sigma)

# Display results
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(img_vibrance, cv2.COLOR_BGR2RGB))
plt.title(f'Vibrance Enhanced Image (a={a})')
plt.axis('off')

plt.subplot(2,2,3)
plt.plot(x_vals, y_vals, color='purple')
plt.title('Intensity Transformation Function')
plt.xlabel('Input Saturation')
plt.ylabel('Output Saturation')
plt.grid(True)


plt.tight_layout()
plt.show()
