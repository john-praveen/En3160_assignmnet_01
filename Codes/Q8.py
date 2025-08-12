import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def zoom_image(img, factor, method="nearest"):
    if method == "nearest":
        interp = cv.INTER_NEAREST
    elif method == "bilinear":
        interp = cv.INTER_LINEAR
    else:
        raise ValueError("Choose 'nearest' or 'bilinear'")
    new_w = int(img.shape[1] * factor)
    new_h = int(img.shape[0] * factor)
    return cv.resize(img, (new_w, new_h), interpolation=interp)

def normalized_ssd(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    ssd = np.sum((img1 - img2) ** 2)
    return ssd / img1.size

original = cv.imread("images/im01.png", cv.IMREAD_COLOR)
small = cv.imread("images/im01small.png", cv.IMREAD_COLOR)

factor = 4
zoomed_nn = zoom_image(small, factor, method="nearest")
zoomed_bi = zoom_image(small, factor, method="bilinear")

original_resized = cv.resize(original, (zoomed_nn.shape[1], zoomed_nn.shape[0]))

ssd_nn = normalized_ssd(zoomed_nn, original_resized)
ssd_bi = normalized_ssd(zoomed_bi, original_resized)

print(f"Normalized SSD (Nearest Neighbor): {ssd_nn:.4f}")
print(f"Normalized SSD (Bilinear): {ssd_bi:.4f}")



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(zoomed_nn, cv.COLOR_BGR2RGB))
plt.title("Zoomed Nearest")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(zoomed_bi, cv.COLOR_BGR2RGB))
plt.title("Zoomed Bilinear")
plt.axis('off')

plt.tight_layout()
plt.show()
