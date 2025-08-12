import cv2
import numpy as np

def zoom_image(img, scale_factor, method='nearest'):
    """
    Zoom image by scale_factor using 'nearest' or 'bilinear' interpolation.

    Args:
      img: input image (numpy array)
      scale_factor: float in (0, 10]
      method: 'nearest' or 'bilinear'

    Returns:
      zoomed_img: scaled image
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    zoomed_img = np.zeros((new_h, new_w, *img.shape[2:]), dtype=img.dtype) if img.ndim == 3 else np.zeros((new_h, new_w), dtype=img.dtype)

    if method == 'nearest':
        for i in range(new_h):
            for j in range(new_w):
                # Map back to source pixel coordinates
                src_x = min(int(round(i / scale_factor)), h - 1)
                src_y = min(int(round(j / scale_factor)), w - 1)
                zoomed_img[i, j] = img[src_x, src_y]

    elif method == 'bilinear':
        for i in range(new_h):
            for j in range(new_w):
                # Map coordinates in original image
                x = i / scale_factor
                y = j / scale_factor

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, h - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, w - 1)

                dx = x - x0
                dy = y - y0

                if img.ndim == 3:  # color image
                    top_left = img[x0, y0].astype(np.float32)
                    top_right = img[x0, y1].astype(np.float32)
                    bottom_left = img[x1, y0].astype(np.float32)
                    bottom_right = img[x1, y1].astype(np.float32)

                    top = top_left * (1 - dy) + top_right * dy
                    bottom = bottom_left * (1 - dy) + bottom_right * dy
                    pixel = top * (1 - dx) + bottom * dx

                    zoomed_img[i, j] = np.clip(pixel, 0, 255).astype(img.dtype)
                else:  # grayscale
                    top_left = float(img[x0, y0])
                    top_right = float(img[x0, y1])
                    bottom_left = float(img[x1, y0])
                    bottom_right = float(img[x1, y1])

                    top = top_left * (1 - dy) + top_right * dy
                    bottom = bottom_left * (1 - dy) + bottom_right * dy
                    pixel = top * (1 - dx) + bottom * dx

                    zoomed_img[i, j] = np.clip(pixel, 0, 255).astype(img.dtype)

    else:
        raise ValueError("Method must be 'nearest' or 'bilinear'")

    return zoomed_img


def normalized_ssd(img1, img2):
    """
    Compute normalized sum of squared difference (SSD) between two images.

    Args:
      img1, img2: images (same shape)

    Returns:
      normalized SSD scalar
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    ssd = np.sum(diff ** 2)
    norm_ssd = ssd / (img1.size)
    return norm_ssd


# Example usage:
if __name__ == '__main__':
    # Load original large image and small image
    original = cv2.imread('Images\im01.png', cv2.IMREAD_COLOR)
    small = cv2.imread('Images\im01small.png', cv2.IMREAD_COLOR)

    scale_factor = 4

    # Zoom small image back up using nearest neighbor
    zoomed_nearest = zoom_image(small, scale_factor, method='nearest')
    # Zoom small image back up using bilinear interpolation
    zoomed_bilinear = zoom_image(small, scale_factor, method='bilinear')

    # Crop or resize original to match zoomed images shape if needed
    h, w = zoomed_nearest.shape[:2]
    original_cropped = original[:h, :w]

    # Compute normalized SSD
    ssd_nearest = normalized_ssd(zoomed_nearest, original_cropped)
    ssd_bilinear = normalized_ssd(zoomed_bilinear, original_cropped)

    print(f'Normalized SSD (Nearest Neighbor): {ssd_nearest:.4f}')
    print(f'Normalized SSD (Bilinear): {ssd_bilinear:.4f}')

    # Save or show images
    cv2.imwrite('zoomed_nearest.png', zoomed_nearest)
    cv2.imwrite('zoomed_bilinear.png', zoomed_bilinear)

    cv2.imshow('Original', original_cropped)
    cv2.imshow('Zoomed Nearest', zoomed_nearest)
    cv2.imshow('Zoomed Bilinear', zoomed_bilinear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
