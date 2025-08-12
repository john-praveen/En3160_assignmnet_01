import cv2
import numpy as np

# Read the TIFF image (grayscale or color)
img = cv2.imread(r"Images\shells.tif", cv2.IMREAD_UNCHANGED)  

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
total_pixels = int(hist.sum())
new_hist=[]
calc=0

for i in range(len(hist)):
    calc=hist[i][0] + calc
    calc_norm=(calc/total_pixels)*255
    rounded_pix = np.round(calc_norm, decimals=0)
    new_hist.append([int(rounded_pix)])
lut = np.array(new_hist, dtype=np.uint8)

# Apply the LUT to the image for histogram equalization
equalized_img = lut[img]

# Save and display the equalized image
cv2.imwrite('equalized_shells.tif', equalized_img)
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

