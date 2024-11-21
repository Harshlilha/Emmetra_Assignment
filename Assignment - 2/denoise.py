# Managing the imports
import cv2
import numpy as np

# File parameters
file_path = "C:/Users/ultim/Downloads/eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
width = 1920
height = 1280
bit_depth = 12  # Bits per pixel

# Read the raw file
with open(file_path, 'rb') as f:
    raw_data = np.frombuffer(f.read(), dtype=np.uint16)  # 12-bit packed into 16-bit

# Reshape into 2D array (height, width)
raw_image = raw_data.reshape((height, width))

# Normalize the 12-bit data to 8-bit for visualization
normalized_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply a Bayer pattern interpolation (demosaicing)
# OpenCV expects BGR output, convert the GRBG Bayer pattern
img_bgr = cv2.cvtColor(normalized_img, cv2.COLOR_BayerGR2BGR)

# Resize the image to 1280x760
img_bgr_resized = cv2.resize(img_bgr, (600, 300))

# Apply filtering operations on the resized image
# Gaussian Blur
output_gaus = cv2.GaussianBlur(img_bgr_resized, (5, 5), 0)

# Median Blur (reduction of noise)
output_med = cv2.medianBlur(img_bgr_resized, 5)

# Bilateral filtering (Reduction of noise + Preserving of edges)
output_bil = cv2.bilateralFilter(img_bgr_resized, 5, 6, 6)

# Displaying the images
cv2.imshow('Gaussian', output_gaus)
cv2.imshow('Median Blur', output_med)
cv2.imshow('Bilateral', output_bil)
cv2.imshow('Original Resized', img_bgr_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

