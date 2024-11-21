import cv2
import numpy as np
import os

def load_raw_image(file_path, width, height, bit_depth):
    """
    Load a raw image, normalize it, and return as an 8-bit grayscale image.
    Assumes the raw image is in 12-bit format, packed into 16-bit containers.
    """
    # Read the raw image data as 16-bit (2 bytes per pixel)
    with open(file_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)  # Read 16-bit data
    
    # Extract the 12-bit data (use lower 12 bits of each 16-bit value)
    raw_image = raw_data & 0x0FFF  # Mask the lower 12 bits

    # Reshape into 2D array (height, width)
    raw_image = raw_image.reshape((height, width))

    # Normalize the 12-bit data to 8-bit for visualization
    normalized_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_img

# File parameters
file_path = "C:/Users/ultim/Downloads/eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
width = 1920
height = 1280
bit_depth = 12  # Bits per pixel

# Load the raw image
raw_image = load_raw_image(file_path, width, height, bit_depth)

# Convert to grayscale (already grayscale, but ensure compatibility)
gray_frame = raw_image

# Apply Gaussian Blur
blur_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

# Apply Laplacian for edge detection
laplace_frame = cv2.Laplacian(blur_frame, cv2.CV_64F)

# Resize Laplacian frame
percent_scale = 0.4
width_frame = int(laplace_frame.shape[1] * percent_scale)
height_frame = int(laplace_frame.shape[0] * percent_scale)
dimension_frame = (width_frame, height_frame)
scaled_laplace = cv2.resize(laplace_frame, dimension_frame, interpolation=cv2.INTER_AREA)

# Save the Laplacian image in the current directory
output_path = os.path.join(os.getcwd(), "laplacian_image.jpg")
cv2.imwrite(output_path, scaled_laplace.astype(np.uint8))
print(f"Laplacian image saved at: {output_path}")

# Resize original frame
width_frame = int(gray_frame.shape[1] * percent_scale)
height_frame = int(gray_frame.shape[0] * percent_scale)
dimension_frame = (width_frame, height_frame)
scaled_original = cv2.resize(gray_frame, dimension_frame, interpolation=cv2.INTER_AREA)

# Display frames
cv2.imshow("Laplace Frame", scaled_laplace)
cv2.imshow("Original Frame", scaled_original)

# Wait for any key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
