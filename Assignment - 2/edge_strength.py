import cv2
import numpy as np
import os

def load_raw_image(file_path, width, height, bit_depth):
    """
    Load a raw image, normalize it, and return as an 8-bit grayscale image.
    Assumes the raw image is in 12-bit format, packed into 16-bit containers.
    """
    with open(file_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)
    
    # Extract 12-bit data
    raw_image = raw_data & 0x0FFF
    raw_image = raw_image.reshape((height, width))

    # Normalize the 12-bit data to 8-bit
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

# Compute the gradients using Sobel operator
sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

# Compute the gradient magnitude (edge strength)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalize the gradient magnitude for visualization
gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save the gradient magnitude image
output_path = os.path.join(os.getcwd(), "gradient_magnitude_image.jpg")
cv2.imwrite(output_path, gradient_magnitude_normalized)  # Save as an 8-bit image
print(f"Gradient magnitude image saved at: {output_path}")
percent_scale = 0.4
width_frame = int(gray_frame.shape[1] * percent_scale)
height_frame = int(gray_frame.shape[0] * percent_scale)
dimension_frame = (width_frame, height_frame)
gray_frame = cv2.resize(gray_frame, dimension_frame, interpolation=cv2.INTER_AREA)

percent_scale = 0.4
width_frame = int(gradient_magnitude_normalized.shape[1] * percent_scale)
height_frame = int(gradient_magnitude_normalized.shape[0] * percent_scale)
dimension_frame = (width_frame, height_frame)
gradient_magnitude_normalized = cv2.resize(gradient_magnitude_normalized, dimension_frame, interpolation=cv2.INTER_AREA)

# Display the original and gradient magnitude images
cv2.imshow("Original Image", gray_frame)
cv2.imshow("Edge Strength (Gradient Magnitude)", gradient_magnitude_normalized)

# Wait until a key is pressed and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
