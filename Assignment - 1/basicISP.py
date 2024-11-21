import streamlit as st
import numpy as np
from skimage import filters
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance
from skimage import color
import cv2


# Bayer Demosaicing
def demosaic(image):
    h, w = image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint16)

    # Assuming a GRBG Bayer pattern
    rgb_image[0::2, 0::2, 1] = image[0::2, 0::2]  # Green pixels
    rgb_image[0::2, 1::2, 0] = image[0::2, 1::2]  # Red pixels
    rgb_image[1::2, 0::2, 2] = image[1::2, 0::2]  # Blue pixels
    rgb_image[1::2, 1::2, 1] = image[1::2, 1::2]  # Green pixels

    # Interpolate missing values
    rgb_image[:, :, 0] = gaussian_filter(rgb_image[:, :, 0], sigma=1)
    rgb_image[:, :, 1] = gaussian_filter(rgb_image[:, :, 1], sigma=1)
    rgb_image[:, :, 2] = gaussian_filter(rgb_image[:, :, 2], sigma=1)

    return rgb_image


# White Balance
def white_balance(image):
    avg = np.mean(image, axis=(0, 1))
    gain = np.mean(avg) / avg
    balanced = image * gain
    return np.clip(balanced, 0, 65535).astype(np.uint16)


# Denoising
def denoise(image, sigma=1.0):
    return gaussian_filter(image, sigma=(sigma, sigma, 0))


# Gamma Correction
def gamma_correction(image, gamma=2.2, saturation_factor=1.5):
    normalized = image / np.max(image)
    gamma_corrected = np.power(normalized, 1 / gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    pil_img = Image.fromarray(gamma_corrected)
    enhancer = ImageEnhance.Color(pil_img)
    enhanced_img = enhancer.enhance(saturation_factor)
    return np.array(enhanced_img)

def simplest_color_balance(img, percent):
    """Apply simplest color balance using percentiles."""
    out_channels = []
    channels = cv2.split(img)
    for channel in channels:
        low_val, high_val = np.percentile(channel, [percent, 100 - percent])
        channel = np.clip(channel, low_val, high_val)
        channel = (channel - low_val) * (255 / (high_val - low_val))
        out_channels.append(channel)
    return cv2.merge(out_channels)

# Sharpen
def sharpen(image, radius=1.0, amount=1.0, color_balance_percent=1):
    # Step 1: Sharpen the Image
    pil_img = Image.fromarray(image.astype(np.uint8))
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharpened_img = np.array(enhancer.enhance(amount))

    # Step 2: Apply Simplest Color Balance
    if color_balance_percent > 0:
        sharpened_img = simplest_color_balance(sharpened_img, color_balance_percent)

    return sharpened_img



# Controlled Edge Enhancement
def enhance_edges(image, edge_weight=0.5, blur_radius=2):
    """Refine edge enhancement to avoid artifacts."""
    # Convert image to grayscale for edge detection
    grayscale = image.mean(axis=2).astype(np.uint8)
    edges = filters.sobel(grayscale)  # Edge detection using Sobel
    blurred_edges = gaussian_filter(edges, sigma=blur_radius)  # Smooth edges
    edges_normalized = (blurred_edges / blurred_edges.max() * 255).astype(np.uint8)
    enhanced = np.clip(image + edge_weight * edges_normalized[:, :, None], 0, 255)
    return enhanced.astype(np.uint8)
    

# Process RAW Image
def process_image(image, steps, gamma_value=2.2, denoise_sigma=1.0, sharpen_amount=1.5, edge_weight=1.0, saturation_factor=1.5):
    output = image.copy()
    if "Demosaic" in steps:
        output = demosaic(output)
    if "White Balance" in steps:
        output = white_balance(output)
    if "Denoise" in steps:
        output = denoise(output, sigma=denoise_sigma)
    if "Gamma Correction" in steps:
        output = gamma_correction(output, gamma=gamma_value, saturation_factor=saturation_factor)
    if "Sharpen" in steps:
        output = sharpen(output, amount=sharpen_amount)
    if "Enhance Edges" in steps:
        output = enhance_edges(output, edge_weight=edge_weight)
   

    # Convert the processed image to uint8 for proper visualization in Streamlit
    if output.max() > 255:
        output = np.clip(output / (output.max() / 255), 0, 255).astype(np.uint8)
    return output


# Streamlit UI
def main():
    st.title(" RAW Image Signal Processing Tool ")

    # Upload RAW Image
    uploaded_file = st.file_uploader("Upload a RAW Image (e.g., .RAW or .DNG)", type=["raw", "dng", "png"])
    if uploaded_file is not None:
        # Load and display RAW image
        raw_image = np.frombuffer(uploaded_file.read(), dtype=np.uint16)
        raw_image = raw_image.reshape((1280, 1920))  # Adjust dimensions to match your image format
        st.image(raw_image, caption="Uploaded RAW Image", use_column_width=True, clamp=True, channels="GRAY")

        # Processing Steps
        st.sidebar.title("Processing Steps")
        steps = st.sidebar.multiselect(
            "Select Processing Steps",
            ["Demosaic", "White Balance", "Denoise", "Gamma Correction", "Sharpen", "Enhance Edges" ],
            default=["Demosaic"]
        )

        # Parameters
        st.sidebar.title("Parameters")
        gamma_value = st.sidebar.slider("Gamma Correction Value", 1.0, 3.0, 1.0, step=0.1)
        denoise_sigma = st.sidebar.slider("Denoise Sigma", 0.5, 5.0, 0.9, step=0.1)
        sharpen_amount = st.sidebar.slider("Sharpen Amount", 0.5, 3.0, 3.0, step=0.1)
        edge_weight = st.sidebar.slider("Edge Weight", 0.5, 3.0, 1.0, step=0.1)
        saturation_factor = st.sidebar.slider("Color Saturation Factor", 0.5, 3.0, 3.0, step=0.1)
        

        # Process Image
        processed_image = process_image(
            raw_image, steps, gamma_value=gamma_value,
            denoise_sigma=denoise_sigma,
            sharpen_amount=sharpen_amount,
            edge_weight=edge_weight,
            saturation_factor=saturation_factor,
        )

        # Display Processed Image
        st.image(processed_image, caption="Processed Image", use_column_width=True, clamp=True)

        # Log Observations
        st.text("Observations:")
        st.text(f"Steps Applied: {', '.join(steps)}")


if __name__ == "__main__":
    main()
