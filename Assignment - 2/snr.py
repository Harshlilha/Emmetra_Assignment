import cv2
import numpy as np

def compute_snr(original, filtered):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - filtered) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_snr_for_gray_tones(original, filtered):
    low_tone = (0, 85)
    mid_tone = (85, 170)
    high_tone = (170, 255)
    
    low_region = (original >= low_tone[0]) & (original < low_tone[1])
    mid_region = (original >= mid_tone[0]) & (original < mid_tone[1])
    high_region = (original >= high_tone[0]) & (original < high_tone[1])
    
    snr_low = compute_snr(original[low_region], filtered[low_region])
    snr_mid = compute_snr(original[mid_region], filtered[mid_region])
    snr_high = compute_snr(original[high_region], filtered[high_region])
    
    return snr_low, snr_mid, snr_high, low_region, mid_region, high_region

def get_bounding_rects(region):
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = [cv2.boundingRect(contour) for contour in contours]
    return bounding_rects

def visualize_regions_with_rectangles(original, low_region, mid_region, high_region, snr_low, snr_mid, snr_high):
    # Convert original image to color for visualization
    visualized_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Get bounding rectangles for each region
    low_rects = get_bounding_rects(low_region)
    mid_rects = get_bounding_rects(mid_region)
    high_rects = get_bounding_rects(high_region)
    
    # Define colors
    low_color = (255, 0, 0)  # Blue
    mid_color = (0, 255, 0)  # Green
    high_color = (0, 0, 255)  # Red
    
    # Draw rectangles for low gray tones
    for rect in low_rects:
        x, y, w, h = rect
        cv2.rectangle(visualized_image, (x, y), (x + w, y + h), low_color, 2)
    # Draw rectangles for medium gray tones
    for rect in mid_rects:
        x, y, w, h = rect
        cv2.rectangle(visualized_image, (x, y), (x + w, y + h), mid_color, 2)
    # Draw rectangles for high gray tones
    for rect in high_rects:
        x, y, w, h = rect
        cv2.rectangle(visualized_image, (x, y), (x + w, y + h), high_color, 2)

    # Add SNR annotations with corresponding colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualized_image, f"Low SNR: {snr_low:.2f} dB", (10, 30), font, 1, low_color, 2, cv2.LINE_AA)
    cv2.putText(visualized_image, f"Mid SNR: {snr_mid:.2f} dB", (10, 70), font, 1, mid_color, 2, cv2.LINE_AA)
    cv2.putText(visualized_image, f"High SNR: {snr_high:.2f} dB", (10, 110), font, 1, high_color, 2, cv2.LINE_AA)
    
    return visualized_image

def load_raw_image(file_path, width, height, bit_depth):
    with open(file_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)
    raw_image = raw_data.reshape((height, width))
    normalized_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized_img

def main():
    raw_file_path = "C:/Users/ultim/Downloads/eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    jpg_file_path = "median_filtered_image.jpg"
    
    width = 1920
    height = 1280
    bit_depth = 12
    
    original = load_raw_image(raw_file_path, width, height, bit_depth)
    compressed = cv2.imread(jpg_file_path, cv2.IMREAD_GRAYSCALE)
    compressed_resized = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    snr_low, snr_mid, snr_high, low_region, mid_region, high_region = compute_snr_for_gray_tones(original, compressed_resized)
    
    result_image = visualize_regions_with_rectangles(original, low_region, mid_region, high_region, snr_low, snr_mid, snr_high)
    
    # Resize to 1280x720
    resized_result_image = cv2.resize(result_image, (1280, 720))
    
    cv2.imshow("SNR Visualization with Rectangles", resized_result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
