import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load differently exposed images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg"]  # Ensure you use 3 images only
img_list = [cv.imread(fn, cv.IMREAD_COLOR) for fn in img_fn]

# Check if images are loaded properly
if any(img is None for img in img_list):
    print("Error: One or more images could not be loaded. Check file paths.")
    exit()

# Exposure times for the images (in seconds)
exposure_times = np.array([15.0, 2.5, 0.25], dtype=np.float32)

# Estimate camera response function (CRF) using Debevec method
cal_debevec = cv.createCalibrateDebevec()
crf_debevec = cal_debevec.process(img_list, times=exposure_times)

# Merge exposures using the Debevec method with CRF
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())

# Estimate camera response function (CRF) using Robertson method
cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)

# Merge exposures using the Robertson method with CRF
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())

# Tonemap HDR images to 8-bit LDR for display (Debevec)
tonemap_debevec = cv.createTonemap(gamma=2.2)
ldr_debevec = tonemap_debevec.process(hdr_debevec.copy())
ldr_debevec_8bit = np.clip(ldr_debevec * 255, 0, 255).astype('uint8')

# Tonemap HDR images to 8-bit LDR for display (Robertson)
tonemap_robertson = cv.createTonemap(gamma=2.2)
ldr_robertson = tonemap_robertson.process(hdr_robertson.copy())
ldr_robertson_8bit = np.clip(ldr_robertson * 255, 0, 255).astype('uint8')

# Display Camera Response Functions
def plot_crf(crf, title):
    crf = crf.squeeze()  # Remove any singleton dimensions
    if crf.ndim == 2 and crf.shape[1] == 3:  # Multi-channel CRF
        plt.figure(figsize=(8, 6))
        for i, color in enumerate(['r', 'g', 'b']):
            plt.plot(crf[:, i], color, label=f'{color.upper()} Channel')
    elif crf.ndim == 1 or (crf.ndim == 2 and crf.shape[1] == 1):  # Single-channel CRF
        crf = crf[:, 0] if crf.ndim == 2 else crf
        plt.figure(figsize=(8, 6))
        plt.plot(crf, 'k', label='Single Channel (Grayscale)')
    else:
        print(f"Unexpected CRF shape: {crf.shape}")
        return
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Exposure')
    plt.legend()
    plt.grid()
    plt.show()

# Plot CRFs
plot_crf(crf_debevec, "Debevec CRF")
plot_crf(crf_robertson, "Robertson CRF")

# Save the results
cv.imwrite("ldr_debevec.jpg", ldr_debevec_8bit)
cv.imwrite("ldr_robertson.jpg", ldr_robertson_8bit)

# Display the images using Matplotlib
def display_image(title, img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.figure()
    plt.title(title)
    plt.axis("off")
    plt.imshow(img_rgb)
    plt.show()

# Display the results
display_image("Debevec Tone Mapped Result", ldr_debevec_8bit)
display_image("Robertson Tone Mapped Result", ldr_robertson_8bit)

print("Images saved: 'ldr_debevec.jpg', 'ldr_robertson.jpg'")
