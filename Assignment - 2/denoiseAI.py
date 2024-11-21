import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# Transformation for the model input
transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raw_image(raw_path, width, height, channels):
    """
    Load a raw image and preprocess it.
    """
    with open(raw_path, "rb") as f:
        raw_data = f.read()

    # Load raw image as a 2D numpy array
    image_array = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

    # Normalize the 12-bit data to 8-bit range (0-255)
    image_array = (image_array / (2**12 - 1)) * 255.0
    image_array = image_array.astype(np.uint8)

    # Apply debayering if required to convert Bayer pattern to RGB
    if channels == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BAYER_GR2RGB)

    # Debug raw image
    print("Raw image min:", image_array.min(), "max:", image_array.max())
    plt.imshow(image_array)
    plt.title("Loaded Raw Image (Converted to RGB)")
    plt.axis("off")
    plt.show()

    return Image.fromarray(image_array)


def denoise_image(image_path, model_path, image_type='raw', width=1920, height=1280, channels=3):
    """
    Load an image, denoise it using a pre-trained model, and visualize the results.
    """
    # Load the raw or standard image
    if image_type == 'raw':
        image = load_raw_image(image_path, width, height, channels)
    else:
        image = Image.open(image_path).convert("RGB")

    # Transform the image into a tensor
    input_image = transform(image).unsqueeze(0).to(device)
    print("Input tensor shape:", input_image.shape)

    # Debug the input tensor
    input_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    print("Input tensor min:", input_np.min(), "max:", input_np.max())
    plt.imshow(np.clip(input_np, 0, 1))
    plt.title("Model Input Tensor")
    plt.axis("off")
    plt.show()

    # Define the model architecture
    model = DenoiseCNN()  # Replace with your actual model class
    model = model.to(device)

    # Load the weights into the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Denoise the image
    with torch.no_grad():
        denoised_image = model(input_image)

    print("Model output shape:", denoised_image.shape)

    # Convert to numpy array for saving and debugging
    denoised_image = denoised_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denoised_image = np.clip(denoised_image, 0, 1)
    denoised_image_scaled = (denoised_image * 255).astype(np.uint8)

    # Debug denoised image
    print("Denoised image min:", denoised_image_scaled.min(), "max:", denoised_image_scaled.max())
    plt.imshow(denoised_image)
    plt.title("Denoised Image (Post Model)")
    plt.axis("off")
    plt.show()

    # Save the denoised image
    denoised_image_pil = Image.fromarray(denoised_image_scaled, mode='RGB')
    denoised_image_pil.save("denoised_output.jpg")
    print("Denoised image saved as 'denoised_output.jpg'.")

    # Display original and denoised images side-by-side
    plt.figure(figsize=(10, 5))

    # Original noisy image
    noisy_image = input_np
    plt.subplot(1, 2, 1)
    plt.title("Original Image with Noise")
    plt.imshow(np.clip(noisy_image, 0, 1))
    plt.axis("off")

    # Denoised image
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image)
    plt.axis("off")

    plt.show()


# Test case
test_image_path = "/content/eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"  # Replace with your test image path
model_path = "/content/denoise_model.pth"  # Replace with your model path

# Visualize raw image to ensure correctness
raw_image = load_raw_image(test_image_path, width=1920, height=1280, channels=3)
raw_image.show()

# Run the denoising function
denoise_image(test_image_path, model_path, image_type='raw', width=1920, height=1280, channels=3)
