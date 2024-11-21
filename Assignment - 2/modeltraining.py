import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple Denoising CNN
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure a fixed size for all images
    transforms.ToTensor(),
])

# Load the DIV2K dataset
train_dir = '/content/drive/MyDrive/DIV2K_train_HR'  # Replace with the path to your DIV2K dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = DenoiseCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        clean_images = batch[0].to(device)

        # Simulate noisy images
        noise = torch.randn_like(clean_images) * 0.1
        noisy_images = clean_images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # Forward pass
        denoised_images = model(noisy_images)
        loss = criterion(denoised_images, clean_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "denoise_model.pth")
print("Model saved as 'denoise_model.pth'.")

# Function to denoise and visualize an image
def denoise_image(image_path, model_path):
    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)

    # Denoise the image
    with torch.no_grad():
        denoised_image = model(input_image)

    # Convert the tensor back to an image
    denoised_image = denoised_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denoised_image = np.clip(denoised_image, 0, 1)  # Ensure pixel values are in [0, 1]
    denoised_image_pil = Image.fromarray((denoised_image * 255).astype(np.uint8))

    # Save the denoised image
    denoised_image_pil.save("denoised_output.jpg")
    print("Denoised image saved as 'denoised_output.jpg'.")

    # Display the images
    plt.figure(figsize=(10, 5))

    # Original noisy image
    noisy_image = input_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    noisy_image = np.clip(noisy_image, 0, 1)

    plt.subplot(1, 2, 1)
    plt.title("Original Image with Noise")
    plt.imshow(noisy_image)
    plt.axis("off")

    # Denoised image
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image_pil)
    plt.axis("off")

    plt.show()

# Test the model with a sample image
test_image_path = "/content/clean-water-1080x675.jpg"  # Replace with your test image path
denoise_image(test_image_path, "denoise_model.pth")
