
## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1: 
Import necessary libraries including PyTorch, torchvision, and matplotlib.

### STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.

### STEP 3:
Add Gaussian noise to training and testing images using a custom function.

## STEP 4:
Initialize model, define loss function (MSE) and optimizer (Adam).

## STEP 5:
Train the model using noisy images as input and original images as target.

## STEP 6:
Visualize and compare original, noisy, and denoised images.
## PROGRAM
### Name:PRAKASH R
### Register Number: 212222240074
```
# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> [32, 7, 7]
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# Instantiate the model
model = Autoencoder().to(device)

# Initialize model, loss function and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, _ in loader:
            noisy_imgs = imgs + 0.5 * torch.randn_like(imgs)  # Add noise
            noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
            imgs, noisy_imgs = imgs.to(device), noisy_imgs.to(device)

            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```

## OUTPUT

### Model Summary
![Screenshot 2025-05-19 081923](https://github.com/user-attachments/assets/2225d1e5-e044-46e9-85b2-3c46f8aee0de)



### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/c25b0b69-782d-418f-a600-217f317bed6f)


## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
