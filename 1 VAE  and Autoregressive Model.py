import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data loading helper function (provided for you)
def load_mnist_from_csv(csv_path='mnist_train.csv'):
    """Load MNIST data from CSV file - this handles the data loading for you"""
    print("Loading MNIST data from CSV...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully. Shape: {df.shape}")
        
        # Handle different possible CSV structures
        if df.shape[1] == 785:  # 784 pixels + 1 label (common format)
            labels = df.iloc[:, 0].values
            pixels = df.iloc[:, 1:].values
        elif 'label' in df.columns:
            labels = df['label'].values
            pixel_cols = [col for col in df.columns if col != 'label']
            pixels = df[pixel_cols].values
        else:
            labels = df.iloc[:, 0].values
            pixels = df.iloc[:, 1:].values
        
        # Ensure we have 784 pixels per image
        if pixels.shape[1] != 784:
            if pixels.shape[1] > 784:
                pixels = pixels[:, :784]
            else:
                padding = np.zeros((pixels.shape[0], 784 - pixels.shape[1]))
                pixels = np.concatenate([pixels, padding], axis=1)
        
        # Normalize and convert to tensors
        pixels = pixels.astype(np.float32)
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        
        pixel_tensor = torch.from_numpy(pixels)
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        
        print(f"Successfully processed {len(pixel_tensor)} samples")
        return pixel_tensor, label_tensor
        
    except Exception as e:
        print(f"CSV loading failed: {e}")
        print("Creating synthetic MNIST-like data as fallback...")
        n_samples = 5000
        pixels = torch.rand(n_samples, 784) * 0.5 + 0.25
        labels = torch.randint(0, 10, (n_samples,))
        return pixels, labels

# Load MNIST data from CSV (this replaces the usual torchvision download)
train_data, train_labels = load_mnist_from_csv('mnist_train.csv')
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print(f"✅ Dataset ready: {len(train_dataset)} samples loaded")


# ## Task 1: Build and Explore a VAE [30 minutes]
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20)  # output: mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return h1[:, :10], h1[:, 10:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Note: train_dataset and train_loader are already loaded from CSV in the imports section

# VAE Training
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

print("Starting VAE training...")
for epoch in range(10):
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
    
    if epoch % 2 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print("VAE training completed!")

# Generate samples
with torch.no_grad():
    sample = torch.randn(64, 10)
    sample = model.decode(sample).cpu()
    
# Visualization
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample[i].view(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle('VAE Generated Samples')
plt.show()

# Compare with original data
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Show original images
for i in range(5):
    axes[0, i].imshow(train_data[i].view(28, 28), cmap='gray')
    axes[0, i].set_title(f'Original (Label: {train_labels[i]})')
    axes[0, i].axis('off')

# Show reconstructions
model.eval()
with torch.no_grad():
    test_input = train_data[:5]
    recon, _, _ = model(test_input)
    for i in range(5):
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].set_title('VAE Reconstruction')
        axes[1, i].axis('off')

plt.tight_layout()
plt.show()


# ## Task 2: Implement an Autoregressive Model [30 minutes]
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out)

import string

# Create character dataset
chars = string.ascii_lowercase + ' '
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Simple text data
text = "hello world this is a simple text for training"
data = [char_to_idx[c] for c in text if c in char_to_idx]

# Model training
model = SimpleRNN(len(chars), 128, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
seq_length = 10
print("Training autoregressive model...")
for epoch in range(100):
    for i in range(len(data) - seq_length):
        inputs = torch.tensor(data[i:i+seq_length]).unsqueeze(0)
        targets = torch.tensor(data[i+1:i+seq_length+1])
        
        inputs_onehot = torch.zeros(1, seq_length, len(chars))
        inputs_onehot.scatter_(2, inputs.unsqueeze(2), 1)
        
        outputs = model(inputs_onehot)
        loss = criterion(outputs.squeeze(0), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print("Autoregressive training completed!")

# Generation code
def generate_text(model, start_char, length=50):
    model.eval()
    result = start_char
    input_char = char_to_idx[start_char]
    
    for _ in range(length):
        input_tensor = torch.zeros(1, 1, len(chars))
        input_tensor[0, 0, input_char] = 1
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0, 0], dim=0)
            input_char = torch.multinomial(probabilities, 1).item()
            result += idx_to_char[input_char]
    
    return result

# Generate some text
generated = generate_text(model, 'h', 30)
print(f"Generated text: '{generated}'")


# ## Task 3: Compare and Reflect [30 minutes]
