import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# Paths
DATA_PATH = "data/processed/X_audio.npy"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# ----- VAE Model -----
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU()
        )

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Loss function
def loss_fn(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

def main():
    # Load data
    X = np.load(DATA_PATH)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=True)

    # Initialize model
    model = VAE(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training VAE...")

    # Train
    for epoch in range(30):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/30 - Loss: {total_loss/len(loader):.4f}")

    # Extract latent vectors (mean)
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        Z = mu.numpy()

    # Save outputs
    np.save(os.path.join(RESULT_DIR, "latent_vectors.npy"), Z)
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "vae_model.pth"))

    print("Training complete!")
    print("Saved:")
    print(" - results/latent_vectors.npy")
    print(" - results/vae_model.pth")

if __name__ == "__main__":
    main()
