import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

AUDIO_PATH = "data/processed/X_audio.npy"
LYRICS_PATH = "data/processed/X_lyrics.npy"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# ---- Load data ----
X_audio = np.load(AUDIO_PATH)
X_lyrics = np.load(LYRICS_PATH)

# Concatenate features
X = np.concatenate([X_audio, X_lyrics], axis=1)
X_tensor = torch.tensor(X, dtype=torch.float32)

loader = DataLoader(TensorDataset(X_tensor), batch_size=128, shuffle=True)

# ---- VAE ----
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512,256), nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,256), nn.ReLU(),
            nn.Linear(256,512), nn.ReLU(),
            nn.Linear(512,input_dim)
        )

    def encode(self,x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def loss_fn(recon,x,mu,logvar):
    recon_loss = nn.functional.mse_loss(recon,x,reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# ---- Train ----
model = VAE(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Multimodal VAE...")

for epoch in range(20):
    total_loss = 0
    for batch in loader:
        x = batch[0]
        recon, mu, logvar = model(x)
        loss = loss_fn(recon,x,mu,logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/20 Loss: {total_loss/len(loader):.4f}")

# ---- Save Latent ----
with torch.no_grad():
    mu,_ = model.encode(X_tensor)
    Z = mu.numpy()

np.save(os.path.join(RESULT_DIR,"latent_multimodal.npy"), Z)
torch.save(model.state_dict(), os.path.join(RESULT_DIR,"multimodal_vae.pth"))

print("Saved multimodal latent vectors to results/latent_multimodal.npy")
