import numpy as np
import umap
import matplotlib.pyplot as plt
import os

RESULT_DIR = "results"
DATA_DIR = "data/processed"

# Load latent vectors and labels
Z = np.load(os.path.join(RESULT_DIR, "latent_vectors.npy"))
labels = np.load(os.path.join(RESULT_DIR, "labels_kmeans.npy"))
lang = np.load(os.path.join(DATA_DIR, "language.npy"), allow_pickle=True)

# ---- Run UMAP ----
reducer = umap.UMAP(random_state=42)
Z_2d = reducer.fit_transform(Z)

# ---- Plot by Cluster ----
plt.figure(figsize=(8,6))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap="tab10", s=6)
plt.colorbar(label="Cluster ID")
plt.title("UMAP of VAE Latent Space (KMeans Clusters)")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "umap_clusters.png"), dpi=300)
plt.show()

# ---- Plot by Language ----
unique_langs = list(set(lang))
lang_to_id = {l:i for i,l in enumerate(unique_langs)}
lang_ids = np.array([lang_to_id[l] for l in lang])

plt.figure(figsize=(8,6))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=lang_ids, cmap="tab20", s=6)
plt.colorbar(label="Language ID")
plt.title("UMAP of VAE Latent Space (Languages)")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "umap_languages.png"), dpi=300)
plt.show()

print("Saved visualizations to:")
print(" - results/umap_clusters.png")
print(" - results/umap_languages.png")
