
# 🎵 Unsupervised Music Clustering using Variational Autoencoders

This project performs unsupervised clustering of music tracks using learned latent representations from Variational Autoencoders (VAE).  
We implement three experiments:

- Baseline Audio VAE  
- Multimodal VAE (Audio + Lyrics)  
- Beta-VAE (Disentangled Latent Space)

Clustering is evaluated using KMeans, Agglomerative Clustering, and DBSCAN with standard clustering metrics.

---

## 📁 Project Structure

```
Project/
│
├── data/
│   ├── spotify_songs.csv
│   └── processed/
│       ├── X_audio.npy
│       ├── X_lyrics.npy
│       └── language.npy
│
├── src/
│   ├── preprocess.py
│   ├── vae.py
│   ├── multimodal_vae.py
│   ├── beta_vae.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualize_latent.py
│
├── results/
│   ├── latent_vectors.npy
│   ├── latent_multimodal.npy
│   ├── latent_beta_vae.npy
│   ├── clustering_metrics.csv
│   ├── clustering_metrics_multimodal.csv
│   ├── clustering_metrics_beta_vae.csv
│   ├── umap_clusters.png
│   └── umap_languages.png
│
├── notebooks/
│   └── experiments.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone https://github.com/your-username/music-vae-clustering.git
cd music-vae-clustering
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

Download the Spotify dataset from Kaggle and place it in:

```
data/spotify_songs.csv
```

---

## 🚀 Running the Full Pipeline

Run each step once. All intermediate and final outputs are saved in the `results/` folder.

### Step 1 — Preprocess data

```bash
python src/preprocess.py
```

### Step 2 — Train Baseline Audio VAE

```bash
python src/vae.py
```

### Step 3 — Train Multimodal VAE

```bash
python src/multimodal_vae.py
```

### Step 4 — Train Beta-VAE

```bash
python src/beta_vae.py
```

### Step 5 — Run clustering

Edit `src/clustering.py`:

```python
Z_PATH = "results/latent_vectors.npy"         # Baseline
# or
Z_PATH = "results/latent_multimodal.npy"     # Multimodal
# or
Z_PATH = "results/latent_beta_vae.npy"       # Beta-VAE
```

Then run:

```bash
python src/clustering.py
```

### Step 6 — Evaluate clustering

Edit `src/evaluation.py`:

```python
df.to_csv("results/clustering_metrics.csv")                  # Baseline
df.to_csv("results/clustering_metrics_multimodal.csv")       # Multimodal
df.to_csv("results/clustering_metrics_beta_vae.csv")         # Beta-VAE
```

Then run:

```bash
python src/evaluation.py
```

### Step 7 — Visualize latent space (optional)

```bash
python src/visualize_latent.py
```

---

## 📊 Results

Final clustering results are stored in the `results/` directory.

---

## 💻 Requirements

- Python 3.9+
- PyTorch
- Scikit-learn
- UMAP-learn
- Pandas
- NumPy
- Matplotlib

---

## 👤 Author

Argha Das  
BRAC University  
Email: email.arghadas@gmail.com
Linkedin: https://www.linkedin.com/in/argha-das-08899223b/

---

## 📌 Reproducibility Note

Run scripts in the order listed above.  
All outputs are saved to `results/` for full reproducibility.
