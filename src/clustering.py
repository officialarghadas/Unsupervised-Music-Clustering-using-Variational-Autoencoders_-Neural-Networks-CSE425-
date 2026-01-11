import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import os

Z_PATH = "results/latent_beta_vae.npy"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

def main():
    # Load latent vectors
    Z = np.load(Z_PATH)

    # --- KMeans ---
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(Z)
    np.save(os.path.join(RESULT_DIR, "labels_kmeans.npy"), labels_kmeans)

    # --- Agglomerative Clustering ---
    print("Running Agglomerative Clustering...")
    agg = AgglomerativeClustering(n_clusters=10)
    labels_agg = agg.fit_predict(Z)
    np.save(os.path.join(RESULT_DIR, "labels_agg.npy"), labels_agg)

    # --- DBSCAN ---
    print("Running DBSCAN...")
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    labels_dbscan = dbscan.fit_predict(Z)
    np.save(os.path.join(RESULT_DIR, "labels_dbscan.npy"), labels_dbscan)

    print("Clustering complete! Labels saved in results/")

if __name__ == "__main__":
    main()
