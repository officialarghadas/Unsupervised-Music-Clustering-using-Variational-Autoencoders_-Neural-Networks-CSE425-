import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
import os

Z_PATH = "results/latent_vectors.npy"
LANG_PATH = "data/processed/language.npy"
RESULT_DIR = "results"

def evaluate(Z, labels, lang):
    n_clusters = len(set(labels))

    # If only one cluster, skip silhouette and related metrics
    if n_clusters < 2:
        print("Warning: Only one cluster found. Skipping silhouette/CH/DB scores.")
        return {
            "Silhouette": np.nan,
            "Calinski_Harabasz": np.nan,
            "Davies_Bouldin": np.nan,
            "ARI_language": adjusted_rand_score(lang, labels)
        }

    return {
        "Silhouette": silhouette_score(Z, labels),
        "Calinski_Harabasz": calinski_harabasz_score(Z, labels),
        "Davies_Bouldin": davies_bouldin_score(Z, labels),
        "ARI_language": adjusted_rand_score(lang, labels)
    }

def main():
    Z = np.load(Z_PATH)
    lang = np.load(LANG_PATH, allow_pickle=True)

    reports = []

    for method in ["kmeans", "agg", "dbscan"]:
        labels = np.load(f"results/labels_{method}.npy")
        metrics = evaluate(Z, labels, lang)
        metrics["Method"] = method
        reports.append(metrics)

    df = pd.DataFrame(reports)
    df.to_csv(os.path.join(RESULT_DIR,"clustering_metrics_beta_vae.csv"), index=False)
    print(df)
    print("\nMetrics saved to results/clustering_metrics_beta_vae.csv")

if __name__ == "__main__":
    main()
