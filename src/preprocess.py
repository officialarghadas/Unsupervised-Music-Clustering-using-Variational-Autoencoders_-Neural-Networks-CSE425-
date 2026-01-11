import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Paths
DATA_PATH = "data/spotify_songs.csv"
OUT_DIR = "data/processed"

# Create output directory if not exists
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing lyrics
    df = df.dropna(subset=["lyrics"])

    print(f"Dataset size after cleaning: {len(df)} songs")

    # ---- AUDIO FEATURES ----
    audio_cols = [
        'danceability','energy','key','loudness','mode',
        'speechiness','acousticness','instrumentalness',
        'liveness','valence','tempo','duration_ms'
    ]

    X_audio = df[audio_cols].values

    # Normalize audio features
    scaler = StandardScaler()
    X_audio = scaler.fit_transform(X_audio)

    # ---- LYRICS FEATURES ----
    print("Vectorizing lyrics using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_lyrics = vectorizer.fit_transform(df["lyrics"]).toarray()

    # ---- LANGUAGE LABELS ----
    lang = df["language"].values

    # ---- SAVE ARRAYS ----
    np.save(os.path.join(OUT_DIR, "X_audio.npy"), X_audio)
    np.save(os.path.join(OUT_DIR, "X_lyrics.npy"), X_lyrics)
    np.save(os.path.join(OUT_DIR, "language.npy"), lang)

    print("Preprocessing complete!")
    print("Saved files:")
    print(" - data/processed/X_audio.npy")
    print(" - data/processed/X_lyrics.npy")
    print(" - data/processed/language.npy")

if __name__ == "__main__":
    main()
