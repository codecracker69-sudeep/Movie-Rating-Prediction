"""
Feature Engineering Module
============================
Creates ML-ready features from the cleaned dataset:
  - One-hot encoded genres (top 10)
  - Target-encoded Director and Actor features (Bayesian smoothed)
  - Director/Actor movie counts (experience proxy)
  - Year-based features (movie age, decade)
  - Log-transformed votes

Usage:
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
from utils import load_clean_data, FEATURE_DATA_PATH, print_section


# ──────────────────────────────────────────────────────────────
# ENCODING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def encode_genres(df, top_n=10):
    """
    One-hot encode the top N genres.
    
    Movies can belong to multiple genres (e.g., "Drama, Thriller"),
    so each movie can have multiple genre columns set to 1.
    
    Args:
        df: DataFrame with a 'Genre' column
        top_n: Number of top genres to encode
    
    Returns:
        df: DataFrame with new binary genre columns added
    """
    # Find the top N most common genres across all movies
    all_genres = df['Genre'].str.split(',').explode().str.strip()
    top_genres = all_genres.value_counts().head(top_n).index.tolist()

    # Create a binary column for each top genre
    for genre in top_genres:
        col_name = f'genre_{genre.lower().replace(" ", "_").replace("-", "_")}'
        df[col_name] = df['Genre'].str.contains(genre, na=False).astype(int)

    print(f"  ✓ Created {top_n} genre features: {top_genres}")
    return df


def target_encode(df, column, target='Rating', smoothing=10):
    """
    Target-encode a categorical column using Bayesian smoothing.
    
    Formula:
        encoded = (count × category_mean + smoothing × global_mean) / (count + smoothing)
    
    This prevents categories with very few samples from getting extreme
    encoded values — they are pulled toward the global mean.
    
    Args:
        df: DataFrame
        column: Categorical column to encode
        target: Target variable name (default: 'Rating')
        smoothing: Smoothing factor; higher = more regularization
    
    Returns:
        df: DataFrame with new encoded column
        encoded_col: Name of the new column
    
    Note:
        In a production setting, target encoding should be fit on the training
        set only to prevent data leakage. For this educational project, we use
        the full dataset for simplicity and note this caveat.
    """
    global_mean = df[target].mean()

    # Calculate per-category statistics
    stats = df.groupby(column)[target].agg(['mean', 'count'])

    # Apply Bayesian smoothing
    smoothed_mean = (
        (stats['count'] * stats['mean'] + smoothing * global_mean)
        / (stats['count'] + smoothing)
    )

    # Map encoded values back to the DataFrame
    encoded_col = f'{column.lower().replace(" ", "_")}_encoded'
    df[encoded_col] = df[column].map(smoothed_mean).fillna(global_mean)

    return df, encoded_col


# ──────────────────────────────────────────────────────────────
# MAIN FEATURE ENGINEERING FUNCTION
# ──────────────────────────────────────────────────────────────

def engineer_features(df=None):
    """
    Main feature engineering function. Creates all ML features.
    
    Args:
        df: Optional DataFrame. If None, loads from clean_data.csv
    
    Returns:
        X: Feature matrix (numpy array of shape [n_samples, n_features])
        y: Target variable (numpy array of shape [n_samples])
        feature_names: List of feature column names
    """
    if df is None:
        df = load_clean_data()

    print_section("STEP 3: FEATURE ENGINEERING")

    # Work on a copy to avoid modifying the original
    df = df.copy()
    print("📦 Creating features...\n")

    # ── 1. Genre One-Hot Encoding ──────────────────────────────
    df = encode_genres(df, top_n=10)

    # ── 2. Director Target Encoding ────────────────────────────
    df, director_col = target_encode(df, 'Director', smoothing=10)
    print(f"  ✓ Director target-encoded → '{director_col}'")

    # Director movie count (proxy for experience)
    director_counts = df['Director'].value_counts()
    df['director_movie_count'] = df['Director'].map(director_counts)
    print(f"  ✓ Created 'director_movie_count'")

    # ── 3. Actor Target Encoding ───────────────────────────────
    for actor_col in ['Actor 1', 'Actor 2', 'Actor 3']:
        df, encoded_name = target_encode(df, actor_col, smoothing=10)

        # Actor movie count
        actor_counts = df[actor_col].value_counts()
        count_col = f'{actor_col.lower().replace(" ", "_")}_movie_count'
        df[count_col] = df[actor_col].map(actor_counts)

        print(f"  ✓ {actor_col}: target-encoded + movie count")

    # ── 4. Year-Based Features ─────────────────────────────────
    current_year = 2026
    df['movie_age'] = current_year - df['Year']
    df['decade'] = (df['Year'] // 10) * 10
    print(f"  ✓ Created 'movie_age' (current year − release year)")
    print(f"  ✓ Created 'decade' (e.g., 2010, 2000)")

    # ── 5. Votes Transformation ────────────────────────────────
    df['log_votes'] = np.log1p(df['Votes'])
    print(f"  ✓ Created 'log_votes' (log-transformed to reduce skewness)")

    # ── 6. Duration (already clean and numeric) ────────────────
    # Duration is kept as-is — no transformation needed

    # ── 7. Assemble Final Feature Matrix ───────────────────────
    genre_cols = [col for col in df.columns if col.startswith('genre_')]

    feature_columns = (
        # Numerical features
        ['Duration', 'log_votes', 'movie_age', 'Year']
        # Genre features
        + genre_cols
        # Director features
        + ['director_encoded', 'director_movie_count']
        # Actor features
        + ['actor_1_encoded', 'actor_1_movie_count']
        + ['actor_2_encoded', 'actor_2_movie_count']
        + ['actor_3_encoded', 'actor_3_movie_count']
    )

    X = df[feature_columns].values
    y = df['Rating'].values
    feature_names = feature_columns

    # Save feature matrix to CSV for reference
    feature_df = df[feature_columns + ['Rating']]
    feature_df.to_csv(FEATURE_DATA_PATH, index=False)

    print(f"\n{'─' * 60}")
    print(f"✅ FEATURE ENGINEERING COMPLETE!")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features:   {len(feature_names)}")
    print(f"   Target range:         {y.min():.1f} – {y.max():.1f}")
    print(f"   Saved to: {FEATURE_DATA_PATH}")
    print(f"\n   Features created:")
    for i, name in enumerate(feature_names, 1):
        print(f"     {i:2d}. {name}")
    print(f"{'─' * 60}")

    return X, y, feature_names


# ──────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y, features = engineer_features()
