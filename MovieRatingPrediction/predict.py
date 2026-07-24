"""
Movie Rating Predictor — Simple Version
=========================================
Just enter a movie name and get the predicted rating!

Usage:
    python3 predict.py
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import MODEL_DIR, CLEAN_DATA_PATH, RAW_DATA_PATH


def load_model_and_data():
    """Load the trained model and dataset."""
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')

    if not os.path.exists(model_path):
        print("❌ Model not found! Run 'python3 main.py' first.")
        sys.exit(1)

    model = joblib.load(model_path)
    df = pd.read_csv(CLEAN_DATA_PATH)
    return model, df


def get_target_encoding(df, column, value, smoothing=10):
    """Get target-encoded value for a director/actor."""
    global_mean = df['Rating'].mean()
    stats = df.groupby(column)['Rating'].agg(['mean', 'count'])
    if value in stats.index:
        cat_mean = stats.loc[value, 'mean']
        cat_count = stats.loc[value, 'count']
        return (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing), int(cat_count)
    return global_mean, 1


def predict_for_movie(model, df, movie_row):
    """Predict rating using movie details from the dataset."""
    top_genres = ['Drama', 'Action', 'Romance', 'Comedy', 'Crime',
                  'Thriller', 'Family', 'Musical', 'Mystery', 'Adventure']

    genre_str = str(movie_row['Genre'])
    genre_features = [1 if g in genre_str else 0 for g in top_genres]

    dir_enc, dir_cnt = get_target_encoding(df, 'Director', movie_row['Director'])
    a1_enc, a1_cnt = get_target_encoding(df, 'Actor 1', movie_row['Actor 1'])
    a2_enc, a2_cnt = get_target_encoding(df, 'Actor 2', movie_row['Actor 2'])
    a3_enc, a3_cnt = get_target_encoding(df, 'Actor 3', movie_row['Actor 3'])

    features = np.array([
        movie_row['Duration'], np.log1p(movie_row['Votes']),
        2026 - movie_row['Year'], movie_row['Year'],
        *genre_features,
        dir_enc, dir_cnt, a1_enc, a1_cnt, a2_enc, a2_cnt, a3_enc, a3_cnt,
    ]).reshape(1, -1)

    pred = model.predict(features)[0]
    return round(max(1.0, min(10.0, pred)), 1)


def search_movie(df, name):
    """Search for a movie by name (case-insensitive, partial match)."""
    # Try exact match first
    exact = df[df['Name'].str.lower() == name.lower()]
    if len(exact) > 0:
        return exact

    # Try partial match
    partial = df[df['Name'].str.lower().str.contains(name.lower(), na=False)]
    return partial


def main():
    print("\n" + "🎬 " * 15)
    print("   MOVIE RATING PREDICTOR")
    print("🎬 " * 15)

    model, df = load_model_and_data()
    print(f"\n✅ Model loaded! ({len(df)} movies in database)\n")

    while True:
        movie_name = input("🎬 Enter movie name (or 'quit' to exit): ").strip()

        if movie_name.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break

        if not movie_name:
            continue

        # Search for the movie
        results = search_movie(df, movie_name)

        if len(results) == 0:
            print(f"   ❌ Movie '{movie_name}' not found in the dataset.\n")
            continue

        # If multiple matches, show them
        if len(results) > 1:
            print(f"\n   Found {len(results)} matches:")
            for i, (_, row) in enumerate(results.head(10).iterrows(), 1):
                print(f"   {i}. {row['Name']} ({int(row['Year'])})")

            try:
                choice = input(f"\n   Pick a number (1-{min(len(results), 10)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < min(len(results), 10):
                    movie = results.iloc[idx]
                else:
                    movie = results.iloc[0]
            except (ValueError, IndexError):
                movie = results.iloc[0]
        else:
            movie = results.iloc[0]

        # Predict rating
        predicted = predict_for_movie(model, df, movie)
        actual = movie['Rating']

        # Display result
        print(f"\n   {'═' * 45}")
        print(f"   🎬 {movie['Name']} ({int(movie['Year'])})")
        print(f"   🎭 {movie['Genre']}  |  ⏱ {int(movie['Duration'])} min")
        print(f"   🎬 Director: {movie['Director']}")
        print(f"   ⭐ Cast: {movie['Actor 1']}, {movie['Actor 2']}, {movie['Actor 3']}")
        print(f"   {'─' * 45}")
        print(f"   📊 Actual Rating:    {actual} / 10")
        print(f"   🤖 Predicted Rating: {predicted} / 10")
        print(f"   {'═' * 45}\n")


if __name__ == '__main__':
    main()
