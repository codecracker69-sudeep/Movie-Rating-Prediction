"""
Movie Rating Prediction — Main Pipeline
=========================================
Run this script to execute the complete ML pipeline:
  1. Data Preprocessing  — Clean the raw dataset
  2. Exploratory Data Analysis — Generate visualizations
  3. Feature Engineering — Create ML-ready features
  4. Model Training — Train, evaluate, and save the best model

Usage:
    python main.py

Prerequisites:
    - Place 'IMDb Movies India.csv' in the data/ folder
    - Install dependencies: pip install -r requirements.txt
"""

import sys
import os

# Add src/ directory to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_preprocessing import preprocess_data
from eda import run_eda
from feature_engineering import engineer_features
from model_training import train_and_evaluate


def main():
    """Run the complete Movie Rating Prediction pipeline."""

    print("\n" + "🎬 " * 20)
    print("     MOVIE RATING PREDICTION — ML PIPELINE")
    print("🎬 " * 20 + "\n")

    # ── Step 1: Data Preprocessing ──
    df_clean = preprocess_data()

    # ── Step 2: Exploratory Data Analysis ──
    run_eda(df_clean)

    # ── Step 3: Feature Engineering ──
    X, y, feature_names = engineer_features(df_clean)

    # ── Step 4: Model Training & Evaluation ──
    best_model, results_df = train_and_evaluate(X, y, feature_names)

    # ── Done! ──
    print("\n" + "🎬 " * 20)
    print("     ✅ PIPELINE COMPLETE!")
    print("     📂 Check outputs/plots/  for visualizations")
    print("     📂 Check outputs/models/ for saved model")
    print("🎬 " * 20 + "\n")


if __name__ == '__main__':
    main()
