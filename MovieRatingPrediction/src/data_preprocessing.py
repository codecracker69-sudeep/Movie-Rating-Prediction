"""
Data Preprocessing Module
==========================
Cleans the raw IMDb India Movies dataset:
  - Parses Year, Duration, Votes from string formats
  - Handles missing values
  - Removes duplicates and outliers
  - Saves cleaned data to data/clean_data.csv

Usage:
    python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np
import re
from utils import create_directories, load_raw_data, CLEAN_DATA_PATH, print_section


# ──────────────────────────────────────────────────────────────
# COLUMN CLEANING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def clean_year(year_str):
    """
    Convert Year from string format to integer.
    
    Examples:
        '(2020)'    → 2020
        'I (2015)'  → 2015
        '(1995)'    → 1995
    """
    if pd.isna(year_str):
        return np.nan
    match = re.search(r'\d{4}', str(year_str))
    return int(match.group()) if match else np.nan


def clean_duration(duration_str):
    """
    Convert Duration from string format to integer (minutes).
    
    Examples:
        '142 min' → 142
        '90 min'  → 90
    """
    if pd.isna(duration_str):
        return np.nan
    match = re.search(r'\d+', str(duration_str))
    return int(match.group()) if match else np.nan


def clean_votes(vote_str):
    """
    Convert Votes from string with commas to integer.
    
    Examples:
        '1,234'   → 1234
        '178,545' → 178545
    """
    if pd.isna(vote_str):
        return np.nan
    cleaned = str(vote_str).replace(',', '').strip()
    try:
        return int(cleaned)
    except ValueError:
        return np.nan


# ──────────────────────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTION
# ──────────────────────────────────────────────────────────────

def preprocess_data():
    """
    Main preprocessing function that cleans the raw dataset.
    
    Steps:
        1. Load raw CSV with latin-1 encoding
        2. Parse Year, Duration, Votes from string formats
        3. Drop rows with missing target (Rating) or critical features
        4. Fill missing Director/Actor names with 'Unknown'
        5. Remove duplicate movies (same Name + Year)
        6. Remove Duration outliers using the IQR method
        7. Save cleaned data to data/clean_data.csv
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis
    """
    create_directories()
    print_section("STEP 1: DATA PREPROCESSING")

    # ── Load Raw Data ──────────────────────────────────────────
    df = load_raw_data()
    print(f"📊 Raw dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\n❌ Missing values:\n{df.isnull().sum()}")
    print(f"\nMissing value %:\n{(df.isnull().sum() / len(df) * 100).round(2)}")

    # ── Clean Individual Columns ───────────────────────────────
    print("\n🔧 Cleaning columns...")

    df['Year'] = df['Year'].apply(clean_year)
    print("  ✓ Year: converted to numeric (e.g., '(2020)' → 2020)")

    df['Duration'] = df['Duration'].apply(clean_duration)
    print("  ✓ Duration: converted to minutes (e.g., '142 min' → 142)")

    df['Votes'] = df['Votes'].apply(clean_votes)
    print("  ✓ Votes: removed commas (e.g., '1,234' → 1234)")

    # ── Handle Missing Values ──────────────────────────────────
    print("\n🧹 Handling missing values...")

    # Drop rows where the target variable (Rating) is missing
    before = len(df)
    df = df.dropna(subset=['Rating'])
    print(f"  ✓ Dropped {before - len(df)} rows with missing Rating (target)")

    # Drop rows missing critical numerical features
    before = len(df)
    df = df.dropna(subset=['Year', 'Duration', 'Votes'])
    print(f"  ✓ Dropped {before - len(df)} rows with missing Year/Duration/Votes")

    # Drop rows missing Genre (needed for feature engineering)
    before = len(df)
    df = df.dropna(subset=['Genre'])
    print(f"  ✓ Dropped {before - len(df)} rows with missing Genre")

    # Fill missing categorical values with 'Unknown'
    for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        missing_count = df[col].isnull().sum()
        df[col] = df[col].fillna('Unknown')
        if missing_count > 0:
            print(f"  ✓ Filled {missing_count} missing '{col}' values with 'Unknown'")

    # ── Remove Duplicates ──────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['Name', 'Year'], keep='first')
    print(f"\n🔄 Removed {before - len(df)} duplicate entries (same Name + Year)")

    # ── Handle Outliers (IQR Method for Duration) ──────────────
    print("\n📏 Removing outliers (IQR method on Duration)...")

    Q1 = df['Duration'].quantile(0.25)
    Q3 = df['Duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before = len(df)
    df = df[(df['Duration'] >= lower_bound) & (df['Duration'] <= upper_bound)]
    print(f"  ✓ Removed {before - len(df)} Duration outliers "
          f"(kept {lower_bound:.0f}–{upper_bound:.0f} min range)")

    # ── Convert Data Types ─────────────────────────────────────
    df['Year'] = df['Year'].astype(int)
    df['Duration'] = df['Duration'].astype(int)
    df['Votes'] = df['Votes'].astype(int)
    df['Rating'] = df['Rating'].astype(float)

    # ── Reset Index ────────────────────────────────────────────
    df = df.reset_index(drop=True)

    # ── Save Cleaned Data ──────────────────────────────────────
    df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"\n{'─' * 60}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"   Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Saved to: {CLEAN_DATA_PATH}")
    print(f"\n📈 Rating Statistics:")
    print(df['Rating'].describe().to_string())
    print(f"{'─' * 60}")

    return df


# ──────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = preprocess_data()
