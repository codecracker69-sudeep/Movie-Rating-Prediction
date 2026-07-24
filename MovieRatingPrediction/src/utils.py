"""
Utility Functions & Constants
==============================
Shared helper functions and path constants used across all modules.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# PATH CONSTANTS (all paths are relative to the project root)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'IMDb Movies India.csv')
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'clean_data.csv')
FEATURE_DATA_PATH = os.path.join(DATA_DIR, 'features.csv')


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def create_directories():
    """Create all required project directories if they don't exist."""
    for directory in [DATA_DIR, PLOT_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)


def load_raw_data():
    """
    Load the raw IMDb India Movies CSV file.
    Uses 'latin-1' encoding to handle special characters in movie names.
    """
    print(f"  Loading data from: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH, encoding='latin-1')


def load_clean_data():
    """Load the cleaned dataset from clean_data.csv."""
    return pd.read_csv(CLEAN_DATA_PATH)


def save_plot(fig, filename):
    """
    Save a matplotlib figure to the outputs/plots/ directory.
    
    Args:
        fig: matplotlib Figure object
        filename: Name of the file (e.g., '01_rating_distribution.png')
    """
    filepath = os.path.join(PLOT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved plot: {filename}")


def print_section(title):
    """Print a formatted section header for console output."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
