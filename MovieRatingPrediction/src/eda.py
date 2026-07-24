"""
Exploratory Data Analysis (EDA) Module
========================================
Generates 10 comprehensive visualizations to understand the dataset:
  1.  Rating distribution (histogram + KDE)
  2.  Movies released per year
  3.  Average rating trend over years
  4.  Top 15 most common genres
  5.  Rating distribution by genre (box plot)
  6.  Duration analysis (distribution + vs Rating)
  7.  Votes analysis (distribution + vs Rating)
  8.  Top directors (by count + by avg rating)
  9.  Correlation heatmap
  10. Rating vs Votes by genre (bubble chart)

All plots are saved to outputs/plots/.

Usage:
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_clean_data, save_plot, print_section

# ── Set Global Plot Style ─────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)

# Consistent color palette for the project
COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'accent': '#4CAF50',
    'dark': '#37474F',
    'light': '#ECEFF1',
}


# ──────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ──────────────────────────────────────────────────────────────

def plot_rating_distribution(df):
    """Plot 1: Distribution of movie ratings with mean and median lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['Rating'], bins=20, color=COLORS['primary'],
            edgecolor='white', alpha=0.8, density=True, label='Histogram')
    df['Rating'].plot.kde(ax=ax, color=COLORS['secondary'], linewidth=2.5,
                          label='KDE Curve')

    # Add mean and median reference lines
    mean_val = df['Rating'].mean()
    median_val = df['Rating'].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5,
               label=f'Median: {median_val:.2f}')

    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of IMDb Movie Ratings', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    save_plot(fig, '01_rating_distribution.png')


def plot_movies_per_year(df):
    """Plot 2: Number of movies released per year (1990 onwards)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    yearly_counts = df['Year'].value_counts().sort_index()
    recent = yearly_counts[yearly_counts.index >= 1990]

    ax.bar(recent.index, recent.values, color=COLORS['primary'],
           edgecolor='white', alpha=0.85)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    ax.set_title('Number of Movies Released Per Year (1990+)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    save_plot(fig, '02_movies_per_year.png')


def plot_avg_rating_by_year(df):
    """Plot 3: Average rating trend over years with a trend line."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Only include years with at least 10 movies for reliability
    yearly_avg = df.groupby('Year')['Rating'].agg(['mean', 'count'])
    yearly_avg = yearly_avg[yearly_avg['count'] >= 10]

    ax.plot(yearly_avg.index, yearly_avg['mean'], color=COLORS['primary'],
            linewidth=2, marker='o', markersize=4, alpha=0.8, label='Avg Rating')

    # Add linear trend line
    z = np.polyfit(yearly_avg.index, yearly_avg['mean'], 1)
    p = np.poly1d(z)
    ax.plot(yearly_avg.index, p(yearly_avg.index), color=COLORS['secondary'],
            linestyle='--', linewidth=2, alpha=0.8, label='Trend Line')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Rating', fontsize=12)
    ax.set_title('Average Movie Rating by Year (min 10 movies/year)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    save_plot(fig, '03_avg_rating_by_year.png')


def plot_top_genres(df):
    """Plot 4: Top 15 most common genres (horizontal bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Split multi-genre entries (e.g., "Drama, Thriller") and count each
    all_genres = df['Genre'].str.split(',').explode().str.strip()
    genre_counts = all_genres.value_counts().head(15)

    colors = sns.color_palette('husl', len(genre_counts))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                   color=colors, edgecolor='white', alpha=0.85)

    # Add count labels on bars
    for bar, count in zip(bars, genre_counts.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_title('Top 15 Most Common Genres', fontsize=14, fontweight='bold')

    save_plot(fig, '04_top_genres.png')


def plot_rating_by_genre(df):
    """Plot 5: Rating distribution by top 10 genres (box plot)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get top 10 genres
    all_genres = df['Genre'].str.split(',').explode().str.strip()
    top_genres = all_genres.value_counts().head(10).index.tolist()

    # Create a row for each genre a movie belongs to
    genre_ratings = []
    for _, row in df.iterrows():
        genres = [g.strip() for g in str(row['Genre']).split(',')]
        for genre in genres:
            if genre in top_genres:
                genre_ratings.append({'Genre': genre, 'Rating': row['Rating']})

    genre_df = pd.DataFrame(genre_ratings)

    # Order genres by their median rating (descending)
    order = (genre_df.groupby('Genre')['Rating']
             .median().sort_values(ascending=False).index)

    sns.boxplot(data=genre_df, x='Genre', y='Rating', order=order,
                ax=ax, palette='husl', fliersize=3)

    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Rating Distribution by Top 10 Genres',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    save_plot(fig, '05_rating_by_genre.png')


def plot_duration_analysis(df):
    """Plot 6: Duration distribution + Duration vs Rating scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Duration distribution
    axes[0].hist(df['Duration'], bins=30, color=COLORS['primary'],
                 edgecolor='white', alpha=0.8)
    mean_dur = df['Duration'].mean()
    axes[0].axvline(mean_dur, color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_dur:.0f} min')
    axes[0].set_xlabel('Duration (minutes)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Movie Duration',
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Right: Duration vs Rating scatter with trend line
    axes[1].scatter(df['Duration'], df['Rating'], alpha=0.3, s=10,
                    color=COLORS['primary'])
    z = np.polyfit(df['Duration'], df['Rating'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Duration'].min(), df['Duration'].max(), 100)
    axes[1].plot(x_line, p(x_line), color='red', linewidth=2,
                 linestyle='--', label='Trend Line')
    axes[1].set_xlabel('Duration (minutes)', fontsize=11)
    axes[1].set_ylabel('Rating', fontsize=11)
    axes[1].set_title('Duration vs Rating', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_plot(fig, '06_duration_analysis.png')


def plot_votes_analysis(df):
    """Plot 7: Votes distribution (log scale) + Votes vs Rating scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    log_votes = np.log1p(df['Votes'])

    # Left: Log-transformed Votes distribution
    axes[0].hist(log_votes, bins=30, color=COLORS['accent'],
                 edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Log(Votes + 1)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Votes (Log Scale)',
                      fontsize=13, fontweight='bold')

    # Right: Log-Votes vs Rating scatter with trend line
    axes[1].scatter(log_votes, df['Rating'], alpha=0.3, s=10,
                    color=COLORS['accent'])
    z = np.polyfit(log_votes, df['Rating'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(log_votes.min(), log_votes.max(), 100)
    axes[1].plot(x_line, p(x_line), color='red', linewidth=2,
                 linestyle='--', label='Trend Line')
    axes[1].set_xlabel('Log(Votes + 1)', fontsize=11)
    axes[1].set_ylabel('Rating', fontsize=11)
    axes[1].set_title('Votes vs Rating (Log Scale)',
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_plot(fig, '07_votes_analysis.png')


def plot_top_directors(df):
    """Plot 8: Top 15 directors by movie count + by avg rating."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Calculate director statistics
    director_stats = df.groupby('Director').agg(
        movie_count=('Rating', 'count'),
        avg_rating=('Rating', 'mean')
    ).sort_values('movie_count', ascending=False)

    # Exclude 'Unknown' directors
    director_stats = director_stats[director_stats.index != 'Unknown']

    # Left: Top 15 directors by movie count
    top_by_count = director_stats.head(15)
    axes[0].barh(top_by_count.index[::-1], top_by_count['movie_count'].values[::-1],
                 color=COLORS['primary'], edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Number of Movies', fontsize=11)
    axes[0].set_title('Top 15 Directors (by Movie Count)',
                      fontsize=13, fontweight='bold')

    # Right: Top 15 directors by avg rating (minimum 5 movies)
    experienced = director_stats[director_stats['movie_count'] >= 5]
    top_by_rating = experienced.sort_values('avg_rating', ascending=False).head(15)
    axes[1].barh(top_by_rating.index[::-1], top_by_rating['avg_rating'].values[::-1],
                 color=COLORS['secondary'], edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('Average Rating', fontsize=11)
    axes[1].set_title('Top 15 Directors by Avg Rating\n(min 5 movies)',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 10)

    plt.tight_layout()
    save_plot(fig, '08_top_directors.png')


def plot_correlation_heatmap(df):
    """Plot 9: Correlation heatmap of numerical features."""
    fig, ax = plt.subplots(figsize=(8, 6))

    numerical_cols = ['Rating', 'Year', 'Duration', 'Votes']
    corr_matrix = df[numerical_cols].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, ax=ax,
                annot_kws={'fontsize': 12})

    ax.set_title('Correlation Heatmap of Numerical Features',
                 fontsize=14, fontweight='bold')

    save_plot(fig, '09_correlation_heatmap.png')


def plot_rating_vs_votes_by_genre(df):
    """Plot 10: Average Rating vs Average Votes by genre (bubble chart)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 12 genres
    all_genres = df['Genre'].str.split(',').explode().str.strip()
    genre_list = all_genres.value_counts().head(12).index.tolist()

    # Calculate stats per genre
    genre_stats = []
    for genre in genre_list:
        mask = df['Genre'].str.contains(genre, na=False)
        subset = df[mask]
        genre_stats.append({
            'Genre': genre,
            'Avg Rating': subset['Rating'].mean(),
            'Avg Votes': subset['Votes'].mean(),
            'Count': len(subset),
        })

    stats_df = pd.DataFrame(genre_stats)
    colors = sns.color_palette('husl', len(stats_df))

    # Bubble size proportional to movie count
    ax.scatter(stats_df['Avg Votes'], stats_df['Avg Rating'],
               s=stats_df['Count'] * 2, c=colors, alpha=0.7,
               edgecolors='white', linewidth=1.5)

    # Add genre labels next to each bubble
    for _, row in stats_df.iterrows():
        ax.annotate(row['Genre'],
                    (row['Avg Votes'], row['Avg Rating']),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    fontweight='bold')

    ax.set_xlabel('Average Votes', fontsize=12)
    ax.set_ylabel('Average Rating', fontsize=12)
    ax.set_title('Average Rating vs Votes by Genre\n(bubble size = number of movies)',
                 fontsize=14, fontweight='bold')

    save_plot(fig, '10_rating_vs_votes_by_genre.png')


# ──────────────────────────────────────────────────────────────
# MAIN EDA FUNCTION
# ──────────────────────────────────────────────────────────────

def run_eda(df=None):
    """
    Run the complete Exploratory Data Analysis pipeline.
    Generates 10 visualizations and saves them to outputs/plots/.
    
    Args:
        df: Optional DataFrame. If None, loads from clean_data.csv
    """
    if df is None:
        df = load_clean_data()

    print_section("STEP 2: EXPLORATORY DATA ANALYSIS")

    # Quick dataset overview
    print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(f"📈 Quick Statistics:")
    print(f"   Rating range:     {df['Rating'].min()} – {df['Rating'].max()}")
    print(f"   Year range:       {df['Year'].min()} – {df['Year'].max()}")
    print(f"   Avg Duration:     {df['Duration'].mean():.1f} min")
    print(f"   Avg Votes:        {df['Votes'].mean():,.0f}")
    print(f"   Unique Directors: {df['Director'].nunique()}")
    unique_genres = df['Genre'].str.split(',').explode().str.strip().nunique()
    print(f"   Unique Genres:    {unique_genres}")

    print(f"\n🎨 Generating visualizations...")

    plot_rating_distribution(df)        # Plot 1
    plot_movies_per_year(df)            # Plot 2
    plot_avg_rating_by_year(df)         # Plot 3
    plot_top_genres(df)                 # Plot 4
    plot_rating_by_genre(df)            # Plot 5
    plot_duration_analysis(df)          # Plot 6
    plot_votes_analysis(df)             # Plot 7
    plot_top_directors(df)              # Plot 8
    plot_correlation_heatmap(df)        # Plot 9
    plot_rating_vs_votes_by_genre(df)   # Plot 10

    print(f"\n{'─' * 60}")
    print(f"✅ EDA COMPLETE! Generated 10 visualizations")
    print(f"   Saved to: outputs/plots/")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_eda()
