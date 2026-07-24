"""
Model Training & Evaluation Module
=====================================
Trains multiple regression models, compares performance, tunes
hyperparameters, and saves the best model.

Models:
  1. Linear Regression       — Baseline
  2. Decision Tree Regressor — Captures non-linear patterns
  3. Random Forest Regressor — Ensemble, reduces overfitting
  4. Gradient Boosting       — Often best overall performance

Evaluation Metrics:
  - MAE  (Mean Absolute Error)
  - MSE  (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R²   (Coefficient of Determination)

Usage:
    python src/model_training.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import load_clean_data, save_plot, print_section, MODEL_DIR
from feature_engineering import engineer_features

# ── Plot Style ────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)


# ──────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on the test set and return metrics.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: True test labels
        model_name: Name string for display
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R² Score': round(r2, 4),
    }


# ──────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────────────────────

def plot_model_comparison(results_df):
    """Bar chart comparing all models across 4 metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['MAE', 'MSE', 'RMSE', 'R² Score']
    colors = sns.color_palette('husl', len(results_df))

    for ax, metric in zip(axes.flat, metrics):
        bars = ax.bar(results_df['Model'], results_df[metric],
                      color=colors, edgecolor='white', alpha=0.85)

        # Add value labels on top of bars
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Model Performance Comparison', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot(fig, '11_model_comparison.png')


def plot_predictions_vs_actual(y_test, y_pred, model_name):
    """Scatter plot of predicted vs actual ratings."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color='#2196F3',
               label='Predictions')

    # Draw the perfect prediction diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')

    ax.set_xlabel('Actual Rating', fontsize=12)
    ax.set_ylabel('Predicted Rating', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs Actual Ratings',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    save_plot(fig, '12_predictions_vs_actual.png')


def plot_feature_importance(model, feature_names, model_name):
    """Horizontal bar chart of top 15 most important features."""
    fig, ax = plt.subplots(figsize=(10, 8))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    colors = sns.color_palette('viridis', len(top_features))
    ax.barh(top_features[::-1], top_importances[::-1],
            color=colors, edgecolor='white', alpha=0.85)

    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'{model_name}: Top 15 Feature Importances',
                 fontsize=14, fontweight='bold')

    save_plot(fig, '13_feature_importance.png')


def plot_residuals(y_test, y_pred, model_name):
    """Residual distribution and residuals vs predicted values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    residuals = y_test - y_pred

    # Left: Residual histogram
    axes[0].hist(residuals, bins=30, color='#2196F3', edgecolor='white',
                 alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Residual (Actual − Predicted)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Residual Distribution', fontsize=13, fontweight='bold')

    # Right: Residuals vs Predicted scatter
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, color='#4CAF50')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Predicted Rating', fontsize=11)
    axes[1].set_ylabel('Residual', fontsize=11)
    axes[1].set_title('Residuals vs Predicted Values',
                      fontsize=13, fontweight='bold')

    fig.suptitle(f'{model_name} — Residual Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, '14_residual_analysis.png')


# ──────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(X=None, y=None, feature_names=None):
    """
    Main training function.
    
    Steps:
        1. Train-test split (80/20)
        2. Feature scaling (StandardScaler)
        3. Train 4 regression models
        4. Compare metrics
        5. Cross-validate the best model
        6. Hyperparameter tuning (GridSearchCV)
        7. Generate evaluation plots
        8. Save the best model
    
    Args:
        X: Feature matrix. If None, will be computed from clean data.
        y: Target array. If None, will be computed from clean data.
        feature_names: List of feature names for plots.
    
    Returns:
        best_model: The best trained model
        results_df: DataFrame with all model metrics
    """
    # Load features if not provided
    if X is None or y is None:
        df = load_clean_data()
        X, y, feature_names = engineer_features(df)

    print_section("STEP 4: MODEL TRAINING & EVALUATION")

    # ── 1. Train-Test Split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set:  {X_test.shape[0]} samples\n")

    # ── 2. Feature Scaling ─────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler\n")

    # ── 3. Define Models ───────────────────────────────────────
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            random_state=42, max_depth=10
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=15, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, random_state=42, max_depth=5, learning_rate=0.1
        ),
    }

    # ── 4. Train and Evaluate Each Model ───────────────────────
    results = []
    trained_models = {}

    print(f"🏋️  Training models...\n")
    print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R² Score':<10}")
    print(f"{'─' * 55}")

    for name, model in models.items():
        # Linear Regression benefits from scaled features;
        # tree-based models are scale-invariant
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)

        results.append(metrics)
        trained_models[name] = model

        print(f"{name:<25} {metrics['MAE']:<10} {metrics['RMSE']:<10} "
              f"{metrics['R² Score']:<10}")

    results_df = pd.DataFrame(results)

    # ── 5. Visualize Model Comparison ──────────────────────────
    print(f"\n🎨 Generating evaluation plots...")
    plot_model_comparison(results_df)

    # ── 6. Identify Best Model ─────────────────────────────────
    best_idx = results_df['R² Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]
    best_r2 = results_df.loc[best_idx, 'R² Score']

    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2})")

    # ── 7. Cross-Validation on Best Model ──────────────────────
    print(f"\n📊 5-Fold Cross-Validation for {best_model_name}...")

    if best_model_name == 'Linear Regression':
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train,
                                    cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(best_model, X_train, y_train,
                                    cv=5, scoring='r2')

    print(f"   CV R² Scores: {cv_scores.round(4)}")
    print(f"   Mean CV R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. Hyperparameter Tuning (tree-based models only) ──────
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        print(f"\n🔧 Hyperparameter Tuning for {best_model_name}...")

        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:  # Gradient Boosting
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
            }
            base_model = GradientBoostingRegressor(random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV R²:      {grid_search.best_score_:.4f}")

        # Update best model with the tuned version
        best_model = grid_search.best_estimator_
        tuned_metrics = evaluate_model(
            best_model, X_test, y_test, f"{best_model_name} (Tuned)"
        )
        print(f"   Tuned Test R²:   {tuned_metrics['R² Score']}")
        print(f"   Tuned Test RMSE: {tuned_metrics['RMSE']}")

    # ── 9. Generate Detailed Plots for Best Model ──────────────
    if best_model_name == 'Linear Regression':
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)

    plot_predictions_vs_actual(y_test, y_pred, best_model_name)
    plot_residuals(y_test, y_pred, best_model_name)

    # Feature importance plot (only available for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feature_names, best_model_name)

    # ── 10. Save Best Model & Scaler ───────────────────────────
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Model saved:  {model_path}")
    print(f"💾 Scaler saved: {scaler_path}")

    # ── 11. Final Summary ──────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  📋 FINAL RESULTS SUMMARY")
    print(f"{'═' * 60}")
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score:  {best_r2}")
    print(f"   RMSE:      {results_df.loc[best_idx, 'RMSE']}")
    print(f"   MAE:       {results_df.loc[best_idx, 'MAE']}")
    print(f"{'═' * 60}")

    return best_model, results_df


# ──────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_and_evaluate()
