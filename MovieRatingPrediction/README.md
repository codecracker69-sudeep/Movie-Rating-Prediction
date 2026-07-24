# 🎬 Movie Rating Prediction

> **Internship Project** — Build a machine learning model that predicts IMDb ratings of Indian movies based on features like genre, director, actors, duration, and votes.

---

## 📋 Project Overview

This project uses the [IMDb India Movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) dataset (~15,000+ Indian movies) to train regression models that predict movie ratings. The pipeline covers:

1. **Data Preprocessing** — Cleaning, handling missing values, removing outliers
2. **Exploratory Data Analysis** — 10 publication-quality visualizations
3. **Feature Engineering** — Genre encoding, target encoding, derived features
4. **Model Training** — 4 regression models with hyperparameter tuning

---

## 📁 Project Structure

```
Movie Rating Prediction/
├── data/
│   └── IMDb Movies India.csv       # Raw dataset (download from Kaggle)
├── src/
│   ├── utils.py                    # Shared constants & helpers
│   ├── data_preprocessing.py       # Data cleaning module
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── feature_engineering.py      # Feature creation & encoding
│   └── model_training.py           # Model training & evaluation
├── outputs/
│   ├── plots/                      # Generated visualizations
│   └── models/                     # Saved trained models
├── main.py                         # Run the full pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- Place `IMDb Movies India.csv` inside the `data/` folder

### 3. Run the Pipeline
```bash
python main.py
```

This runs all 4 steps automatically and saves results to `outputs/`.

### Run Individual Steps
```bash
python src/data_preprocessing.py    # Step 1: Clean data
python src/eda.py                   # Step 2: Generate plots
python src/feature_engineering.py   # Step 3: Create features
python src/model_training.py        # Step 4: Train models
```

---

## 📊 Dataset

| Column   | Description                          | Example          |
|----------|--------------------------------------|------------------|
| Name     | Movie title                          | Dangal           |
| Year     | Release year                         | (2016)           |
| Duration | Movie length                         | 161 min          |
| Genre    | Genre(s)                             | Action, Biography|
| Rating   | IMDb rating (target variable, 1–10)  | 8.4              |
| Votes    | Number of user votes                 | 178,545          |
| Director | Director name                        | Nitesh Tiwari    |
| Actor 1  | Lead actor                           | Aamir Khan       |
| Actor 2  | Supporting actor                     | Sakshi Tanwar    |
| Actor 3  | Supporting actor                     | Fatima Hussain   |

---

## 🤖 Models Used

| Model                      | Description                                    |
|----------------------------|------------------------------------------------|
| Linear Regression          | Baseline linear model                          |
| Decision Tree Regressor    | Captures non-linear relationships              |
| Random Forest Regressor    | Ensemble of decision trees, reduces overfitting |
| Gradient Boosting Regressor| Sequential ensemble, often best performance     |

### Evaluation Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Proportion of variance explained

---

## 🛠 Technologies

- **Python 3.8+**
- **pandas** & **numpy** — Data manipulation
- **matplotlib** & **seaborn** — Visualization
- **scikit-learn** — Machine learning models & evaluation
- **joblib** — Model serialization

---

## 📈 Key Features Engineered

- **Genre One-Hot Encoding** — Top 10 genres as binary features
- **Director Target Encoding** — Historical average rating per director (Bayesian smoothed)
- **Actor Target Encoding** — Historical average rating per actor
- **Director/Actor Movie Count** — Experience proxy
- **Movie Age** — Years since release
- **Log Votes** — Log-transformed vote count to reduce skewness

---

## 📝 License

This project is created for educational/internship purposes.
