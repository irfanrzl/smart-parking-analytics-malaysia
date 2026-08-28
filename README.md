# Analysis of Parking Demand Forecasting and Usage Pattern Clustering for Smart Urban Decision Support in Malaysia

Final Year Project — BSc (Hons) Computer Science (Data Analytics), Asia Pacific University (APU).

## Overview

This project analyses parking transaction records from three anonymised Klang Valley
parking facilities (Site A, B, C) across 2025. It combines temporal pattern analysis,
behavioural clustering, and demand forecasting into an interactive decision-support
dashboard for smart urban planning.

## Objectives

1. **Temporal Pattern Analysis** — Identify when parking demand occurs across daily,
   weekly, and seasonal cycles (EDA-driven).
2. **Behavioural Clustering** — Characterise how each site operates using combined
   temporal and occupancy features (K-Means, DBSCAN).
3. **Demand Forecasting** — Predict future parking demand using LSTM, XGBoost, and
   Linear Regression, and compare model behaviour by site.

## Repository Structure

| File | Description |
|------|-------------|
| `01_data_exploration.ipynb` | Exploratory data analysis and temporal pattern discovery |
| `02_data_preprocessing.ipynb` | Data cleaning, feature engineering, and dataset preparation |
| `03_clustering_analysis.ipynb` | Behavioural clustering with K-Means and DBSCAN |
| `04_forecasting_model.ipynb` | Demand forecasting (LSTM, XGBoost, Linear Regression) |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies |

## Methods

- **Clustering:** K-Means and DBSCAN, evaluated with Silhouette and Davies–Bouldin
  scores (internal validation only).
- **Forecasting:** LSTM (sequence-based), XGBoost (gradient-boosted trees), and
  Linear Regression baseline. Model performance is site-dependent, reflecting
  different demand mechanisms.
- **Features:** Known future covariates (day of week, public holiday flag) are used
  as forecasting inputs; post-hoc information (closure flags, cluster labels) is
  excluded to avoid data leakage.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

To reproduce the analysis, run the notebooks in order (01 → 04).

## Data

The raw parking transaction data is not included in this repository due to
confidentiality. The notebooks expect data files in a `data/` directory. See
the preprocessing notebook for the expected schema.

## Tech Stack

Python · Jupyter · pandas · scikit-learn · XGBoost · TensorFlow/Keras · Streamlit

## Author

**Muhammad Irfan Bin Mohd Rizal** (TP078491)
Supervised by Dr. Murugananthan Velayutham

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
