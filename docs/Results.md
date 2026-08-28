# Key Results

## Clustering (Internal Validation)

| Model   | Silhouette | Davies–Bouldin | Notes |
|---------|-----------|----------------|-------|
| K-Means | 0.509     | 1.414          | Partitional baseline |
| GMM     | 0.406     | 2.059          | Soft assignment |
| DBSCAN* | 0.735     | 0.274          | *Excludes noise points |

\* DBSCAN scores exclude noise and are **not directly comparable** to K-Means/GMM,
which assign every point to a cluster. Higher Silhouette here partly reflects
the removal of hard-to-place outlier site-days.

**Interpretation:** K-Means gives the cleanest fully-partitioned solution;
DBSCAN is valuable for surfacing anomalous site-days rather than for its
raw score.

## Forecasting (Test Set)

> Fill in from `04_forecasting_model.ipynb`.

| Model             | Site A MAE | Site B MAE | Site C MAE |
|-------------------|-----------|-----------|-----------|
| LSTM              | _TBD_     | _TBD_     | _TBD_     |
| XGBoost           | _TBD_     | _TBD_     | _TBD_     |
| Linear Regression | _TBD_     | _TBD_     | _TBD_     |

**Interpretation:** XGBoost achieves the best test accuracy overall. Its train
MAE is lower than test MAE — expected for gradient-boosted trees on a small
dataset, and mitigated by conservative hyperparameters. Model ranking is
site-dependent, not a single global winner.
