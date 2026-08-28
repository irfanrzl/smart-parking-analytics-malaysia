# Methodology

## Data Pipeline
1. **Exploration** (`01`) — Structure checks, missing/duplicate audit,
   transaction counts by site, date-range validation.
2. **Preprocessing** (`02`) — Cleaning, date parsing, feature engineering.
3. **Clustering** (`03`) — Daily site-behaviour aggregation and clustering.
4. **Forecasting** (`04`) — Sequence and tabular demand models.

## Clustering
Daily transactions are aggregated into per-site behaviour records, then
standardised relative to each site to remove scale differences between
facilities. Three algorithms are compared:

- **K-Means** — partitional baseline.
- **Gaussian Mixture Model (GMM)** — soft, probabilistic assignment.
- **DBSCAN** — density-based, isolates noise/outlier site-days.

Models are assessed with **internal validation only** (Silhouette and
Davies–Bouldin). DBSCAN's scores exclude noise points, so they are not
directly comparable to K-Means or GMM.

## Forecasting
Three models capture different demand mechanisms:

- **LSTM** — learns sequence dependency across consecutive days.
- **XGBoost** — gradient-boosted trees over engineered daily features.
- **Linear Regression** — interpretable baseline.

**Leakage control:** Only *known future covariates* (day of week, public
holiday flag) are used as inputs — these are available before the target day.
Post-hoc information (site closure flags, cluster labels) is excluded, since
it is unknown at prediction time. "Which model is best" is treated as
**site-dependent**, reflecting whether a site's demand is sequence-driven or
better explained by independent daily features.

## Validation Metrics
- **Clustering:** Silhouette (cohesion + separation), Davies–Bouldin
  (average worst-case cluster similarity).
- **Forecasting:** MAE / RMSE on a held-out test period.
