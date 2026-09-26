# 02 ML Engine - Machine Learning, Quantiles & Recommendations

**Folder 2 of AuraCycle AI Architecture**  
Assigned to: **Machine Learning & Recommender Systems Lead**

---

## Overview
This folder contains the core Machine Learning pipelines, non-leaking feature transformations, LightGBM Tri-Quantile Regressors, GroupKFold cross-validation, collaborative and content-based recommendation engines, and Oxford CEBM clinical evidence ranking.

---

## Directory Structure
```text
02_ml_engine/
├── models/
│   ├── lgbm_quantile_10.joblib       # LightGBM regressor for tau = 0.10
│   ├── lgbm_quantile_50.joblib       # LightGBM regressor for tau = 0.50 (Median)
│   ├── lgbm_quantile_90.joblib       # LightGBM regressor for tau = 0.90
│   └── feature_metadata.joblib       # Serialized feature list and mappings
├── collaborative_filter.py           # User-User nearest-neighbor cohort recommender
├── content_recommender.py            # Cosine similarity biomarker-intervention engine
├── evidence_ranker.py                # Oxford CEBM clinical evidence ranking
├── feature_transformer.py            # 43+ biological non-leaking feature engineering
├── train_quantile_lgbm.py            # GroupKFold training for tau = 0.10, 0.50, 0.90
└── evaluate_metrics.py               # MAE, RMSE, PICP, Pinball Loss evaluation
```

---

## Key Algorithms

### 1. Tri-Quantile LightGBM Regression
Trained using pinball loss function:
$$\mathcal{L}_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$$
Outputs the 80% Prediction Interval: $[q_{10}, q_{90}]$ with median estimate $q_{50}$.

### 2. Cohort Collaborative Filtering (`collaborative_filter.py`)
Computes cosine similarity across patient telemetry vectors in the 60,000-user cohort:
$$\text{Sim}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
Ranks symptom remedies by consensus satisfaction from nearest "cycle twins".

### 3. Content-Based Recommendation (`content_recommender.py`)
Vectorizes patient biometrics, diagnoses (PCOS/Endometriosis), and NLP symptoms, matching them against evidence-backed clinical protocols via TF-IDF cosine similarity.

### 4. Oxford CEBM Evidence Ranker (`evidence_ranker.py`)
Classifies and ranks clinical disruption factors according to the Oxford Centre for Evidence-Based Medicine (Levels 1 to 5).

---

## Standalone Execution

```bash
# Retrain quantile models using GroupKFold
python 02_ml_engine/train_quantile_lgbm.py

# Evaluate model metrics (MAE, Pinball loss, Coverage)
python 02_ml_engine/evaluate_metrics.py
```
