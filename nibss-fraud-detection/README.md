# NIBSS Fraud Detection — AI-Powered Financial Crime Analysis
**Author:** Afolabi Olawale Goodluck | **Institution:** QuCoon AI | **Lagos, Nigeria · 2026**

---

## Research Paper

📄 **[Fraud Detection in Nigerian Banking: Temporal, Behavioral, and Geographic Patterns in One Million Transactions](https://drive.google.com/file/d/1mN2d2bi-nVqAhlVMA0JXdGBZg8xO6OlR/view?usp=sharing)**

> **Note:** the paper reports the corrected Random Forest result (AUC-ROC 0.923) from a later feature-discovery audit that isn't merged into this branch yet — see `main`'s numbers below vs. the `feature-discovery` branch, which has the retrained model, the audit trail, and the code behind the paper's Section 3.6.

---

## Overview

This project applies machine learning to detect fraudulent transactions in the Nigerian banking ecosystem using the NIBSS Fraud Dataset — a synthetic, research-grade dataset modelling one million real-world banking transactions across 10 commercial banks and 6 transaction channels. The analysis reveals adversarial adaptation patterns unique to Nigeria's digital payment infrastructure, including a distinct "midnight window" vulnerability, mobile automation signatures, and significant geographic concentration of fraud risk.

---

## Dataset Citation

> Owolabi, S. (2026). *NIBSS Fraud Dataset*. Kaggle. Licensed under CC BY-SA 4.0.  
> Retrieved from: https://www.kaggle.com/

The dataset is a synthetic simulation of NIBSS (Nigeria Inter-Bank Settlement System) transactional data, comprising 1,000,000 records across 38 engineered features. Fraud cases represent 0.300% of all transactions (3,000 cases).

---

## Folder Structure

```
nibss-fraud-detection/
├── data/
│   ├── raw/                    ← original CSV and pickle datasets
│   └── processed/              ← cleaned and feature-engineered output
├── notebooks/
│   ├── 01_eda_analysis.ipynb   ← Exploratory Data Analysis (EDA)
│   └── 02_ml_modeling.ipynb    ← Machine Learning Models (LR + RF)
├── models/
│   └── .gitkeep                ← saved model artifacts go here
├── outputs/
│   └── .gitkeep                ← plots, reports, exported visualisations
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place the dataset** — copy `nibss_fraud_dataset.csv` and `nibss_fraud_dataset.pkl` into `data/raw/`

3. **Run EDA notebook** — open and run `notebooks/01_eda_analysis.ipynb` top-to-bottom

4. **Run Modeling notebook** — open and run `notebooks/02_ml_modeling.ipynb` top-to-bottom

---

## Key Findings

- **Midnight Window:** Fraud peaks at 1:00 AM (0.363% rate vs 0.256% business hours), with night hours accounting for only 14.05% of transaction volume but disproportionately higher fraud risk.
- **ATM Anomaly:** Fraudulent ATM transactions average ₦238,959 vs ₦34,143 for legitimate ones — a 600% amount differential signalling automated large-value withdrawal attacks.
- **Geographic Concentration:** Abuja is the highest-risk state (0.328% fraud rate) with Sterling Bank showing a 3.6-fold risk variation across locations (Abuja: 0.470%, Rivers: 0.130%).
- **Age Vulnerability:** Customers aged 40+ account for 40.8% of all fraud cases despite proportional population representation, suggesting targeted social engineering.
- **Model Performance:** Random Forest achieves AUC-ROC of 0.822, outperforming Logistic Regression (0.702) by 17.1% relative improvement, detecting 64% of fraud cases with zero data leakage features.

---

## Requirements

See `requirements.txt` for full dependency list. Key libraries:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — machine learning models
- `matplotlib`, `seaborn` — visualisations
- `joblib` — model persistence
- `xgboost`, `lightgbm` — available for future experimentation

---

*Afolabi Olawale Goodluck · QuCoon AI · Lagos, Nigeria · 2026*
