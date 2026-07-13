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
├── explainability/              ← SHAP + Gemini fraud explanation pipeline
│   ├── design.md                ← research question, prompt schema, metrics
│   ├── compute_shap.py           ← offline SHAP values for the RF model
│   ├── prompt_builder.py         ← builds the analyst-facing prompt
│   ├── llm_client.py             ← Gemini client with model fallback chain
│   ├── generate_explanations.py  ← batch-generates explanations (resumable)
│   ├── evaluate.py               ← faithfulness + usefulness scoring
│   ├── human_rating_template.py  ← exports blind human-rating sheet
│   └── export_frontend_data.py   ← exports cases for the frontend below
├── feature_discovery/           ← LLM-assisted feature discovery + leakage audit
│   ├── FINDINGS.md               ← full audit trail: what was found, fixed, still open
│   ├── analyze_false_negatives.py
│   ├── propose_features.py
│   ├── leakage_check.py          ← temporal-leakage audit (added after review)
│   └── test_feature_impact.py
├── frontend/                    ← Next.js triage dashboard (see its README)
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
- **Model Performance:** Random Forest achieves AUC-ROC of 0.923, outperforming Logistic Regression (0.702) by 31.4% relative improvement, detecting 67.5% of fraud cases. Revised from an earlier 0.822 after a feature-discovery audit found the original feature selection had dropped a highly predictive feature and that a related candidate had look-ahead leakage — both fixed; see `feature_discovery/FINDINGS.md` for the full audit trail, including one residual leakage question (`composite_risk`) that couldn't be fully resolved.

---

## Model Performance — Before / After Feature-Discovery Fix

The original model used 7 base features. A closed-loop audit (analyze
missed fraud → LLM proposes candidate features → empirically test AUC →
audit for leakage, see `feature_discovery/FINDINGS.md`) found the pruning
step had discarded a highly predictive feature (`amount_sum_24h`) and that
a related candidate (`amount_vs_mean_ratio`) had look-ahead leakage in its
source column. Both fixed; model retrained on 9 base features (+10 bank
one-hots = 19 total).

| Metric | Before (17 features) | After (19 features) |
|---|---|---|
| AUC-ROC | 0.822 | **0.923** |
| Recall | 64.0% (384/600 fraud caught) | **67.5%** (405/600 fraud caught) |
| Precision | 0.95% | **7.08%** |
| Accuracy | 79.84% | **97.25%** |
| False positives (of 200,000 test txns) | ~40,098 | **~5,315** |

Note on why this worked despite both new features having very low raw
correlation with `is_fraud` (`amount_sum_24h`: −0.0019, `amount_vs_mean_ratio_safe`:
0.031): correlation only measures a single-column, linear relationship. A
Random Forest can combine features non-linearly — e.g. "large amount **and**
high recent 24h activity **and** unusual relative to this customer's own
history" — a conditional signal invisible to any one-column correlation
check. Fraud transactions average a `amount_vs_mean_ratio_safe` of 3.32
(over 3x their own typical amount) vs 1.27 for legitimate transactions —
a real, substantial difference the correlation coefficient alone
understates.

**Open item:** `composite_risk`, unchanged from the raw dataset, was
itself constructed upstream using the original leaky ratio — a small
residual leakage exposure may remain there that this fix could not reach.

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
