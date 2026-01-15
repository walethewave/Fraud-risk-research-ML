# NIBSS Fraud Detection Research Project

## Overview
This research project analyzes a comprehensive fraud detection dataset from the Nigeria Inter-Bank Settlement System (NIBSS). The dataset contains **1,000,000 transactions** with **38 features** designed to identify fraudulent financial activities across various banking channels.

## Dataset Information
- **File**: `nibss_fraud_dataset.csv`
- **Total Records**: 1,000,000 transactions
- **Fraudulent Transactions**: 3,000 (0.3% fraud rate)
- **Total Features**: 38 columns
- **Memory Size**: ~276.6 MB

## Column Descriptions

### Basic Transaction Information

| Column | Type | Description |
|--------|------|-------------|
| **transaction_id** | object | Unique identifier for each transaction |
| **customer_id** | object | Unique identifier for each customer |
| **timestamp** | object | Date and time when the transaction occurred |
| **amount** | float64 | Transaction amount in Nigerian Naira (₦) |

### Transaction Context

| Column | Type | Description |
|--------|------|-------------|
| **channel** | object | Transaction channel (e.g., ATM, POS, Mobile, Internet Banking, USSD) |
| **merchant_category** | object | Type of merchant or business where transaction occurred |
| **bank** | object | Name of the bank processing the transaction |
| **location** | object | Geographic location where transaction was initiated |

### Customer Demographics

| Column | Type | Description |
|--------|------|-------------|
| **age_group** | object | Age bracket of the customer (e.g., 18-25, 26-35, 36-45, etc.) |

### Temporal Features

| Column | Type | Description |
|--------|------|-------------|
| **hour** | int64 | Hour of the day (0-23) when transaction occurred |
| **day_of_week** | int64 | Day of the week (0=Monday, 6=Sunday) |
| **month** | int64 | Month of the year (1-12) |
| **is_weekend** | bool | Whether transaction occurred on weekend (True/False) |
| **is_peak_hour** | bool | Whether transaction occurred during peak banking hours (True/False) |

### Short-term Behavioral Features (24 hours)

| Column | Type | Description |
|--------|------|-------------|
| **tx_count_24h** | float64 | Number of transactions by this customer in the last 24 hours |
| **amount_sum_24h** | float64 | Total amount transacted by customer in the last 24 hours |

### Medium-term Behavioral Features (7 days)

| Column | Type | Description |
|--------|------|-------------|
| **amount_mean_7d** | float64 | Average transaction amount for this customer over the last 7 days |
| **amount_std_7d** | float64 | Standard deviation of transaction amounts over the last 7 days (measures variability) |

### Long-term Behavioral Features (Historical)

| Column | Type | Description |
|--------|------|-------------|
| **tx_count_total** | int64 | Total number of transactions by this customer (all-time) |
| **amount_mean_total** | float64 | Average transaction amount for this customer (all-time) |
| **amount_std_total** | float64 | Standard deviation of all transaction amounts (all-time) |

### Customer Behavioral Patterns

| Column | Type | Description |
|--------|------|-------------|
| **channel_diversity** | int64 | Number of different channels used by this customer |
| **location_diversity** | int64 | Number of different locations this customer has transacted from |
| **amount_vs_mean_ratio** | float64 | Ratio of current transaction amount to customer's mean amount (outlier detection) |
| **online_channel_ratio** | float64 | Proportion of customer's transactions through online channels |

### Target Variable

| Column | Type | Description |
|--------|------|-------------|
| **is_fraud** | int64 | **Target variable**: 1 = Fraudulent transaction, 0 = Legitimate transaction |
| **fraud_technique** | object | Type of fraud technique used (only populated for fraudulent transactions, 3,000 non-null values) |

### Engineered Features - Cyclical Time Encoding

These features use sine and cosine transformations to preserve cyclical nature of time:

| Column | Type | Description |
|--------|------|-------------|
| **hour_sin** | float64 | Sine transformation of hour (captures cyclical pattern: hour 23 is close to hour 0) |
| **hour_cos** | float64 | Cosine transformation of hour |
| **day_sin** | float64 | Sine transformation of day of week |
| **day_cos** | float64 | Cosine transformation of day of week |
| **month_sin** | float64 | Sine transformation of month |
| **month_cos** | float64 | Cosine transformation of month |

### Engineered Features - Transaction Characteristics

| Column | Type | Description |
|--------|------|-------------|
| **amount_log** | float64 | Natural logarithm of transaction amount (normalizes skewed distribution) |
| **amount_rounded** | int64 | Transaction amount rounded to nearest whole number |

### Risk Scoring Features

| Column | Type | Description |
|--------|------|-------------|
| **velocity_score** | float64 | Measures the speed/frequency of transactions (high velocity can indicate fraud) |
| **merchant_risk_score** | float64 | Risk score assigned to the merchant category |
| **composite_risk** | float64 | Combined risk score aggregating multiple risk factors |

## Research Objectives

This dataset can be used for:
- **Fraud Detection Models**: Building machine learning models to identify fraudulent transactions
- **Pattern Analysis**: Understanding behavioral patterns that distinguish fraud from legitimate transactions
- **Feature Importance**: Identifying which features are most predictive of fraud
- **Temporal Analysis**: Analyzing time-based fraud patterns
- **Channel Security**: Evaluating fraud rates across different banking channels
- **Customer Profiling**: Understanding customer segments most vulnerable to fraud

## Key Insights

### Class Imbalance
- Legitimate transactions: 997,000 (99.7%)
- Fraudulent transactions: 3,000 (0.3%)
- **Note**: This severe class imbalance requires special handling (e.g., SMOTE, class weights, stratified sampling)

### Data Quality
- All 38 columns have 1,000,000 non-null values except `fraud_technique`
- `fraud_technique` only has values for the 3,000 fraudulent transactions
- No missing data in core features

## Getting Started

### Requirements
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Loading the Data
```python
data = pd.read_csv('nibss_fraud_dataset.csv')
print(data.shape)
print(data.info())
```

## Suggested Analysis Approaches

1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of transaction amounts
   - Temporal patterns (hourly, daily, monthly)
   - Channel-wise fraud rates
   - Customer behavior analysis

2. **Feature Engineering**
   - Create interaction features
   - Aggregate risk scores
   - Normalize numerical features

3. **Model Development**
   - Handle class imbalance (SMOTE, undersampling, class weights)
   - Try multiple algorithms (Random Forest, XGBoost, Neural Networks)
   - Use appropriate metrics (F1-score, Precision-Recall, AUC-ROC)

4. **Model Evaluation**
   - Focus on recall (catching fraud) vs precision (avoiding false alarms)
   - Analyze false positives and false negatives
   - Feature importance analysis

## License & Usage

This is a research dataset for fraud detection analysis. Ensure compliance with data protection regulations when using financial data.

## Contact

For questions or collaboration on this research project, please refer to the project documentation.

---
*Last Updated: January 2026*
# Fraud-risk-research-ML
