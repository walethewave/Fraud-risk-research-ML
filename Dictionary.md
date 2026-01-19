# Data Dictionary

This section documents every feature used in the fraud risk research and ML pipeline. Features are grouped by function to reflect how banks and fraud teams reason about transactions.

---

## Core Identifiers

| Feature | Type | Description |
|------|------|------|
| transaction_id | string | Unique identifier for each transaction |
| customer_id | string | Unique identifier for each customer |

---

## Transaction Details

| Feature | Type | Description |
|------|------|------|
| timestamp | datetime | Exact date and time the transaction occurred |
| amount | float | Transaction amount in Nigerian Naira |
| amount_log | float | Log transformed transaction amount to reduce skew |
| amount_rounded | int | Transaction amount rounded to nearest whole value |

---

## Channel and Context

| Feature | Type | Description |
|------|------|------|
| channel | categorical | Transaction channel such as ATM, POS, Mobile, Web, ECOM, IB |
| merchant_category | categorical | Merchant category code or business type |
| bank | categorical | Bank that processed the transaction |
| location | categorical | State or region where the transaction originated |

---

## Customer Demographics

| Feature | Type | Description |
|------|------|------|
| age_group | categorical | Customer age bracket |

---

## Temporal Features

| Feature | Type | Description |
|------|------|------|
| hour | int | Hour of day from 0 to 23 |
| day_of_week | int | Day of week from 0 Monday to 6 Sunday |
| month | int | Month of year from 1 to 12 |
| is_weekend | bool | Indicates weekend activity |
| is_peak_hour | bool | Indicates business hour activity |

---

## Cyclical Time Encoding

| Feature | Type | Description |
|------|------|------|
| hour_sin | float | Cyclical encoding of transaction hour |
| hour_cos | float | Cyclical encoding of transaction hour |
| day_sin | float | Cyclical encoding of day of week |
| day_cos | float | Cyclical encoding of day of week |
| month_sin | float | Cyclical encoding of month |
| month_cos | float | Cyclical encoding of month |

---

## Short Term Behavioral Features 24 Hours

| Feature | Type | Description |
|------|------|------|
| tx_count_24h | float | Number of customer transactions in last 24 hours |
| amount_sum_24h | float | Total transaction value in last 24 hours |

---

## Medium Term Behavioral Features 7 Days

| Feature | Type | Description |
|------|------|------|
| amount_mean_7d | float | Mean transaction amount over last 7 days |
| amount_std_7d | float | Amount variability over last 7 days |

---

## Long Term Behavioral Features

| Feature | Type | Description |
|------|------|------|
| tx_count_total | int | Total historical transaction count |
| amount_mean_total | float | Historical average transaction amount |
| amount_std_total | float | Historical amount variability |

---

## Behavioral Diversity Metrics

| Feature | Type | Description |
|------|------|------|
| channel_diversity | int | Number of distinct channels used |
| location_diversity | int | Number of distinct transaction locations |
| online_channel_ratio | float | Share of transactions from digital channels |

---

## Risk and Anomaly Features

| Feature | Type | Description |
|------|------|------|
| amount_vs_mean_ratio | float | Current amount relative to customer baseline |
| velocity_score | float | Transaction frequency and speed indicator |
| merchant_risk_score | float | Risk score assigned to merchant category |
| composite_risk | float | Aggregated risk signal from multiple factors |

---

## Target Variables

| Feature | Type | Description |
|------|------|------|
| is_fraud | int | Fraud label. 1 fraud. 0 legitimate |
| fraud_technique | categorical | Fraud method label for fraudulent cases |

---

## Notes on Usage

• All behavioral and risk features support supervised and unsupervised learning  
• Clustering excludes target variables and identifiers  
• Modeling workflows address heavy class imbalance  
• Feature design aligns with real bank fraud monitoring systems  

This dictionary will expand as feature engineering and modeling progress.
