# Financial Fraud Research and Machine Learning Study  
**Behavioral, Temporal, and Risk-Based Analysis of Large-Scale Payment Transactions**

## Project Status
This repository documents an **ongoing research and machine learning investigation** into financial fraud dynamics within large-scale interbank payment systems.  
The work is intentionally incomplete. Findings, models, and interpretations will continue to evolve as analysis deepens.

This repo reflects how long-horizon research is conducted inside mature AI and risk organizations. Incremental insight beats premature conclusions.

---

## Research Motivation
Fraud does not occur randomly.  
Fraud adapts to human behavior, system constraints, monitoring gaps, and response latency.

This project studies fraud as a **behavioral system**, not a binary classification problem.

The primary goal is to understand:
- Why fraud risk increases at specific times
- How transaction velocity signals intent
- Where channel design creates asymmetric risk
- When amount size shifts fraudster behavior
- Whether fraud differences arise from banks or from shared infrastructure

---

## Dataset Overview
- **Transactions**: 1,000,000  
- **Fraud Cases**: 3,000  
- **Observed Fraud Rate**: 0.30 percent  
- **Features**: 38 engineered and raw variables  
- **Banks**: Multiple Nigerian commercial banks  
- **Channels**: ATM, POS, Mobile, Web, E-Commerce, Internet Banking  
- **Currency**: Nigerian Naira ₦  
- **Dataset Link**: [Download CSV from Kaggle](https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset?select=nibss_fraud_dataset.csv)  

The dataset mirrors real production constraints.
- Extreme class imbalance
- Overlapping risk signals
- No single dominant predictor
- Fraud hidden inside normal behavior

---

## Feature Design Philosophy
Features reflect **behavioral deviation**, not static thresholds.

### Core Groups
- Transaction context and metadata
- Temporal signals with cyclical encoding
- Short, medium, and long-horizon customer behavior
- Channel usage diversity
- Location variability
- Velocity and composite risk indicators

Each feature exists to answer one question:
How abnormal is this transaction relative to the actor behind it.

---

## Research Axes
Current work focuses on five intersecting axes.

### 1. Time-Based Risk
- Hourly fraud rate shifts
- Night versus business-hour behavior
- Velocity escalation during low-response windows

### 2. Channel Behavior
- Physical versus digital risk asymmetry
- Automated fraud signatures on mobile
- Cautious fraud behavior on monitored channels

### 3. Velocity Dynamics
- Legitimate versus fraudulent transaction speed
- Channel-specific velocity deviations
- Time-constrained fraud execution patterns

### 4. Amount Sensitivity
- Fraud amplification at higher ranges
- Channel-dependent amount thresholds
- Average fraudulent amount inflation across channels

### 5. Bank and Location Interaction
- Systemic versus institution-specific risk
- Risk concentration at bank–location–channel intersections
- Volatility as a stronger signal than raw rate

---

## Key Interim Observations
These findings remain provisional and subject to revision.

- Fraud velocity increases sharply between midnight and early morning hours
- Mobile channels exhibit faster fraudulent velocity than legitimate usage
- Fraudulent transactions consistently exceed legitimate amounts
- Bank-level fraud rates remain tightly clustered, suggesting systemic exposure
- Location risk emerges primarily through interaction effects, not isolation

No single factor explains fraud. Patterns emerge only when dimensions intersect.

---

## Machine Learning Direction
Modeling follows research insight, not the reverse.

Planned approaches include:
- Tree-based and boosting models
- Cost-sensitive optimization
- Precision–recall driven evaluation
- Error surface analysis rather than headline accuracy
- Hybrid rule plus ML reasoning

Models serve the research. Research does not serve the model.



```

Structure will evolve alongside the research.

---

## Research Ethos
This repository favors:
- Clear assumptions over polished narratives
- Measured interpretation over speculation
- Systems thinking over single-metric optimization

The intent is not to present a finished product.  
The intent is to document how fraud understanding is built.

---

## Intended Audience
- Fraud and risk analysts
- Data scientists working with imbalanced systems
- ML engineers building decision pipelines
- Researchers studying adversarial behavior in financial systems



## License
Research and educational use only.


**Fraud adapts faster than rules.  
Understanding behavior slows it down.**

