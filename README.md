# Fighting Fraud with Machine Learning  
**Building interpretable AI systems for real-time payment fraud detection**

## What This Is
I'm building production-ready fraud detection models for large-scale interbank payment systems. This isn't a Kaggle competition or a toy project—it's the messy, iterative work of understanding fraud as an adversarial system that learns and adapts.

The code here reflects how senior ML engineers actually work: baseline models first, systematic improvements, heavy emphasis on interpretability, and constant questioning of assumptions. If you're looking for 99% accuracy claims, you won't find them here. If you want to see how to build reliable fraud detection in the real world, keep reading.

---

## Why This Matters
Fraud detection is fundamentally different from most ML problems. Fraudsters adapt. They test your system's boundaries, find blind spots, and exploit them before you've even retrained your model. Static rules fail. Black-box models fail differently. You need systems that detect *deviation from normal*, not memorized patterns.

I'm approaching this like the adversarial problem it actually is:
- **Time matters** - Fraudsters attack when monitoring is weak (midnight to 5 AM)
- **Channels behave differently** - Mobile fraud looks nothing like ATM fraud
- **Velocity is intent** - How fast someone transacts reveals more than what they buy
- **Amount patterns shift** - Fraudsters adjust transaction sizes based on what gets flagged
- **Context is everything** - A ₦50,000 transaction at 3 AM on mobile from a new location is a different beast than the same transaction at noon at your usual ATM

---

## The Data
**1 million transactions. 3,000 fraudulent. That's 0.3% fraud rate.**

This is the class imbalance you deal with in production. No synthetic oversampling tricks to make the problem easier. No balanced datasets. Just the brutal reality that fraud is rare, expensive when missed, and devastating when you cry wolf too often.

**Scale:**
- 1,000,000 transactions across multiple Nigerian banks
- 38 engineered features (temporal, behavioral, risk-based)
- 6 transaction channels (ATM, POS, Mobile, Web, E-Commerce, Internet Banking)
- Real-world messiness: overlapping signals, no magic bullet feature
- **Dataset**: [NIBSS Fraud Dataset on Kaggle](https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset)

**The Challenge:**
- 0.3% positive class (good luck with accuracy as a metric)
- Fraud hides inside legitimate patterns
- False positives destroy user trust
- False negatives cost real money
- Models must explain decisions (regulatory requirement + trust)

---

## Feature Engineering (The Hard Part)
Features aren't data columns. Features are hypotheses about fraud behavior encoded as numbers.

**What I Built:**
- **Temporal encodings** - Hour-of-day and day-of-week as cyclical features (fraudsters are creatures of habit)
- **Velocity signals** - Transaction count and amount over 24h/7d/lifetime windows
- **Behavioral diversity** - Channel switching and location hopping patterns
- **Risk composites** - Weighted combinations of amount deviation, velocity, and diversity
- **One-hot bank encodings** - Bank-specific fraud patterns (using int8 for memory efficiency)

**What I Dropped (45 features):**  
Started with 62 features. Aggressively pruned low-signal noise. Kept 17 predictors that actually matter. More features ≠ better models. More features = more ways to overfit.

**Engineering Philosophy:**  
Every feature answers: *"How weird is this transaction for this user?"*  
Not "Is this transaction big?" but "Is this bigger than what this user normally does?"

---

## What The Data Actually Says

**Findings that held up under modeling:**

1. **Time-of-day is real** - Fraud risk spikes 2-3x between midnight and 5 AM when monitoring teams are understaffed
2. **Mobile is the wild west** - Faster fraud velocity, higher amounts, weaker controls
3. **Amount inflation is systematic** - Fraudulent transactions average 2x higher than legitimate ones
4. **Bank-level variance is low** - Fraud rates cluster tightly (0.28-0.32%), suggesting shared infrastructure weakness
5. **Interaction effects dominate** - Bank × Channel × Time combinations matter more than any single feature

**What surprised me:**
- Location alone is weak. Location + velocity + unusual channel = strong signal
- More transaction history doesn't always help (fraudsters build legitimacy first)
- The best fraud indicator is "doing something you've never done before, quickly"

---

## Models & Results

### Baseline: Logistic Regression
**AUC-ROC: 0.7020**  
Started here because you need to know what "simple" performance looks like. Logistic Regression with `class_weight='balanced'` gave us a baseline to beat.

**The ugly truth:**
- Caught 54.2% of fraud (missed 46%)
- 99.1% false positive rate at default threshold
- Precision: 0.87% (for every real fraud, 113 false alarms)
- Unusable in production

**What it taught us:**  
Linear models struggle with fraud's non-linear decision boundaries. Time to bring in the trees.

---

### Production Model: Random Forest
**AUC-ROC: 0.8223 (+17% improvement)**

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',  # Critical for imbalanced data
    n_jobs=-1
)
```

**Performance:**
- **Recall: 64%** - Catches nearly 2 out of 3 fraud cases
- **AUC-ROC: 0.8223** - "Good" discrimination (industry-standard metric)
- **384 true positives** vs LR's 325 (+59 fraud cases caught)
- Trade-off: 3,125 more false positives, but catching more fraud

**Why Random Forest?**
1. Handles non-linear patterns (fraud doesn't follow straight lines)
2. Robust to feature scaling (no preprocessing needed)
3. Built-in feature importance (interpretability matters)
4. `class_weight='balanced'` handles the 0.3% imbalance
5. Fast enough for 800K training samples

---

### Interpretability: SHAP Analysis
Built SHAP (SHapley Additive exPlanations) analysis because "the model says so" doesn't fly with fraud analysts or regulators.

**What SHAP revealed:**
- **Transaction amount** dominates (52% importance) - Size matters most
- **Composite risk score** second (18%) - Our engineered feature works
- **Month and temporal features** (9%) - Seasonal fraud patterns confirmed
- **Bank encodings** matter differently - Institution-specific risk profiles

**Visualization Suite:**
- Summary plots (global feature impact)
- Dependence plots (how feature values affect predictions)
- Force plots (individual transaction explanations)
- Waterfall plots (step-by-step decision breakdown)

This isn't just model debugging. This is how you build trust with stakeholders who need to understand *why* a transaction got flagged.

---

## Repository Structure

```plaintext
fraud-detection/
├── logistic regression.ipynb    # Full pipeline: EDA → Preprocessing → Modeling → SHAP
├── data_clean.csv               # Engineered features (local only, not in git)
├── nibss_fraud_dataset.csv      # Raw data (local only, not in git)
└── README.md                    # You are here
```

**Code Quality:**
- Clean, documented cells with markdown headers
- Professional visualizations (matplotlib + seaborn)
- Memory-efficient techniques (int8 encodings, sampling for SHAP)
- Reproducible (random_state=42 everywhere)
- Production-minded (separate train/test, proper CV strategy)

---

## What's Next

**Immediate improvements (in priority order):**
1. **Threshold optimization** - Find the precision/recall sweet spot for business needs
2. **Hyperparameter tuning** - RandomizedSearchCV to push AUC to 0.85+
3. **SMOTE/ADASYN** - Address class imbalance at the data level
4. **XGBoost** - Likely to beat Random Forest by 2-3% AUC
5. **Ensemble methods** - Combine multiple models for robust predictions

**Production considerations:**
- Model versioning and experiment tracking (MLflow)
- Real-time inference pipeline (FastAPI or similar)
- Monitoring for model drift and adversarial adaptation
- A/B testing framework for safe deployment
- Explainability dashboard for fraud analysts

---

## Philosophy

**What I believe about fraud detection:**
- Interpretability isn't optional—it's the product
- False positives destroy user trust faster than fraud destroys money
- Fraudsters adapt faster than you can retrain, so build adaptive systems
- The best model is one that fraud analysts understand and trust
- Engineering matters more than algorithms (80% of the work is feature engineering)

**What this repo is:**
- A realistic view of production ML for adversarial problems
- Clean code that others can actually learn from
- Honest about trade-offs and limitations
- Built for stakeholders, not competitions

**What this repo isn't:**
- A 99% accuracy miracle (if someone claims that on 0.3% fraud rate, they're lying)
- Overly polished (real work is messy)
- A finished product (fraud detection is never "done")

---

## For ML Engineers

If you're building fraud detection systems, here's what I'd do differently next time:
- Start with SMOTE earlier (don't fight the imbalance with weights alone)
- Build threshold optimization into the first model iteration
- Invest more in temporal feature engineering (time windows matter)
- Create a fraud simulation framework for testing adversarial scenarios
- Build the explainability dashboard alongside the model, not after

The hardest part isn't building a model that performs well. It's building a model that performs well *and* that people trust enough to actually use.

---

## Who This Is For
- ML engineers building fraud/risk systems in production
- Data scientists dealing with extreme class imbalance
- Anyone tired of overfitted Kaggle solutions and wanting to see real-world ML
- Fraud analysts who need to understand how these models actually work

---

## License
MIT License. Use it, learn from it, improve it. If you build something cool with this approach, I'd love to hear about it.

---

**Remember:** Fraud detection isn't about catching every fraudster. It's about making fraud expensive enough that it's not worth the effort. Every model improvement is a small increase in the cost of attack. Keep raising the bar.

*Built by someone who believes ML should be interpretable, production-ready, and honest about its limitations.*

