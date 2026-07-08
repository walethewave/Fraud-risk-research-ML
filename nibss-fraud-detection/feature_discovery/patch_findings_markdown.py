"""
One-off fix: updates the markdown "Section Findings" / Limitations /
Conclusion cells in 02_ml_modeling.ipynb to match the real retrained
numbers (19-feature, leak-checked model) instead of the stale 17-feature
numbers left over from before the feature-discovery fix.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_ml_modeling.ipynb"

CELL_34_RF_FINDINGS = """### 5e. Section Findings — Random Forest

- **AUC-ROC: 0.923** — a 31.4% relative improvement over Logistic Regression (from 0.702 to 0.923). Includes two features added after a feature-discovery audit (see `feature_discovery/FINDINGS.md`): `amount_sum_24h` and a leak-safe `amount_vs_mean_ratio_safe`.
- **Recall: 67.5%** — Random Forest detects 405 of 600 fraud cases in the test set, compared to Logistic Regression's ~328.
- **Precision: 7.08%** — a large jump from the pre-fix model's 0.95%, driven by the two added features sharply reducing false positives.
- **Accuracy: 97.25%** — up from 79.84% pre-fix, because far fewer legitimate transactions are now flagged (~5,300 false positives vs ~40,098 before).
- **Key drivers** (Gini importance, actual ranking): `amount_sum_24h` (0.359) is now the single most important feature, followed by `amount` (0.263) and `amount_vs_mean_ratio_safe` (0.190) — together accounting for over 80% of the model's decisions. `composite_risk` (0.084) ranks fourth. This confirms the feature-discovery finding: relative/velocity context the original 7-feature model never had access to turned out to matter more than any single feature it already had.
"""

CELL_43_LIMITATIONS = """### Limitations

1. **Synthetic dataset**: The NIBSS Fraud Dataset is a synthetic simulation of Nigerian banking transactions. While well-constructed, it may not perfectly replicate all statistical properties of live transactional data. Validation on real anonymised NIBSS data is required before operational deployment.

2. **Feature selection was revised after a feature-discovery audit**: the original model used only 7 base features and reported AUC-ROC 0.822. A closed-loop audit (analyze false negatives → LLM proposes candidates → empirically test AUC → check for leakage) found that `amount_sum_24h` — near-useless alone but highly predictive via interaction with `amount` — had been dropped without justification, and that a related candidate (`amount_vs_mean_ratio`) had look-ahead leakage in its source column. Both issues are fixed in this version (AUC-ROC now 0.923); see `feature_discovery/FINDINGS.md` for the full audit trail. This is a reminder that univariate/correlation-based feature pruning can silently discard valuable interaction-only signal.

3. **Residual leakage risk in `composite_risk`**: this feature is inherited unmodified from the raw dataset and was itself constructed upstream using the original (leaky) `amount_vs_mean_ratio`. It was not rebuilt as part of this fix (its exact construction formula is not available), so a small residual exposure to the same look-ahead issue may remain. Recommend requesting the raw formula from the data provider or excluding `composite_risk` in a follow-up ablation.

4. **Precision (7.08%)**: still low in absolute terms — the 332:1 imbalance makes high precision difficult at any useful recall level, even after the AUC improvement. In production, threshold tuning and real-time risk-banding would address this; recall at the current 0.5 threshold should be re-optimized for the new model (see `feature_discovery/FINDINGS.md`'s note that AUC gains don't automatically translate to fewer false negatives at a fixed threshold).

5. **Single time window**: The dataset covers 2023 only. Fraud patterns evolve — a model trained on 2023 data may underperform on 2026+ fraud tactics without periodic retraining.

6. **No SHAP explainability integrated into this notebook**: see `explainability/` for the SHAP + LLM explanation pipeline built separately, which will need re-running against this updated 19-feature model.

7. **Feature count**: 9 base features (`amount`, `hour`, `day_of_week`, `month`, `merchant_risk_score`, `composite_risk`, `age_numeric`, `amount_sum_24h`, `amount_vs_mean_ratio_safe`) + 10 bank one-hot columns = 19 model inputs.

### Future Work

1. **Re-run the SHAP + LLM explainability pipeline** against this 19-feature model (feature importances have changed substantially).
2. **Resolve the `composite_risk` residual leakage question** — request its construction formula from the data provider, or run an ablation without it.
3. **Threshold re-optimization** — the AUC gain did not automatically reduce false negatives at the default 0.5 threshold; re-run the cost-sensitive threshold analysis from section 4d/4e style tuning on this model.
4. **XGBoost / LightGBM** — gradient boosting may further improve AUC on top of the corrected feature set.
5. **Graph neural networks** — model transaction network relationships between customers.
6. **Real-time streaming** — adapt features for sub-second scoring in production APIs (note: `amount_sum_24h` requires a live rolling aggregate, not a batch precomputed column, for real-time use).
7. **Temporal validation** — train on 2023 Q1-Q3, test on Q4 to simulate real deployment drift.
"""

CELL_45_CONCLUSION = """### Conclusion

This study demonstrates that machine learning can deliver meaningful fraud detection performance on Nigerian banking data. The Random Forest model (AUC-ROC = 0.923, Recall = 67.5%) establishes a rigorous, reproducible baseline — revised from an earlier 0.822 AUC version after a feature-discovery audit found the original feature selection had dropped a highly predictive feature (`amount_sum_24h`) and that a related candidate feature had look-ahead leakage (fixed as `amount_vs_mean_ratio_safe`). See `feature_discovery/FINDINGS.md` for the full audit trail, including an honest accounting of what was corrected and what residual risk (via `composite_risk`) remains open.

---

### Business Recommendations

| Priority | Recommendation | Supporting Evidence |
|:---:|:---|:---|
| 🔴 **Critical** | Implement enhanced authentication for all night-window transactions (22:00–05:59) | Fraud rate peaks at 0.363% at 1:00 AM — 40% above business-hour baseline |
| 🔴 **Critical** | Deploy amount-based velocity alerts for ATM channels | ATM fraud transactions average ₦238,959 — 600% above legitimate ATM mean |
| 🔴 **Critical** | Re-run threshold optimization on the updated model | AUC rose from 0.822 to 0.923, but recall at the default 0.5 threshold only rose modestly — the full benefit requires re-tuning the decision threshold |
| 🟠 **High** | Apply elevated Mobile transaction scrutiny for night sessions | Mobile is the only channel where fraud INCREASES at night (+14.0%) |
| 🟠 **High** | Prioritise Abuja-region fraud controls | Abuja has the highest geographic fraud rate (0.328%), with Sterling × Abuja peaking at 0.470% |
| 🟡 **Medium** | Introduce age-targeted fraud education for 40+ customers | 40+ accounts for 40.8% of all fraud victims |
| 🟡 **Medium** | Retrain the Random Forest model quarterly | Fraud tactics evolve — the model must adapt to remain effective |
| 🟡 **Medium** | Request the `composite_risk` construction formula from the data provider | Needed to fully resolve the residual leakage question flagged in Limitations |
| 🟢 **Ongoing** | Invest in SHAP-based explainability infrastructure | Necessary for CBN regulatory compliance and customer dispute resolution — re-run against the updated 19-feature model |
"""


def replace_markdown(cells, marker: str, new_source: str):
    for cell in cells:
        if cell["cell_type"] == "markdown" and marker in "".join(cell["source"]):
            cell["source"] = new_source.splitlines(keepends=True)
            return
    raise ValueError(f"Markdown cell containing {marker!r} not found")


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    replace_markdown(cells, "Section Findings — Random Forest", CELL_34_RF_FINDINGS)
    replace_markdown(cells, "### Limitations", CELL_43_LIMITATIONS)
    replace_markdown(cells, "### Conclusion", CELL_45_CONCLUSION)
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print("Patched Section Findings, Limitations, and Conclusion cells")


if __name__ == "__main__":
    main()
