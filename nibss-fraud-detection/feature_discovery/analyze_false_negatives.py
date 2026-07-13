"""
Feature discovery, step 1 — find signal the current model can't see.

The production Random Forest uses only 17 features (amount, hour,
day_of_week, month, merchant_risk_score, composite_risk, age_numeric, and
10 one-hot bank columns). But the raw NIBSS dataset has 38 columns —
several behavioral/velocity features were engineered in the EDA notebook
and then dropped before modeling (per METHODOLOGY.md section 3.3.4/3.3.5).

This script checks: among transactions the model actually MISSED (false
negatives — real fraud predicted as legitimate), do any of those unused
columns show a different distribution than in genuinely legitimate
transactions? If so, that's a candidate feature the model is currently
blind to.

Outputs outputs/fn_analysis.json — per-candidate-column stats for false
negatives (FN), caught fraud (TP), and legitimate transactions (TN).
"""
import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXPLAINABILITY_OUTPUTS = PROJECT_ROOT / "explainability" / "outputs"
SHAP_VALUES_PATH = EXPLAINABILITY_OUTPUTS / "shap_values.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nibss_fraud_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_fraud_model.pkl"
OUTPUT_PATH = HERE / "outputs" / "fn_analysis.json"

# Engineered in the raw dataset but NOT in the current model's 17 features
# (location_diversity excluded: constant at 1 across the whole dataset,
# zero variance, no signal possible).
CANDIDATE_COLUMNS = [
    "tx_count_24h",
    "amount_sum_24h",
    "amount_mean_7d",
    "amount_std_7d",
    "tx_count_total",
    "amount_mean_total",
    "amount_std_total",
    "channel_diversity",
    "online_channel_ratio",
    "amount_vs_mean_ratio",
    "velocity_score",
    "is_weekend",
    "is_peak_hour",
]


def main():
    payload = joblib.load(SHAP_VALUES_PATH)
    model = joblib.load(MODEL_PATH)
    X_test = payload["X_test"][payload["model_features"]]
    y_test = payload["y_test"]

    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    fn_idx = y_test[(y_test == 1) & (y_pred == 0)].index  # missed fraud
    tp_idx = y_test[(y_test == 1) & (y_pred == 1)].index  # caught fraud
    tn_idx = y_test[(y_test == 0) & (y_pred == 0)].index  # correctly passed

    print(f"False negatives (missed fraud): {len(fn_idx)}")
    print(f"True positives (caught fraud):  {len(tp_idx)}")
    print(f"True negatives (correct pass):  {len(tn_idx)}")

    print("Loading candidate columns from raw dataset...")
    raw = pd.read_csv(RAW_DATA_PATH, usecols=CANDIDATE_COLUMNS)

    results = {}
    for col in CANDIDATE_COLUMNS:
        fn_mean = float(raw.loc[fn_idx, col].mean())
        tp_mean = float(raw.loc[tp_idx, col].mean())
        tn_mean = float(raw.loc[tn_idx, col].mean())
        tn_std = float(raw.loc[tn_idx, col].std()) or 1e-9

        # How many standard deviations (of the legitimate-baseline
        # distribution) the missed-fraud mean sits away from normal.
        fn_anomaly_z = (fn_mean - tn_mean) / tn_std

        results[col] = {
            "fn_mean": fn_mean,
            "tp_mean": tp_mean,
            "tn_mean": tn_mean,
            "tn_std": tn_std,
            "fn_anomaly_z": fn_anomaly_z,
        }

    # Rank by |z| — columns where missed fraud looks most different from
    # normal legitimate behavior, i.e. where the model is missing a signal
    # that's actually present in the data.
    ranked = sorted(results.items(), key=lambda kv: -abs(kv[1]["fn_anomaly_z"]))

    print("\nCandidate features ranked by FN anomaly strength (|z-score| vs legitimate baseline):")
    for col, stats in ranked:
        print(f"  {col:25s} z={stats['fn_anomaly_z']:+.2f}  "
              f"(FN mean={stats['fn_mean']:.3f}, legit mean={stats['tn_mean']:.3f})")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "n_false_negatives": len(fn_idx),
        "n_true_positives": len(tp_idx),
        "n_true_negatives": len(tn_idx),
        "candidates_ranked": [{"feature": c, **s} for c, s in ranked],
    }, indent=2))
    print(f"\nSaved analysis to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
