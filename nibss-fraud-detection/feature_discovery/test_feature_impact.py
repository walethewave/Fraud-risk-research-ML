"""
Feature discovery, step 3 — does adding the LLM-proposed feature(s)
actually move AUC?

Retrains the same RandomForestClassifier (identical hyperparameters,
identical random_state, identical stratified 80/20 split) with the
baseline 17 features vs. baseline + candidate feature(s) from
propose_features.py, and compares AUC-ROC, recall, and — critically —
whether the specific false-negative cases identified in step 1 actually
get caught by the augmented model.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_LABEL = "is_fraud"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_model_ready.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nibss_fraud_dataset.csv"
PROPOSAL_PATH = HERE / "outputs" / "feature_proposal.json"
RESULTS_PATH = HERE / "outputs" / "feature_impact_results.json"

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def load_augmented_dataset(candidate_features: list[str]) -> pd.DataFrame:
    base = pd.read_pickle(DATA_PATH)
    extra = pd.read_csv(RAW_DATA_PATH, usecols=candidate_features)
    extra.index = base.index
    return base.join(extra)


def train_and_eval(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols]
    y = df[FRAUD_LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    fn_idx = y_test[(y_test == 1) & (pd.Series(y_pred, index=y_test.index) == 0)].index

    return {
        "auc": float(auc),
        "recall": float(recall),
        "n_fraud": int(y_test.sum()),
        "n_false_negatives": len(fn_idx),
        "false_negative_row_indices": [int(i) for i in fn_idx],
    }


def main():
    if not PROPOSAL_PATH.exists():
        raise FileNotFoundError(f"{PROPOSAL_PATH} not found — run propose_features.py first.")
    proposal = json.loads(PROPOSAL_PATH.read_text())
    candidate_features = proposal["recommended_features"]
    print(f"Testing candidate features: {candidate_features}")

    baseline_model = joblib.load(PROJECT_ROOT / "models" / "random_forest_fraud_model.pkl")
    baseline_feature_cols = list(baseline_model.feature_names_in_)

    df = load_augmented_dataset(candidate_features)

    print("\nTraining baseline model (17 features, re-trained fresh for a fair comparison)...")
    baseline_result = train_and_eval(df, baseline_feature_cols)

    print("Training augmented model (17 + candidate features)...")
    augmented_feature_cols = baseline_feature_cols + candidate_features
    augmented_result = train_and_eval(df, augmented_feature_cols)

    auc_delta = augmented_result["auc"] - baseline_result["auc"]
    recall_delta = augmented_result["recall"] - baseline_result["recall"]

    baseline_fn_set = set(baseline_result["false_negative_row_indices"])
    augmented_fn_set = set(augmented_result["false_negative_row_indices"])
    newly_caught = baseline_fn_set - augmented_fn_set
    newly_missed = augmented_fn_set - baseline_fn_set

    print("\n" + "=" * 60)
    print(f"Baseline  AUC={baseline_result['auc']:.4f}  recall={baseline_result['recall']:.4f}  "
          f"FN={baseline_result['n_false_negatives']}")
    print(f"Augmented AUC={augmented_result['auc']:.4f}  recall={augmented_result['recall']:.4f}  "
          f"FN={augmented_result['n_false_negatives']}")
    print(f"AUC delta:    {auc_delta:+.4f}")
    print(f"Recall delta: {recall_delta:+.4f}")
    print(f"Previously-missed cases now caught: {len(newly_caught)}")
    print(f"Previously-caught cases now missed: {len(newly_missed)}")
    print("=" * 60)

    verdict = (
        "IMPROVED" if auc_delta > 0.002 else
        "NO MEANINGFUL CHANGE" if abs(auc_delta) <= 0.002 else
        "WORSE"
    )
    print(f"\nVerdict: {verdict} (threshold: |ΔAUC| > 0.002 to count as meaningful)")

    RESULTS_PATH.write_text(json.dumps({
        "candidate_features": candidate_features,
        "baseline": baseline_result,
        "augmented": augmented_result,
        "auc_delta": auc_delta,
        "recall_delta": recall_delta,
        "n_newly_caught": len(newly_caught),
        "n_newly_missed": len(newly_missed),
        "verdict": verdict,
    }, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
