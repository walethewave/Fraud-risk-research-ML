"""
Week 2 — Compute SHAP values for the trained Random Forest fraud model.

Reconstructs the exact train/test split used in notebooks/02_ml_modeling.ipynb
(same RANDOM_STATE and TEST_SIZE, stratified on is_fraud), loads the saved
model, and runs shap.TreeExplainer once over the test set. Results are saved
to outputs/shap_values.pkl for reuse by prompt_builder.py — SHAP is never
computed at inference time (see design.md, latency note).
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_LABEL = "is_fraud"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_model_ready.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nibss_fraud_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_fraud_model.pkl"

OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SHAP_VALUES_PATH = OUTPUT_DIR / "shap_values.pkl"


def reconstruct_test_split():
    df = pd.read_pickle(DATA_PATH)
    X = df.drop(columns=[FRAUD_LABEL])
    y = df[FRAUD_LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


def attach_display_context(X_test: pd.DataFrame) -> pd.DataFrame:
    """Join back human-readable channel/location/age_group columns (by row
    index — verified to align 1:1 with the raw CSV) for prompt display only.
    These are NOT model features and are not passed to the classifier."""
    display_cols = pd.read_csv(
        RAW_DATA_PATH, usecols=["channel", "location", "age_group", "bank"]
    )
    return X_test.join(display_cols.loc[X_test.index])


def main():
    print("Loading processed dataset and reconstructing test split...")
    X_test, y_test = reconstruct_test_split()
    print(f"Test set: {len(X_test):,} rows, {y_test.sum():,} fraud cases")

    print("Loading trained Random Forest model...")
    model = joblib.load(MODEL_PATH)

    model_features = list(model.feature_names_in_)
    X_test_model = X_test[model_features]

    print("Running shap.TreeExplainer over the test set...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_model)
    elapsed = time.time() - t0
    print(f"SHAP computation took {elapsed:.1f}s")

    # Binary classifier: shap_values is either (n_samples, n_features, 2)
    # (newer shap) or a list [class_0, class_1] (older shap). Normalize to
    # the fraud-class (class 1) 2D array.
    if isinstance(shap_values, list):
        fraud_shap = shap_values[1]
    elif shap_values.ndim == 3:
        fraud_shap = shap_values[:, :, 1]
    else:
        fraud_shap = shap_values

    assert fraud_shap.shape == X_test_model.shape, (
        f"SHAP shape {fraud_shap.shape} != test set shape {X_test_model.shape}"
    )

    shap_df = pd.DataFrame(
        fraud_shap, columns=model_features, index=X_test_model.index
    )

    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    print("\nGlobal feature importance (mean |SHAP value|):")
    print(mean_abs_shap.to_string())

    top_feature = mean_abs_shap.index[0]
    print(f"\nTop feature by mean |SHAP|: {top_feature}")
    if top_feature not in ("amount", "composite_risk"):
        print(
            "WARNING: top feature does not match METHODOLOGY.md's reported "
            "dominance of 'amount' (52%) / 'composite_risk' (18%) — verify "
            "model/test-set alignment before trusting downstream results."
        )

    X_test_display = attach_display_context(X_test)

    payload = {
        "shap_values": shap_df,
        "X_test": X_test_model,
        "X_test_display": X_test_display,
        "y_test": y_test,
        "expected_value": explainer.expected_value,
        "model_features": model_features,
        "mean_abs_shap": mean_abs_shap,
    }
    joblib.dump(payload, SHAP_VALUES_PATH)
    print(f"\nSaved SHAP values + test set context to {SHAP_VALUES_PATH}")


if __name__ == "__main__":
    main()
