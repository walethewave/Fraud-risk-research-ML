"""
Week 4 — Automate explanation generation.

Loops over all fraud test cases (plus a matched sample of false positives
for contrast, per design.md), builds each prompt via prompt_builder, calls
the LLM, and writes one JSON record per case to outputs/explanations.jsonl.

Usage:
    python3 generate_explanations.py --limit 10      # dry run first (Week 4 step 1)
    python3 generate_explanations.py                  # full run

Requires GEMINI_API_KEY in the environment (see llm_client.py) and that
compute_shap.py has already been run (outputs/shap_values.pkl must exist).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from llm_client import LLMClient
from prompt_builder import Prediction, Transaction, build_prompt

HERE = Path(__file__).resolve().parent
SHAP_VALUES_PATH = HERE / "outputs" / "shap_values.pkl"
EXPLANATIONS_PATH = HERE / "outputs" / "explanations.jsonl"

# Free-tier Gemini quota (5 req/min) makes the full 600+100 case run take
# ~2.5 hours; sampled down per explicit user decision (see manual_prompt_test.md
# sibling note below) to finish in ~20 min while still giving a meaningful
# faithfulness/usefulness sample and enough cases for the 50-case human rating.
N_FRAUD_SAMPLE = 50
N_FALSE_POSITIVE_SAMPLE = 50
TOP_K_SHAP_FEATURES = 3
RANDOM_STATE = 42


def load_payload():
    if not SHAP_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"{SHAP_VALUES_PATH} not found — run compute_shap.py first."
        )
    return joblib.load(SHAP_VALUES_PATH)


def top_shap_for_row(shap_row: pd.Series, k: int = TOP_K_SHAP_FEATURES):
    ranked = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
    return list(ranked.head(k).items())


def select_cases(payload, predictions, probabilities):
    y_test = payload["y_test"]
    is_fraud_pred = predictions == 1

    fraud_case_ids = y_test[y_test == 1].index
    fp_mask = (y_test == 0) & (pd.Series(is_fraud_pred, index=y_test.index))
    fp_case_ids = y_test[fp_mask].index

    rng = np.random.RandomState(RANDOM_STATE)
    if len(fraud_case_ids) > N_FRAUD_SAMPLE:
        fraud_case_ids = pd.Index(
            rng.choice(fraud_case_ids, size=N_FRAUD_SAMPLE, replace=False)
        )
    if len(fp_case_ids) > N_FALSE_POSITIVE_SAMPLE:
        fp_case_ids = pd.Index(
            rng.choice(fp_case_ids, size=N_FALSE_POSITIVE_SAMPLE, replace=False)
        )

    cases = [(idx, "true_fraud") for idx in fraud_case_ids]
    cases += [(idx, "false_positive") for idx in fp_case_ids]
    return cases


def build_case_record(idx, case_type, payload, prediction_label, probability):
    display_row = payload["X_test_display"].loc[idx]
    shap_row = payload["shap_values"].loc[idx]
    top_features = top_shap_for_row(shap_row)

    transaction = Transaction(
        amount=float(display_row["amount"]),
        channel=str(display_row["channel"]),
        hour=int(display_row["hour"]),
        bank=str(display_row["bank"]),
        location=str(display_row["location"]),
        age_group=str(display_row["age_group"]),
    )
    prediction = Prediction(
        is_fraud=bool(prediction_label), fraud_probability=float(probability)
    )
    prompt = build_prompt(transaction, prediction, top_features)

    return {
        "row_index": int(idx),
        "case_type": case_type,
        "actual_is_fraud": int(payload["y_test"].loc[idx]),
        "predicted_is_fraud": int(prediction_label),
        "fraud_probability": float(probability),
        "top_shap_features": [(f, float(v)) for f, v in top_features],
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N cases (dry run)."
    )
    args = parser.parse_args()

    payload = load_payload()
    model_features = payload["model_features"]
    X_test = payload["X_test"][model_features]

    import joblib as _joblib
    model = _joblib.load(HERE.parent / "models" / "random_forest_fraud_model.pkl")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    pred_series = pd.Series(predictions, index=X_test.index)
    prob_series = pd.Series(probabilities, index=X_test.index)

    cases = select_cases(payload, predictions, probabilities)
    print(f"Selected {len(cases)} cases "
          f"({sum(1 for _, t in cases if t == 'true_fraud')} fraud, "
          f"{sum(1 for _, t in cases if t == 'false_positive')} false positives)")

    if args.limit:
        cases = cases[: args.limit]
        print(f"Dry run: limiting to first {len(cases)} cases")

    # Resume support: skip cases already present in explanations.jsonl so
    # re-running after a quota exhaustion doesn't re-spend calls on cases
    # we already have (each case costs real, rate-limited API quota).
    already_done = set()
    if EXPLANATIONS_PATH.exists():
        with open(EXPLANATIONS_PATH) as f:
            for line in f:
                already_done.add(json.loads(line)["row_index"])
    remaining_cases = [(idx, ct) for idx, ct in cases if idx not in already_done]
    print(f"{len(already_done)} cases already done, {len(remaining_cases)} remaining")

    client = LLMClient()
    print(f"Using LLM (starting model): {client.model_name}")

    EXPLANATIONS_PATH.parent.mkdir(exist_ok=True)
    n_written = 0
    with open(EXPLANATIONS_PATH, "a") as f:
        for i, (idx, case_type) in enumerate(remaining_cases, 1):
            record = build_case_record(
                idx, case_type, payload, pred_series.loc[idx], prob_series.loc[idx]
            )
            try:
                explanation = client.explain_prompt(record["prompt"])
            except RuntimeError as exc:
                print(f"\nStopping: all models exhausted for today ({exc})")
                print(f"Wrote {n_written} new cases this run "
                      f"({len(already_done) + n_written}/{len(cases)} total). "
                      f"Re-run this script later to resume — it will skip "
                      f"completed cases automatically.")
                return
            record["explanation"] = explanation
            record["model_used"] = client.model_name
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_written += 1
            print(f"[{i}/{len(remaining_cases)}] row {idx} ({case_type}, {client.model_name}) -> "
                  f"{explanation[:80].replace(chr(10), ' ')}...")

    print(f"\nAppended {n_written} explanations to {EXPLANATIONS_PATH}")


if __name__ == "__main__":
    main()
