"""
Exports outputs/explanations.jsonl (LLM output only) joined with the full
transaction context from outputs/shap_values.pkl (amount, channel, bank,
location, age_group, hour) into a single denormalized JSON array for the
Next.js frontend at ../frontend/data/cases.json.

Re-run this any time explanations.jsonl gains new rows (e.g. after resuming
generate_explanations.py once the Gemini daily quota resets).
"""
import json
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
SHAP_VALUES_PATH = HERE / "outputs" / "shap_values.pkl"
EXPLANATIONS_PATH = HERE / "outputs" / "explanations.jsonl"
FRONTEND_DATA_PATH = HERE.parent / "frontend" / "data" / "cases.json"


def main():
    payload = joblib.load(SHAP_VALUES_PATH)
    display = payload["X_test_display"]

    cases = []
    with open(EXPLANATIONS_PATH) as f:
        for line in f:
            record = json.loads(line)
            idx = record["row_index"]
            row = display.loc[idx]

            cases.append({
                "id": idx,
                "caseType": record["case_type"],
                "actualIsFraud": bool(record["actual_is_fraud"]),
                "predictedIsFraud": bool(record["predicted_is_fraud"]),
                "fraudProbability": record["fraud_probability"],
                "amount": float(row["amount"]),
                "channel": str(row["channel"]),
                "hour": int(row["hour"]),
                "bank": str(row["bank"]),
                "location": str(row["location"]),
                "ageGroup": str(row["age_group"]),
                "topShapFeatures": [
                    {"feature": f, "value": v} for f, v in record["top_shap_features"]
                ],
                "explanation": record["explanation"],
                "modelUsed": record.get("model_used", "gemini-2.5-flash"),
            })

    # Correctness-first ordering: true fraud caught, then false positives,
    # highest confidence first within each group — most useful triage order.
    cases.sort(key=lambda c: (c["caseType"], -c["fraudProbability"]))

    FRONTEND_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Wrote {len(cases)} cases to {FRONTEND_DATA_PATH}")


if __name__ == "__main__":
    main()
