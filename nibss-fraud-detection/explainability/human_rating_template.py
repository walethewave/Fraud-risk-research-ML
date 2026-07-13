"""
Week 5 — Export the blind human-rating sheet.

Samples 50 cases from outputs/explanations.jsonl (mix of true fraud and
false positives, per design.md) and writes a CSV with the transaction
summary + explanation text and a blank score column for a human rater to
fill in (1-5, "would you act on this?"). Deliberately omits SHAP values,
correctness, and case_type so the rating is blind.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPLANATIONS_PATH = HERE / "outputs" / "explanations.jsonl"
TEMPLATE_PATH = HERE / "human_rating_template.csv"

SAMPLE_SIZE = 50
RANDOM_STATE = 42


def main():
    if not EXPLANATIONS_PATH.exists():
        raise FileNotFoundError(
            f"{EXPLANATIONS_PATH} not found — run generate_explanations.py first."
        )

    records = []
    with open(EXPLANATIONS_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    n_per_type = SAMPLE_SIZE // 2
    sampled = (
        df.groupby("case_type", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(n_per_type, len(g)), random_state=RANDOM_STATE
        ))
    )
    if len(sampled) < SAMPLE_SIZE:
        remaining = df.drop(sampled.index)
        top_up = remaining.sample(
            n=min(SAMPLE_SIZE - len(sampled), len(remaining)),
            random_state=RANDOM_STATE,
        )
        sampled = pd.concat([sampled, top_up])

    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    rating_sheet = pd.DataFrame({
        "case_id": range(1, len(sampled) + 1),
        "explanation": sampled["explanation"].values,
        "score": ["" for _ in range(len(sampled))],
    })
    rating_sheet.to_csv(TEMPLATE_PATH, index=False)

    # Keep an internal, non-blind mapping for later analysis (not given to raters)
    mapping = pd.DataFrame({
        "case_id": range(1, len(sampled) + 1),
        "row_index": sampled["row_index"].values,
        "case_type": sampled["case_type"].values,
        "actual_is_fraud": sampled["actual_is_fraud"].values,
    })
    mapping.to_csv(HERE / "outputs" / "human_rating_mapping.csv", index=False)

    print(f"Wrote {len(sampled)} cases to {TEMPLATE_PATH} (fill in the 'score' column, 1-5)")
    print(f"Internal mapping (not for raters) saved to outputs/human_rating_mapping.csv")


if __name__ == "__main__":
    main()
