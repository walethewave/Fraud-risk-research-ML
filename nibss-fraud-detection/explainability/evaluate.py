"""
Week 5 — Evaluate generated explanations.

Two automated metrics (faithfulness, usefulness) computed over
outputs/explanations.jsonl, plus a reader for the human-rating add-on
(outputs/human_ratings.csv, produced from human_rating_template.csv once a
rater has filled in scores).

Usage:
    python3 evaluate.py                       # automated metrics only
    python3 evaluate.py --human-ratings outputs/human_ratings.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPLANATIONS_PATH = HERE / "outputs" / "explanations.jsonl"
RESULTS_PATH = HERE / "outputs" / "evaluation_results.csv"

FEATURE_KEYWORDS = {
    "amount": ["amount", "transaction size", "transaction value"],
    "hour": ["hour", "am", "pm", "night", "midnight", "morning", "afternoon", "evening"],
    "composite_risk": ["composite risk", "risk score"],
    "merchant_risk_score": ["merchant risk"],
    "day_of_week": ["day of week", "weekend", "weekday"],
    "month": ["month", "seasonal"],
    "age_numeric": ["age"],
}

USEFULNESS_SECTIONS = {
    "reason": [r"reason", r"flagged because", r"primary reason"],
    "factors": [r"factor", r"[-•]\s"],
    "risk_level": [r"\b(high|medium|low)\b.{0,15}risk", r"risk level"],
    "action": [r"\b(block|step-up|step up|monitor|freeze|verify|hold)\b"],
}


def _feature_mentioned(feature_name: str, text: str) -> bool:
    text_lower = text.lower()
    if feature_name.startswith("bank_"):
        bank_name = feature_name.removeprefix("bank_").lower()
        return bank_name in text_lower or "bank" in text_lower
    keywords = FEATURE_KEYWORDS.get(feature_name, [feature_name.replace("_", " ")])
    return any(kw in text_lower for kw in keywords)


def faithfulness_score(top_shap_features, explanation_text: str) -> float:
    """Fraction of the top-3 SHAP features mentioned (by name or plain-English
    paraphrase) in the explanation text."""
    if not top_shap_features:
        return 0.0
    hits = sum(
        1 for feature_name, _ in top_shap_features
        if _feature_mentioned(feature_name, explanation_text)
    )
    return hits / len(top_shap_features)


def usefulness_score(explanation_text: str) -> int:
    """0-3: how many of the 4 required sections are present, capped display
    at 3 extra points beyond the first required section (reason is assumed
    baseline content of any explanation)."""
    text_lower = explanation_text.lower()
    present = 0
    for section, patterns in USEFULNESS_SECTIONS.items():
        if any(re.search(p, text_lower) for p in patterns):
            present += 1
    return min(present, 4)


def load_explanations():
    if not EXPLANATIONS_PATH.exists():
        raise FileNotFoundError(
            f"{EXPLANATIONS_PATH} not found — run generate_explanations.py first."
        )
    records = []
    with open(EXPLANATIONS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_automated_eval(records):
    rows = []
    for r in records:
        rows.append({
            "row_index": r["row_index"],
            "case_type": r["case_type"],
            "faithfulness": faithfulness_score(r["top_shap_features"], r["explanation"]),
            "usefulness": usefulness_score(r["explanation"]),
        })
    df = pd.DataFrame(rows)

    print("Automated evaluation")
    print("-" * 40)
    print(f"n = {len(df)}")
    print(f"Mean faithfulness: {df['faithfulness'].mean():.3f}")
    print(f"Mean usefulness:   {df['usefulness'].mean():.3f} / 4")
    print()
    print("By case type:")
    print(df.groupby("case_type")[["faithfulness", "usefulness"]].mean().to_string())

    if df["faithfulness"].std() == 0 or df["usefulness"].std() == 0:
        print(
            "\nWARNING: zero variance in a metric — check that the keyword "
            "matching isn't trivially matching everything or nothing."
        )

    return df


def run_human_eval(human_ratings_path: Path):
    ratings = pd.read_csv(human_ratings_path)
    if "score" not in ratings.columns:
        raise ValueError("human ratings CSV must have a 'score' column (1-5)")
    ratings = ratings.dropna(subset=["score"])
    print("\nHuman rating evaluation")
    print("-" * 40)
    print(f"n = {len(ratings)}")
    print(f"Mean score:     {ratings['score'].mean():.2f} / 5")
    print(f"Std dev:        {ratings['score'].std():.2f}")
    return ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-ratings", type=Path, default=None)
    args = parser.parse_args()

    records = load_explanations()
    auto_df = run_automated_eval(records)
    auto_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved automated results to {RESULTS_PATH}")

    if args.human_ratings:
        run_human_eval(args.human_ratings)


if __name__ == "__main__":
    main()
