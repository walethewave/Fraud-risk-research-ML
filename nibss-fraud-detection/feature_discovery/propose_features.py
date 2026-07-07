"""
Feature discovery, step 2 — LLM proposes which candidate feature(s) to test.

One LLM call (economical on Gemini's shared daily quota), grounded strictly
in the statistics from analyze_false_negatives.py. The model is instructed
to reason only from the given numbers, not invent domain claims about
Nigerian banking that aren't supported by the data — same anti-hallucination
discipline as the explainability prompt template.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "explainability"))

from llm_client import LLMClient  # noqa: E402

FN_ANALYSIS_PATH = HERE / "outputs" / "fn_analysis.json"
PROPOSAL_PATH = HERE / "outputs" / "feature_proposal.json"

PROMPT_TEMPLATE = """You are a fraud model feature engineer reviewing a Random Forest
fraud model for a Nigerian bank. The model currently uses 17 features
(transaction amount, hour, day of week, month, merchant risk score,
composite risk score, customer age, and bank identity).

The model missed {n_fn} fraud cases out of {n_total_fraud} in the test set
(false negatives). Below are behavioral/velocity features that were
engineered during EDA but NOT included in the final model, along with how
their average value compares between missed fraud, caught fraud, and
normal legitimate transactions. The "z-score" shows how many standard
deviations (of the legitimate-transaction distribution) the missed-fraud
average sits away from normal — larger |z| means the missed fraud looks
more anomalous on that feature than the model can currently detect.

CANDIDATE FEATURES (ranked by |z-score|):
{feature_table}

Using ONLY the numbers above — do not invent claims about Nigerian banking
behavior not supported by this data — recommend the TOP 2 candidate
features most likely to help the model catch more of the missed fraud if
added. For each, state:
1. The exact feature name (must match one from the list above)
2. One sentence of reasoning grounded in its z-score and mean comparison
3. A confidence level (High/Medium/Low) — be honest that these z-scores
   are modest (well under 1.0), so confidence should reflect that

Respond in this exact format, one block per recommended feature:
FEATURE: <name>
REASONING: <one sentence>
CONFIDENCE: <High/Medium/Low>
"""


def build_prompt(analysis: dict) -> str:
    rows = []
    for c in analysis["candidates_ranked"]:
        rows.append(
            f"- {c['feature']}: z={c['fn_anomaly_z']:+.2f}, "
            f"missed-fraud mean={c['fn_mean']:.3f}, "
            f"caught-fraud mean={c['tp_mean']:.3f}, "
            f"legitimate mean={c['tn_mean']:.3f}"
        )
    return PROMPT_TEMPLATE.format(
        n_fn=analysis["n_false_negatives"],
        n_total_fraud=analysis["n_false_negatives"] + analysis["n_true_positives"],
        feature_table="\n".join(rows),
    )


def parse_recommended_features(response_text: str, valid_names: list[str]) -> list[str]:
    """Extract FEATURE: <name> lines, keeping only names that actually
    exist in our candidate list (defends against the LLM inventing a name
    or a typo — the pipeline should never try to train on a made-up column)."""
    recommended = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("FEATURE:"):
            name = line.split(":", 1)[1].strip().strip("*").strip()
            if name in valid_names:
                recommended.append(name)
    return recommended


def main():
    if not FN_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"{FN_ANALYSIS_PATH} not found — run analyze_false_negatives.py first."
        )
    analysis = json.loads(FN_ANALYSIS_PATH.read_text())
    valid_names = [c["feature"] for c in analysis["candidates_ranked"]]

    prompt = build_prompt(analysis)
    print("Prompt sent to LLM:\n" + "-" * 60)
    print(prompt)
    print("-" * 60)

    client = LLMClient()
    response = client.explain_prompt(prompt)
    print(f"\nLLM ({client.model_name}) response:\n" + "-" * 60)
    print(response)

    recommended = parse_recommended_features(response, valid_names)
    print(f"\nParsed recommended features (validated against candidate list): {recommended}")

    if not recommended:
        print(
            "\nWARNING: could not parse any valid feature names from the LLM "
            "response — falling back to the top-2 by |z-score| so the "
            "pipeline can still proceed to the empirical test."
        )
        recommended = valid_names[:2]

    PROPOSAL_PATH.write_text(json.dumps({
        "prompt": prompt,
        "raw_response": response,
        "model_used": client.model_name,
        "recommended_features": recommended,
    }, indent=2))
    print(f"\nSaved proposal to {PROPOSAL_PATH}")


if __name__ == "__main__":
    main()
