"""
Packages fn_analysis.json + feature_proposal.json + feature_impact_results.json
into a single JSON file for the frontend's /discovery page.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FRONTEND_DATA_PATH = HERE.parent / "frontend" / "data" / "discovery.json"


def main():
    fn_analysis = json.loads((OUTPUTS / "fn_analysis.json").read_text())
    proposal = json.loads((OUTPUTS / "feature_proposal.json").read_text())
    impact = json.loads((OUTPUTS / "feature_impact_results.json").read_text())

    payload = {
        "falseNegativeAnalysis": {
            "nFalseNegatives": fn_analysis["n_false_negatives"],
            "nTruePositives": fn_analysis["n_true_positives"],
            "nTrueNegatives": fn_analysis["n_true_negatives"],
            "candidates": [
                {
                    "feature": c["feature"],
                    "fnMean": c["fn_mean"],
                    "tpMean": c["tp_mean"],
                    "tnMean": c["tn_mean"],
                    "anomalyZ": c["fn_anomaly_z"],
                }
                for c in fn_analysis["candidates_ranked"]
            ],
        },
        "llmProposal": {
            "modelUsed": proposal["model_used"],
            "rawResponse": proposal["raw_response"],
            "recommendedFeatures": proposal["recommended_features"],
        },
        "impact": {
            "candidateFeatures": impact["candidate_features"],
            "baselineAuc": impact["baseline"]["auc"],
            "baselineRecall": impact["baseline"]["recall"],
            "baselineFalseNegatives": impact["baseline"]["n_false_negatives"],
            "augmentedAuc": impact["augmented"]["auc"],
            "augmentedRecall": impact["augmented"]["recall"],
            "augmentedFalseNegatives": impact["augmented"]["n_false_negatives"],
            "aucDelta": impact["auc_delta"],
            "recallDelta": impact["recall_delta"],
            "nNewlyCaught": impact["n_newly_caught"],
            "nNewlyMissed": impact["n_newly_missed"],
            "verdict": impact["verdict"],
        },
    }

    FRONTEND_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote discovery data to {FRONTEND_DATA_PATH}")


if __name__ == "__main__":
    main()
