"""
One-off fix: cell 41 of 02_ml_modeling.ipynb ("Winner summary") had the
AUC/recall comparison partially hardcoded as literal text, which went
stale the moment the feature set changed. Rewrites it to be fully dynamic
so it self-corrects on any future re-run, and drops the now-inaccurate
"achieved with zero data leakage" blanket claim (true for the new
amount_sum_24h / amount_vs_mean_ratio_safe features, but composite_risk —
inherited unmodified from the raw dataset — was itself built upstream
using the original leaky ratio, so a residual caveat is honest).
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_ml_modeling.ipynb"

NEW_WINNER_CELL = """# ── Winner summary ────────────────────────────────────────────────────────────
auc_relative_improvement = (rf_auc / lr_auc - 1) * 100
recall_pp_improvement = (rf_recall - lr_recall) * 100

print('='*65)
print('  WINNER: RANDOM FOREST')
print('='*65)
print(f'\\n  Random Forest outperforms Logistic Regression across all')
print(f'  primary fraud detection metrics:')
print(f'')
print(f'  \\u2022 AUC-ROC:  {rf_auc:.4f} vs {lr_auc:.4f}  (+{auc_relative_improvement:.1f}% relative improvement)')
print(f'  \\u2022 Recall:   {rf_recall*100:.1f}% vs {lr_recall*100:.1f}%  (+{recall_pp_improvement:.1f} percentage points)')
print(f'  \\u2022 Fraud caught: {tp_rf} vs {tp_lr} ({tp_rf - tp_lr} additional fraud cases detected)')
print(f'  \\u2022 Missed fraud: {fn_rf} vs {fn_lr} ({fn_lr - fn_rf} fewer cases escaping detection)')
print(f'')
print(f'  Justification:')
print(f'  The non-linear decision boundaries of the Random Forest ensemble')
print(f'  are better suited to the multi-feature interaction patterns')
print(f'  documented in the EDA (e.g. high-amount + night-hour + Mobile).')
print(f'')
print(f'  Feature-discovery note: this AUC reflects a corrected 19-feature')
print(f'  set (see feature_discovery/FINDINGS.md) after two fixes: adding')
print(f'  amount_sum_24h (verified leak-free, +0.096 AUC alone) and')
print(f'  replacing amount_vs_mean_ratio with a leak-safe backward-only')
print(f'  version (the original was built on a customer-lifetime constant')
print(f'  that included future transactions). composite_risk, inherited')
print(f'  unmodified from the raw dataset, was itself constructed upstream')
print(f'  using the original leaky ratio, so a small residual exposure to')
print(f'  that issue may remain via composite_risk specifically.')
print('='*65)

# \\u2500\\u2500 Save the winning model \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500
joblib.dump(rf_model, f'{MODEL_PATH}random_forest_fraud_model.pkl')
joblib.dump(lr_model, f'{MODEL_PATH}logistic_regression_fraud_model.pkl')
print(f'\\n\\u2705 Models saved to {MODEL_PATH}')
"""


def main():
    nb = json.loads(NB_PATH.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "WINNER: RANDOM FOREST" in "".join(cell["source"]):
            cell["source"] = NEW_WINNER_CELL.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Patched cell {i}")
            break
    else:
        raise ValueError("Winner cell not found")
    NB_PATH.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
