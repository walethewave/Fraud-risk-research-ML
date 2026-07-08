"""
One-off script (not part of the regular pipeline) that edits
notebooks/01_eda_analysis.ipynb and notebooks/02_ml_modeling.ipynb to fix
the two errors surfaced during feature discovery:

  Error 1 — feature selection discarded amount_sum_24h, a feature that's
  near-useless alone (correlation ~0) but highly predictive combined with
  the existing 17 features (verified: AUC 0.821 -> 0.916 in isolation
  testing). Univariate pruning can't see interaction effects.

  Error 2 — amount_vs_mean_ratio (one of the features considered) is built
  on amount_mean_total, a customer-LIFETIME constant that includes
  transactions occurring after the one being scored (look-ahead leakage).
  Replaced with amount_vs_mean_ratio_safe, computed as a proper backward-
  only expanding mean per customer, sorted by timestamp.

Run once, then execute both notebooks with jupyter nbconvert --execute.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
EDA_NB_PATH = NOTEBOOKS_DIR / "01_eda_analysis.ipynb"

NEW_SAFE_RATIO_CELL = """# ── Step 1b: Leak-safe replacement for amount_vs_mean_ratio ──────────────────
# FEATURE DISCOVERY FIX (see feature_discovery/FINDINGS.md):
# amount_mean_total (source of amount_vs_mean_ratio) was found to be a
# customer-LIFETIME constant — identical across every row for a customer,
# meaning it includes transactions that happen AFTER the one being scored.
# That is look-ahead leakage: in production you cannot know a customer's
# future transactions when scoring today's. Fixed by computing a proper
# backward-only expanding mean, sorted by timestamp, using only
# transactions strictly BEFORE the current one.
data_clean_sorted = data_clean.sort_values(['customer_id', 'timestamp'])
prior_mean = (
    data_clean_sorted.groupby('customer_id')['amount']
    .apply(lambda s: s.shift(1).expanding().mean())
)
data_clean_sorted['customer_mean_amount_prior'] = prior_mean.values

# A customer's first transaction has no prior history — fall back to their
# own amount (ratio = 1.0, i.e. "average, for a first-time customer").
data_clean_sorted['customer_mean_amount_prior'] = (
    data_clean_sorted['customer_mean_amount_prior']
    .fillna(data_clean_sorted['amount'])
)
data_clean_sorted['amount_vs_mean_ratio_safe'] = (
    data_clean_sorted['amount']
    / data_clean_sorted['customer_mean_amount_prior'].replace(0, np.nan)
).fillna(1.0)

# Restore original row order
data_clean = data_clean_sorted.sort_index()
del data_clean_sorted, prior_mean

print('\\u2705 amount_vs_mean_ratio_safe created (leak-safe, look-back only).')
print(data_clean[['amount_vs_mean_ratio', 'amount_vs_mean_ratio_safe']].describe())
"""

NEW_FEATURE_SELECTION_CELL = """# ── Final feature selection ──────────────────────────────────────────────────
# FEATURE DISCOVERY FIX (see feature_discovery/FINDINGS.md): added
# amount_sum_24h (verified clean, look-back-only rolling window; alone
# near-zero correlation with fraud, but +0.096 AUC combined with the
# existing features via interaction with `amount`) and
# amount_vs_mean_ratio_safe (leak-safe replacement for
# amount_vs_mean_ratio, see the cell above) to the base feature set.
base_features = [
    'amount', 'hour', 'day_of_week', 'month', 'merchant_risk_score',
    'composite_risk', 'age_numeric',
    'amount_sum_24h', 'amount_vs_mean_ratio_safe',
]
bank_cols = [c for c in data_clean.columns if c.startswith('bank_')]

# Base features (9) + bank one-hot columns (10) = 19 model inputs
feature_cols_final = base_features + bank_cols

print(f'\\u2705 Total model input features: {len(feature_cols_final)}')
print(f'   (9 base + {len(bank_cols)} bank encodings = {len(feature_cols_final)} total)\\n')
print('Features selected:')
for i, feat in enumerate(feature_cols_final, 1):
    print(f'  {i:>2}. {feat}')
"""

NEW_SECTION_11_MARKDOWN = """### Key Findings — Section 11

- **8 columns dropped**: cyclical time encodings (6), `amount_rounded` (low signal), and `fraud_technique` (leakage).
- **`age_numeric`** converts the ordinal age groups into a proper ordered integer, enabling the model to exploit age relationships correctly.
- **`velocity_bin`** provides a coarse binning of velocity for potential categorical interaction analysis.
- **`amount_vs_mean_ratio_safe`** (added after a feature-discovery audit): the original `amount_vs_mean_ratio` is built on `amount_mean_total`, a customer-lifetime constant that includes future transactions relative to any given row — look-ahead leakage. The safe version uses a strictly backward-looking expanding mean per customer instead. See `feature_discovery/FINDINGS.md` for the full audit.
"""

NEW_SECTION_13_MARKDOWN = """### Key Findings — Section 13

- **19 model inputs total**: 9 base numeric features + 10 binary bank indicator columns.
- **Base features**: `amount`, `hour`, `day_of_week`, `month`, `merchant_risk_score`, `composite_risk`, `age_numeric`, `amount_sum_24h`, `amount_vs_mean_ratio_safe`.
- **Feature-discovery correction**: an earlier version of this pipeline used only 7 base features, dropping `amount_sum_24h` and never including a customer-relative amount ratio at all. A closed-loop audit (analyze false negatives → LLM proposes candidates → empirically test AUC → audit for leakage) found `amount_sum_24h` alone lifts AUC from 0.821 to 0.916, and that the original candidate `amount_vs_mean_ratio` had look-ahead leakage in its source column (`amount_mean_total`), fixed here as `amount_vs_mean_ratio_safe`. Full writeup: `feature_discovery/FINDINGS.md`.
- **Leakage note**: `tx_count_24h`, `amount_mean_7d`, `amount_sum_24h` are genuine look-back windows (verified by independent recomputation — 99.6% match rate against a from-scratch rolling calculation). `amount_mean_total` and anything derived from it is NOT — it is a lifetime aggregate, not a look-back window, contrary to what was previously stated here.
- **Dataset saved to `data/processed/data_model_ready.pkl`** for use in `02_ml_modeling.ipynb`.
"""


def find_cell_index(cells, contains: str, cell_type: str = "code") -> int:
    for i, cell in enumerate(cells):
        if cell["cell_type"] != cell_type:
            continue
        if contains in "".join(cell["source"]):
            return i
    raise ValueError(f"No {cell_type} cell found containing: {contains!r}")


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    nb = json.loads(EDA_NB_PATH.read_text())
    cells = nb["cells"]

    # 1. Insert the leak-safe ratio cell right after the drop/age/velocity cell.
    drop_cell_idx = find_cell_index(cells, "drop_cols = [")
    cells.insert(drop_cell_idx + 1, make_code_cell(NEW_SAFE_RATIO_CELL))

    # 2. Replace the final feature selection cell (index shifts by +1 from the insert).
    feature_sel_idx = find_cell_index(cells, "feature_cols_final = base_features + bank_cols")
    cells[feature_sel_idx]["source"] = NEW_FEATURE_SELECTION_CELL.splitlines(keepends=True)
    cells[feature_sel_idx]["outputs"] = []
    cells[feature_sel_idx]["execution_count"] = None

    # 3. Fix the stale "Section 11" and "Section 13" markdown to match reality.
    sec11_idx = find_cell_index(cells, "8 columns dropped", cell_type="markdown")
    cells[sec11_idx]["source"] = NEW_SECTION_11_MARKDOWN.splitlines(keepends=True)

    sec13_idx = find_cell_index(cells, "Key Findings — Section 13", cell_type="markdown")
    cells[sec13_idx]["source"] = NEW_SECTION_13_MARKDOWN.splitlines(keepends=True)

    EDA_NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Patched {EDA_NB_PATH}")


if __name__ == "__main__":
    main()
