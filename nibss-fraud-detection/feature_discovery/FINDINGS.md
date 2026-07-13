# Feature Discovery — Findings

**Correction note:** the first version of this document reported an
uncorrected +0.13 AUC gain from two combined features and concluded "not
leakage" based only on a label-leakage check (does the feature directly
predict `is_fraud`). That check was insufficient — it missed *temporal*
leakage. This version corrects the record after a closer audit
(`leakage_check.py`). The corrected, validated finding is still real and
still substantial, just smaller and more carefully scoped than first
reported.

**Update — both errors fixed and the actual production notebooks
retrained** (not just the isolated `feature_discovery/` test scripts).
See "Production fix applied" near the bottom for the final, authoritative
numbers — they're better than either number above.

## The pipeline

1. `analyze_false_negatives.py` — compared 13 behavioral/velocity columns
   that exist in the raw NIBSS dataset but were dropped before modeling
   (per `METHODOLOGY.md` 3.3.5), checking whether missed-fraud cases (216
   false negatives in the test set) look statistically different from
   normal legitimate transactions on those columns.
2. `propose_features.py` — sent that statistics table to Gemini
   (single call, grounded strictly in the numbers) and asked it to
   recommend the 2 most promising candidates to test empirically. It
   picked `amount_sum_24h` and `amount_vs_mean_ratio`.
3. `test_feature_impact.py` — retrained the same Random Forest
   (identical hyperparameters, identical split) with vs. without the
   LLM's picks, to see if AUC actually moves.
4. `leakage_check.py` — **added after the AUC jump looked too large for
   the modest z-scores that motivated it.** Checked each candidate
   feature's source column against a real customer's transaction
   timeline to see whether it's a genuine backward-looking window or a
   lifetime aggregate that includes future transactions.

## What the leakage audit found

- `amount_mean_7d`, `amount_sum_24h`, `tx_count_24h` — verified as genuine
  look-back windows (they change transaction-by-transaction, reflecting
  only that customer's *past* activity within the window). **Clean.**
- `amount_mean_total` — verified as a **customer-lifetime constant**,
  identical across every transaction that customer ever made in the
  dataset, including ones that happen *after* the transaction being
  scored. In production you cannot know a customer's future transactions
  when scoring today's — this is look-ahead leakage. This directly
  contradicts `METHODOLOGY.md` section 3.2.4/3.5.4's claim that all
  aggregates use look-back-only windows; that claim is true for the 7d/24h
  windows but not for the lifetime `_total` columns.
- `amount_vs_mean_ratio = amount / amount_mean_total` **inherits this
  leakage** and is excluded from the validated recommendation.
- Notably: `composite_risk`, already one of the deployed model's 17
  features, correlates **0.54** with the leaky `amount_vs_mean_ratio` and
  **0.46** with the clean `amount_sum_24h`. So the *existing* deployed
  model already has some diluted exposure to this same leakage via
  `composite_risk` — this wasn't introduced by this experiment, it was
  already present in the original 0.822 AUC, just blended in rather than
  raw. Worth flagging as a limitation of the base model, not just this
  add-on experiment.

## Validated result (leakage-checked)

| Model | AUC | Recall @ 0.5 threshold | False negatives |
|---|---|---|---|
| Baseline (17 features) | 0.8207 | 0.707 | 176 |
| + `amount_sum_24h` only (clean) | **0.9164** | 0.695 | 183 |

- `amount_sum_24h` alone, verified as a genuine look-back feature, is
  **not** correlated with `is_fraud` in isolation (AUC ≈ 0.53 alone with
  no other features) — the gain comes from interaction with the existing
  17 features, primarily `amount`. Not a shortcut, not a proxy.
- The AUC gain (+0.096) is real and substantial. But **recall at the
  default 0.5 threshold slightly decreased** (176 → 183 false negatives)
  even though ranking quality (AUC) improved — meaning the gain shows up
  as better overall discrimination, not automatically as more fraud
  caught at the current threshold. Realizing it as fewer missed cases
  would require re-running the threshold optimization step from
  `METHODOLOGY.md` section 3.4.5/3.10.3 on the augmented model.
- `amount_vs_mean_ratio` is **excluded** from the recommendation pending
  the dataset's `amount_mean_total` column being regenerated as a proper
  rolling/look-back customer average instead of a lifetime constant.

## Verdict

**IMPROVED**, but smaller and more carefully scoped than first reported:
+0.096 AUC (not +0.13) from one verified-clean feature, with a threshold
re-tuning caveat on recall. The leakage audit itself is as much a finding
as the feature — it also raises a real question about the *existing*
deployed model's 0.822 AUC, since `composite_risk` already partially
encodes the same leaky signal.

## Caveat for the paper

Two lessons, not one:
1. Statistics-only grounding (z-scores) can't see interaction effects —
   the empirical AUC test is load-bearing, not optional (the LLM's own
   "Medium" confidence undersold the real impact of the clean feature).
2. A label-leakage check alone is not sufficient for tabular fraud data
   with engineered aggregate features — temporal leakage (does the
   feature use information from the future relative to the row being
   scored) must be checked separately, and it wasn't until prompted by
   external review. Any automated "LLM proposes → test AUC" feature
   discovery loop needs this as an explicit pipeline step, not an
   afterthought.

## Additional verification: is `amount_sum_24h` leaky too?

Asked directly — the earlier check only eyeballed one customer's
timeline. Verified rigorously by independently recomputing `amount_sum_24h`
from scratch for all 1,000,000 rows (per-customer rolling 24h sum using
only that customer's own timestamps and amounts, closed on both ends) and
comparing to the stored column: **99.6% exact match** (996,182 / 1,000,000
rows). The 0.4% mismatches go in *both* directions (stored sometimes
higher, sometimes lower than recomputed) — consistent with rounding/window-
boundary conventions, not systematic look-ahead (which would only ever
push the stored value *up*). `amount_sum_24h` is genuinely clean.

Also asked: why does a feature with near-zero raw correlation with
`is_fraud` (r ≈ −0.0019) move AUC by +0.10? Because correlation only
measures a *linear, single-column* relationship. `amount_sum_24h` alone
tells you almost nothing — but paired with `amount`, a tree can split on
"is this transaction large **relative to** what's already happened in the
last 24h," a conditional signal no univariate check can see. This is the
core reason the original feature-pruning step (62 → 17 features) discarded
it: pruning by individual correlation/importance is blind to interaction-
only value.

## Production fix applied

Both errors were fixed directly in the actual project notebooks (not just
`feature_discovery/`'s isolated test scripts) and the production model was
retrained:

- **`notebooks/01_eda_analysis.ipynb`**: added a leak-safe replacement for
  `amount_vs_mean_ratio` — a proper backward-only expanding mean per
  customer (sorted by timestamp, excluding the current transaction),
  saved as `amount_vs_mean_ratio_safe`. Added both `amount_sum_24h` and
  `amount_vs_mean_ratio_safe` to the final feature set (7 base → 9 base,
  + 10 bank one-hots = **19 total model inputs**, up from 17). Also
  corrected two stale markdown cells that documented a feature list that
  never actually matched what the code saved (a separate, smaller bug
  found while fixing this).
- **`notebooks/02_ml_modeling.ipynb`**: re-executed end-to-end on the
  corrected 19-feature dataset. Also fixed the "Winner summary" cell,
  which had hardcoded old AUC/recall numbers as literal text instead of
  computing them from the trained model — it's now fully dynamic so it
  can't go stale again on a future re-run.

### Final retrained numbers (real, from the executed notebooks)

| Model | AUC-ROC | Recall | Precision | Accuracy | Fraud caught | Missed |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7020 | 54.7% | 0.88% | 81.35% | 328 / 600 | 272 |
| Random Forest — **before this fix** (17 features) | 0.822 | 64.0% | 0.95% | 79.84% | 384 / 600 | 216 |
| Random Forest — **after this fix** (19 features) | **0.9227** | **67.5%** | **7.08%** | **97.25%** | **405 / 600** | **195** |

Compared to the pre-fix production model: **+0.10 AUC, +3.5 recall
points, precision up ~7.4x (0.95% → 7.08%), false positives down from
~40,098 to ~5,315.** This is better than either the uncorrected +0.13 or
the intermediate +0.096 numbers reported earlier in this document, because
the production retrain includes *both* fixed features together
(`amount_sum_24h` clean, `amount_vs_mean_ratio_safe` leak-fixed rather
than excluded), on the full pipeline rather than an isolated test.

**Actual feature importance in the retrained model** (Gini):
`amount_sum_24h` (0.359, now the single most important feature) >
`amount` (0.263) > `amount_vs_mean_ratio_safe` (0.190) > `composite_risk`
(0.084). The three new/relative-context features account for over 80% of
the model's decisions combined.

**What's still open:**
- Recall improved this time (unlike the isolated `amount_sum_24h`-only
  test, which saw recall dip slightly) — but threshold re-optimization
  (`METHODOLOGY.md` 3.4.5/3.10.3) still hasn't been re-run on this model,
  so there's likely more recall available at a better-tuned threshold.
- `composite_risk` is unchanged, inherited from the raw dataset, and was
  itself built using the original leaky `amount_vs_mean_ratio` — a small
  residual leakage exposure may remain there that this fix could not
  reach (its construction formula isn't available to us).
- `explainability/` (the SHAP + Gemini pipeline) was built against the
  old 17-feature model and needs to be re-run against this 19-feature one
  — feature importances have changed substantially.
