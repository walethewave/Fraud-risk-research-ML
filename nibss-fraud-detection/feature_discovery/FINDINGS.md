# Feature Discovery — Findings

## The pipeline

1. `analyze_false_negatives.py` — compared 13 behavioral/velocity columns
   that exist in the raw NIBSS dataset but were dropped before modeling
   (per `METHODOLOGY.md` 3.3.5), checking whether missed-fraud cases (216
   false negatives in the test set) look statistically different from
   normal legitimate transactions on those columns.
2. `propose_features.py` — sent that statistics table to Gemini
   (single call, grounded strictly in the numbers) and asked it to
   recommend the 2 most promising candidates to test empirically.
3. `test_feature_impact.py` — retrained the same Random Forest
   (identical hyperparameters, identical split) with vs. without the
   LLM's picks, to see if AUC actually moves.

## What the LLM picked, and why

The FN-vs-legitimate z-scores were all modest (max |z| = 0.21 — the LLM
was explicitly instructed to flag this and rated its own confidence
"Medium," not "High," which was the right call given the data). It
recommended:

- **`amount_sum_24h`** — 24-hour rolling transaction sum
- **`amount_vs_mean_ratio`** — current amount ÷ customer's historical
  average amount

## What actually happened when tested

| Model | AUC | Recall | False negatives |
|---|---|---|---|
| Baseline (17 features) | 0.821 | 0.707 | 176 |
| + `amount_sum_24h` only | 0.916 | — | — |
| + `amount_vs_mean_ratio` only | 0.931 | — | — |
| + both | **0.951** | 0.745 | 153 |

That's a real +0.13 AUC gain, not noise — 87 previously-missed fraud cases
get caught (against 64 previously-caught cases now missed, net positive).

## Why the gain is this large (and why it isn't leakage)

Checked before trusting this number, because a jump this size from z-scores
this small is the signature of either a bug or label leakage:

- Each candidate feature **alone**, with no other features, only reaches
  AUC 0.53 (barely above random) — so there's no trivial univariate
  shortcut to the label.
- Raw linear correlation with `is_fraud` is ~0 (−0.002) for both — so it's
  not a simple leaked proxy either.
- The gain only appears when combined with the *existing* 17 features,
  specifically `amount`.

The actual explanation: **the baseline model has no access to any
customer-history baseline at all.** Its 17 features describe the
transaction in isolation (amount, time, bank, merchant risk) plus
`composite_risk`, which — per `METHODOLOGY.md` — is itself only a
*diluted, weighted blend* of velocity/amount-ratio signals, not the raw
ratio. `amount_vs_mean_ratio` answers a question the model literally could
not ask before: **"is this transaction unusual for this specific
customer?"** — which is exactly the design philosophy already stated in
the top-level README ("not 'is this transaction big?' but 'is this bigger
than what this user normally does?'"). The original feature-selection step
(originally 62 features pruned to 17) discarded this signal in favor of
its lossy compressed form.

## Verdict

**IMPROVED**, and the mechanism is understood and defensible, not a
leakage artifact. This is a genuine, reportable finding for the paper's
"LLM-assisted feature discovery" claim — the closed loop (missed-fraud
analysis → LLM proposal → empirical AUC test) surfaced a real gap in the
original feature-selection step, not just a marginal tweak.

## Caveat for the paper

The LLM's own stated confidence ("Medium," based on z-scores alone) was an
*underestimate* of the actual impact — the z-score analysis (marginal,
univariate) couldn't see the interaction effect that made this valuable.
This itself is worth noting: statistics-only grounding limits what an LLM
can correctly judge about feature usefulness in tree ensembles; the
empirical AUC test step is not optional, it's load-bearing.
