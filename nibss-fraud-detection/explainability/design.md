# Design: ML + SLM Fraud Explanation Layer

## Research Question

Does an ML + SLM hybrid system produce more actionable fraud explanations
than raw model output in the Nigerian banking context?

## Architecture

```
Transaction features (amount, channel, hour, bank, location, age_group,
merchant_risk_score, composite_risk)
        |
        v
Random Forest (existing, trained model)
Output: is_fraud (0/1), fraud_probability (0.0-1.0)
        |
        v
SHAP TreeExplainer (pre-computed offline, not real-time)
Output: per-feature contribution to the fraud_probability
        |
        v
Prompt builder
Combines: transaction features + prediction + top-3 SHAP contributions
        |
        v
Gemini 1.5/2.0 Flash
        |
        v
Analyst-facing explanation
```

Latency note: SHAP is not run at inference time. It is computed once,
offline, over the saved test set, and the pre-computed values are reused
by the prompt builder. This is a stated scope boundary for a research
prototype, not a gap to solve here.

## Prompt Input Schema

- `transaction_amount`
- `channel`
- `hour`
- `bank`
- `location`
- `fraud_probability`
- `is_fraud` (model prediction)
- `top_3_shap_features` — list of (feature_name, shap_value) tuples

## Required SLM Output Fields

1. Primary reason for flagging (1 sentence)
2. Supporting factors (2-3 bullet points)
3. Risk level (High / Medium / Low)
4. Recommended action (block / step-up auth / monitor)

## Evaluation Metrics

### Faithfulness (automated)
Does the explanation text mention the top-3 SHAP features (by name or a
recognizable plain-English paraphrase, e.g. "amount" -> "transaction size")?
Score 1 if yes, 0 if no, per feature; average across the top-3 and across
the test set.

### Usefulness (automated)
Does the explanation contain all four required output fields (reason,
factors, risk level, action)? Score 0-3 per explanation (one point per
missing-vs-present section beyond the first), averaged across the set.

### Human rating (manual add-on)
- **Scale:** 1-5, "Would you act on this explanation as a fraud analyst?"
  (1 = not useful/misleading, 5 = immediately actionable)
- **Sample size:** 50 cases, drawn to include both true positives and
  false positives so raters see both a correct flag and a false alarm
  explained.
- **Blinding:** Rater sees only the transaction summary and the generated
  explanation text — not the SHAP values, not whether the flag was
  correct, not which model/prompt version produced it (if multiple
  variants are ever compared).
- **Output:** mean and variance of the 1-5 score across the 50 rated
  cases, reported alongside the automated metrics.

## SLM Configuration

- **Model:** Gemini 2.5 Flash via the `google-genai` SDK (the successor to
  the now-deprecated `google-generativeai` package).
- **Interface:** single function `explain_prompt(prompt) -> str` on
  `LLMClient`, kept provider-agnostic so an alternate LLM (e.g. Claude) can
  be swapped in later without changing callers.

## Sample Size (Amendment, Week 4)

The free tier for `gemini-2.5-flash` is rate-limited to 5 requests/minute.
Running all 600 fraud cases + a 100-case false-positive contrast sample
(700 total) at that pace takes ~2.5 hours. By explicit decision, Week 4's
run instead samples **50 fraud cases + 50 false positives (100 total,
random_state=42)** — large enough to produce meaningful faithfulness/
usefulness averages and to fully cover the 50-case human-rating sample,
while finishing in ~20 minutes. This is a stated scope limitation for the
paper (free-tier API constraint), not a methodological choice, and should
be reported as such in the Limitations section alongside the SHAP-latency
note above.

## Prompt Iteration Note (Week 3 finding)

The first version of the prompt template, tested by hand on a real
false-positive case (row 412686, see `manual_prompt_test.md`), caused
Gemini to invent a plausible-sounding risk factor ("early morning hour")
that was not actually in the top-3 SHAP list, and to describe a negative
(risk-reducing) SHAP contribution as if it were risk-increasing. The
template was tightened to (1) explicitly instruct the model to use *only*
the listed factors and (2) clarify that negative contributions reduce
risk. This fixed both issues on re-test and is itself a reportable finding
about LLM-narrated SHAP explanations: without explicit grounding
instructions, the model will hallucinate supporting evidence, especially
on borderline-confidence cases.
