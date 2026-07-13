# Week 3 — Manual Prompt Test Notes

Two real test-set cases were run by hand through Gemini 2.5 Flash before
automating anything, per design.md / the plan.

## Case 1: True positive (row 478793)
₦6,188,809 IB transfer, FirstBank, 3 PM, 40+ age group. Model: FRAUD, 83%
confidence. Top SHAP factors: amount (+0.224), composite_risk (+0.145),
merchant_risk_score (-0.020).

First-pass output was good: correctly identified amount as primary driver,
stated High risk, recommended block + verification. No issues.

## Case 2: False positive (row 412686)
₦170,841 IB transfer, Wema, 6 AM, 40+ age group. Model: FRAUD (incorrect),
50% confidence. Top SHAP factors: amount (+0.066), merchant_risk_score
(-0.044), month (+0.036).

**Problem found in first-pass output:** the model invented "early morning
hour (6 AM)" as a supporting risk factor — `hour` was never in the top-3
SHAP list for this row. It also glossed over `merchant_risk_score`, a
risk-*reducing* factor, without saying so.

**Fix applied to prompt_builder.py:**
1. Added an explicit instruction: "using ONLY the factors listed above — do
   not invent or mention any other transaction detail as a risk factor."
2. Added a sentence clarifying that negative contributions reduce risk
   (push toward legitimate), not add to it.

**Re-test after fix:** correctly cited only amount, month, and
merchant_risk_score, and explicitly described merchant_risk_score as having
"partially mitigated the overall risk." Prompt template frozen at this
version.

**Takeaway for the paper:** this is itself a finding worth reporting —
without an explicit "don't invent factors" instruction, an LLM narrating
SHAP values will hallucinate plausible-sounding but unsupported risk
factors, especially on borderline (~50% confidence) cases. Faithfulness
scoring in evaluate.py should be read with this in mind: a high faithfulness
score confirms the top-3 SHAP features are mentioned, but does not by
itself rule out additional hallucinated factors being mentioned alongside
them. A stricter check (does the explanation mention any feature NOT in the
top-3) would be a natural extension of the faithfulness metric.
