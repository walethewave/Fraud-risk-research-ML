# What We're Building — Plain English

## The original problem

Your NIBSS fraud model (Random Forest) looks at a transaction and outputs
two numbers: `is_fraud: 1` and `probability: 0.83`. That's it. A fraud
analyst staring at that has no idea *why* the model flagged it — they'd
have to guess, or trust the model blindly. Neither is good enough for a
real bank (regulators require explanations; analysts won't act on a black
box).

## The fix, in three steps

**Step 1 — SHAP.** For every flagged transaction, we compute which
features pushed the prediction toward "fraud" and by how much. E.g.
"amount contributed +0.22, composite_risk contributed +0.14." This is
math, not English — a bar chart of numbers.

**Step 2 — Gemini turns the numbers into English.** We hand Gemini the
transaction details, the model's prediction, and those SHAP numbers, and
ask it to write what a fraud analyst would actually want to read:
*"Flagged primarily due to the transaction amount (₦6.2M), which is far
above typical. Risk level: High. Recommend: block and verify with
customer."* That's the "SLM" (small language model) half of the "ML+SLM"
system your research question is about.

**Step 3 — The frontend is how you *see* this working.** Everything above
happens in Python scripts (`explainability/*.py`). The frontend is just a
window onto the results — a webpage instead of reading raw JSON files.

## What the frontend actually is

**Not a chatbot.** It's a dashboard, like a fraud analyst's queue at a
bank:

1. **Home page** — a table of flagged transactions. Amount, channel,
   confidence score, risk badge (High/Medium/Low), and whether it was
   actually fraud or a false alarm. You scan this the way you'd scan an
   inbox.
2. **Click a row** — full detail page: the transaction, the model's
   prediction, a bar chart of the SHAP numbers, and the plain-English
   explanation Gemini wrote.
3. **"Ask a follow-up" box at the bottom** — this is the one chat-like
   piece. If the explanation isn't enough, you can type a question
   ("why didn't location matter here?") and get a live answer from
   Gemini, grounded in that specific case's data only (it's instructed
   not to make things up beyond what's on screen).

So: 95% dashboard, one small chat box bolted onto each case for follow-ups.
That mirrors how real fraud-review tools work — nobody wants to *chat*
their way through 100 transactions, they want to scan and drill in.

## How the pieces connect

```
Random Forest model (already trained)
        |
compute_shap.py        → outputs/shap_values.pkl   (the "why" numbers)
        |
generate_explanations.py → outputs/explanations.jsonl (Gemini's English version)
        |
export_frontend_data.py  → frontend/data/cases.json   (packaged for the website)
        |
frontend/ (Next.js website) → what you see in the browser
```

Each arrow is a script you (or I) run by hand — nothing here auto-updates.
If you generate more explanations later, you re-run
`export_frontend_data.py` to refresh what the website shows.

## Why there are only 12 cases right now

Google's Gemini free API key only allows ~20 requests per day, shared
across all its cheaper "Flash" models. We used most of that testing and
got 12 real transactions fully processed before hitting the wall. The
other 88 are queued up and will generate automatically (no rework needed)
once either: the daily quota resets, or you turn on billing for the
Gemini key (this would cost fractions of a cent for the remaining calls).

## Why this exists — the research angle

This is the basis for a short paper: does pairing the ML model's
prediction with an LLM-written explanation, grounded in real SHAP values,
actually help a fraud analyst more than the raw model output? The
`evaluate.py` script scores this two ways: does the explanation actually
mention the real risk factors (**faithfulness**), and does it contain the
four things an analyst needs — reason, factors, risk level, action
(**usefulness**). There's also a blind human-rating step
(`human_rating_template.csv`) for you to score explanations 1-5 by hand,
since automated metrics alone are a weak substitute for "would a real
analyst act on this."
