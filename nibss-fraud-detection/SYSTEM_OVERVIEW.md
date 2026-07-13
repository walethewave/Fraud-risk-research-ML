# What This System Actually Is — Frontend + Backend, Live vs. Static

You've been looking at the deployed site and asking "why is this static, where's
the live part" — fair question, because three different pages behave three
different ways and it's not obvious from just clicking around. This doc
explains all of it, plainly.

## The two backends (Python, not part of the website)

Everything under `explainability/` and `feature_discovery/` is **offline
Python** — scripts you or I run by hand in a terminal. None of it runs when
someone visits the website. Think of it as a factory that produces JSON files,
which the website then displays.

**`explainability/` — turns model predictions into English**
1. `compute_shap.py` — loads the trained Random Forest model, runs it against
   200,000 test transactions, and for each one calculates *why* it got its
   score (SHAP values — a number per feature, e.g. "amount contributed +0.22").
2. `generate_explanations.py` — takes those SHAP numbers plus the transaction
   details, builds a prompt, and asks Gemini to write a plain-English
   explanation. Saves everything to `outputs/explanations.jsonl`.
3. `export_frontend_data.py` — packages that into `frontend/data/cases.json`,
   the file the website actually reads.

**`feature_discovery/` — checks if the model can be improved**
1. `analyze_false_negatives.py` — finds transactions the model got wrong
   (missed fraud) and checks if any unused data columns look different for
   those cases.
2. `propose_features.py` — one Gemini call: "here are the patterns, which
   columns should we try adding?"
3. `leakage_check.py` — before trusting any of that, verifies the proposed
   columns aren't secretly cheating (using future information).
4. The actual notebooks (`notebooks/01_eda_analysis.ipynb`,
   `02_ml_modeling.ipynb`) get edited and re-run to retrain the real model
   with the fix.
5. `export_frontend_data.py` — packages the before/after results into
   `frontend/data/discovery.json`.

**Bottom line: both of these produce files. The website reads the files. That's it.**

## The frontend (Next.js website — what you're clicking around in)

Three pages, three different behaviors:

| Page | URL | Behavior |
|---|---|---|
| Dashboard | `/` | **Static.** Reads `cases.json`, shows a table. No live computation. |
| Feature Discovery | `/discovery` | **Static.** Reads `discovery.json`, shows the before/after audit. No live computation. |
| Case detail | `/case/426596` (any id) | **Mostly static** (transaction details, SHAP chart, and the written explanation all come from `cases.json`) **— except the "Ask a follow-up" box at the bottom, which is genuinely live.** |

### The one live thing: the chat box on a case page

When you type a question into "Ask a follow-up" and hit send, your browser
calls `/api/chat` — a real server function (`app/api/chat/route.ts`) that:
1. Looks up that specific case's transaction data and SHAP values
2. Builds a prompt telling Gemini "only use these facts, don't invent anything"
3. Calls Gemini live, right then, and streams the answer back to your screen

This is the only part of the whole system that runs in real time when you use
it. Everything else (the table, the risk badges, the SHAP bars, the written
explanation you see before you even ask a question) was pre-computed by the
Python scripts above and baked into the deployed site.

## Why it's built this way (not one big live system)

Running the actual Random Forest + SHAP computation live, per click, would be
slow (SHAP took ~6 minutes for 200,000 rows) and would need a Python backend
hosted somewhere, not just a Next.js app. For a research prototype, the
practical split is:
- **Expensive, slow stuff** (model inference, SHAP, generating the written
  explanation) → done once, offline, results saved.
- **Cheap, interactive stuff** (answering a follow-up question about a case
  that's already been analyzed) → done live, since it's fast and benefits from
  being interactive.

## Why the dashboard only shows "Actual fraud" cases right now

`generate_explanations.py` processes 50 fraud cases first, then 50
false-positive cases, in that order. We're currently at 38/100 completed
(rate-limited by Gemini's free-tier daily quota), so it hasn't reached the
false-positive batch yet. Nothing broken — once more of the 100 finish
generating (or the quota resets and it's re-run), false positives will start
appearing in the table too.
