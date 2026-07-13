# NIBSS Fraud Review — Frontend

A dark, glass-styled triage dashboard for the ML + SLM fraud explanation
system in `../explainability/`. Not a chatbot — it's a
case queue (like a fraud analyst's real workflow): scan flagged transactions,
click into one to see the Random Forest prediction, the top-3 SHAP risk
factors, and a Gemini-narrated plain-English explanation, then optionally
ask a grounded follow-up question in the chat panel.

This directory lives at `nibss-fraud-detection/frontend/`, alongside the
`explainability/` pipeline it visualizes.

## Data source

`data/cases.json` is a static export — it does **not** call the Python
model at runtime. It's generated from
`../explainability/outputs/explanations.jsonl` by:

```bash
cd ..   # nibss-fraud-detection/
source .venv/bin/activate
python3 explainability/export_frontend_data.py
```

Re-run that any time `generate_explanations.py` produces more cases (e.g.
after Gemini's free-tier daily quota resets — see the parent project's
`design.md` for why the sample is currently 12/100 cases).

## Live follow-up chat

`app/api/chat/route.ts` calls Gemini server-side, grounded in the selected
case's transaction details, SHAP factors, and original explanation — it's
instructed not to invent details beyond what's listed (the same fix applied
to the Python prompt template after the Week 3 manual test caught
hallucination on a false-positive case).

Requires `GEMINI_API_KEY` in `.env.local` (gitignored, not committed).

## Model fallback

Both the chat route and the Python generation script fall back through
`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.5-flash-lite` on quota
exhaustion or transient errors. Google's free tier appears to share a
single daily request quota across models on one API key/project, not a
per-model quota — expect the chat to occasionally return a 503 if the
day's quota is already spent by the batch generation script.

## Local development

```bash
npm install
npm run dev   # http://localhost:3000
```

## Deploy to Vercel

1. Push this repo (or just the `frontend/` directory as its own Vercel
   project root) to GitHub.
2. In Vercel: **New Project** → import the repo → set **Root Directory**
   to `frontend`.
3. Add environment variable `GEMINI_API_KEY` in the Vercel project
   settings (Production + Preview).
4. Deploy. `data/cases.json` is committed and bundled at build time, so
   no database or extra backend is needed for the dashboard itself — only
   the chat route needs the live API key.

To update the demo data after a Vercel deploy, regenerate
`data/cases.json` locally (see above), commit it, and push — Vercel will
redeploy automatically.
