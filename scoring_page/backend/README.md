# Question Scorer – Backend

Node.js (Express) service that scores student questions against the seven-dimension rubric.

> **Status:** standalone prototype. The canonical scoring implementation is the Python service
> in `grading_feature/backend/scoring_service.py`. This component is slated for retirement —
> see task **A21** in `capstone_fall2026/PROJECT_PLAN.md`. Do not add features here.

## Quick start

```bash
cd backend
cp .env.example .env       # then put your own OPENAI_API_KEY in .env
npm install
npm start
# Server on http://localhost:${PORT:-8787}
```

### Environment

Set these in `backend/.env`. **Never commit real values.**

- `OPENAI_API_KEY` — required, unless `MOCK_SCORER=1`.
- `OPENAI_MODEL` — defaults to `gpt-4o-mini`.
- `OPENAI_BASE_URL` — optional; defaults to `https://api.openai.com/v1`. Set it only when
  pointing at an OpenAI-compatible endpoint from another provider.
- `MOCK_SCORER` — optional; set to `1` to score with a deterministic stub and make no API calls.
- `PORT` — defaults to `8787`.

### API

`POST /api/score`
```json
{
  "questions": ["What is your name?", "Did you change employers in 2024?"]
}
```
Response:
```json
{
  "results": [ { "... per-question scores ..." } ],
  "aggregate": {
    "count": 2,
    "avg_total_0_14": 12.5,
    "overall_0_100": 89,
    "distribution": {"0-3":0,"4-6":0,"7-10":1,"11-14":1},
    "per_dimension_avg": { "relevance":1.5, "...":1.8 },
    "habit_feedback": ["Politeness is consistently strong.", "..."]
  }
}
```
