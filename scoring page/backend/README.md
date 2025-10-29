# Qwen Question Scorer – Backend

## Quick start
```bash
cd backend
cp .env.example .env
# Fill QWEN_API_KEY and optionally QWEN_COMPAT_BASE_URL
npm install
npm start
# Server on http://localhost:${PORT:-8787}
```

### Environment
- `QWEN_API_KEY`: sk-b3c27f7856344ee09bfd0723742d6dea
- `QWEN_COMPAT_BASE_URL`: OpenAI-compatible base URL for your Qwen endpoint (e.g., DashScope compatible mode).
- `QWEN_MODEL`: defaults to `qwen3-14b-instruct`.
- `PORT`: defaults to `8787`.

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
