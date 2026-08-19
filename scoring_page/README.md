# Question Quality Scorer

- Frontend: vanilla HTML + Tailwind + Chart.js
- Backend: Node.js (Express) calling the OpenAI API (`gpt-4o-mini` by default)

> **Status:** standalone prototype. The canonical scoring implementation is the Python service
> in `grading_feature/backend/scoring_service.py`, and the rubric and ABI formulas are currently
> duplicated between the two. This component is slated for retirement — see task **A21** in
> `capstone_fall2026/PROJECT_PLAN.md`. Do not add features here.

## Setup

Put your own API key in `backend/.env` — copy `backend/.env.example` and fill it in.
**Never put a key in `run.sh` or any other committed file.**

## Run

```bash
./run.sh install
./run.sh start
```

## JSON Input

The extractor walks any JSON tree and collects items with:
```json
{"role":"user","content":"<text that looks like a question>"}
```
It is compatible with the OpenWebUI export formats (objects under `chat.history.messages`, etc.).

## Output

- Overall average score (0–14) and normalized 0–100
- Distribution bins: 0–3, 4–6, 7–10, 11–14
- Habit feedback synthesized from dimension averages
- Per-question table with all dimension scores
