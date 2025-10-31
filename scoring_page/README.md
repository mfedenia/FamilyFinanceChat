# Question Quality Scorer (Qwen3‑14B‑Instruct)

- Frontend: vanilla HTML + Tailwind + Chart.js
- Backend: Node.js (Express) calling Qwen 3‑14B Instruct via an OpenAI‑compatible endpoint

## Run
1. Backend
```bash
cd backend
cp .env.example .env
# Fill QWEN_API_KEY and optionally QWEN_COMPAT_BASE_URL
npm install
npm start
```
2. Frontend
Open `frontend/index.html` in your browser.

## JSON Input
The extractor walks any JSON tree and collects items with:
```json
{"role":"user","content":"<text that looks like a question>"}
```
It is compatible with the screenshot formats you shared (objects under `chat.history.messages` etc.).

## Output
- Overall average score (0–14) and normalized 0–100
- Distribution bins: 0–3, 4–6, 7–10, 11–14
- Habit feedback synthesized from dimension averages
- Per‑question table with all dimension scores
