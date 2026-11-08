from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Any
from logger import logging
from extract_chats import main as extract_data
from scoring_service import score_questions, OPENAI_MODEL, MOCK_SCORER
import uvicorn
import json
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("professor_dashboard")

DATA_PATH = os.getenv("DATA_PATH")
# Built frontend (Vite dist/). When set, this service also serves the dashboard itself.
STATIC_DIR = os.getenv("STATIC_DIR")
# Comma-separated origins allowed to call the API cross-origin. Unset by default: the hosted
# build and the Vite dev proxy are both same-origin, so CORS is only needed when the frontend
# is built with VITE_API_BASE_URL pointing here from somewhere else.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title = "Professor Dashboard")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def load_data():
    try:
        with open(DATA_PATH, 'r') as f:
           return json.load(f)
    except Exception as e:
        logger.error("DB not found, please refresh")   
        return []


class ScoreRequest(BaseModel):
    questions: list[Any]
    useAbi: bool = False


def is_question_like(text: str) -> bool:
    if not isinstance(text, str):
        return False
    trimmed = text.strip()
    if not trimmed:
        return False

    lower = trimmed.lower()
    starters = [
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "would ",
        "will ",
        "should ",
    ]
    return "?" in trimmed or any(lower.startswith(prefix) for prefix in starters)


def extract_questions_for_user(user: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen = set()

    chats = user.get("chats") or []
    for chat_idx, chat in enumerate(chats):
        pairs = chat.get("message_pairs") or []
        for pair_idx, pair in enumerate(pairs):
            question_text = (pair.get("question") or "").strip()
            if not is_question_like(question_text):
                continue

            dedupe_key = f"{user.get('user_id')}::{question_text}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            questions.append(
                {
                    "id": f"{user.get('user_id')}-{chat_idx}-{pair_idx}",
                    "text": question_text,
                    "studentId": user.get("user_id"),
                    "studentName": user.get("name") or user.get("email") or "Unknown",
                }
            )
    return questions


def extract_questions_for_users(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for user in users:
        if isinstance(user, dict):
            questions.extend(extract_questions_for_user(user))
    return questions


@app.get("/api/health")
def score_health():
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "mock": MOCK_SCORER,
    }


@app.post("/api/score")
async def score_endpoint(payload: ScoreRequest):
    if not payload.questions:
        raise HTTPException(status_code=400, detail={"error": "questions must be a non-empty array"})

    try:
        results, aggregate = await score_questions(payload.questions, payload.useAbi)
        return {
            "ok": True,
            "results": results,
            "aggregate": aggregate,
        }
    except Exception as e:
        logger.exception(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/student-feedback")
async def student_feedback_dashboard(user_id: str = "all", useAbi: bool = True):
    data = load_data()
    normalized_user_id = (user_id or "all").strip()

    if normalized_user_id.lower() in {"", "all", "*"}:
        selected_users = [user for user in data if isinstance(user, dict)]
        selected_user_summary = {
            "user_id": "all",
            "name": "All users",
            "email": None,
        }
        scope = "all"
    else:
        selected_user = next((u for u in data if u.get("user_id") == normalized_user_id), None)
        if not selected_user:
            raise HTTPException(status_code=404, detail={"error": "User not found"})

        selected_users = [selected_user]
        selected_user_summary = {
            "user_id": selected_user.get("user_id"),
            "name": selected_user.get("name"),
            "email": selected_user.get("email"),
        }
        scope = "user"

    questions = extract_questions_for_users(selected_users)
    if not questions:
        return {
            "ok": True,
            "scope": scope,
            "selected_user_count": len(selected_users),
            "user": selected_user_summary,
            "results": [],
            "aggregate": {
                "count": 0,
                "avg_total_0_14": 0,
                "overall_0_100": 0,
                "dims": {},
                "distribution": {"labels": ["0-3", "4-6", "7-10", "11-14"], "counts": [0, 0, 0, 0]},
                "habits": ["No question-like prompts were detected yet."],
                "perStudent": {},
                "abi_global": None,
            },
        }

    try:
        results, aggregate = await score_questions(questions, useAbi)
        return {
            "ok": True,
            "scope": scope,
            "selected_user_count": len(selected_users),
            "user": selected_user_summary,
            "results": results,
            "aggregate": aggregate,
        }
    except Exception as e:
        logger.exception(f"Student feedback scoring failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/student-feedback/{user_id}")
async def student_feedback(user_id: str, useAbi: bool = True):
    return await student_feedback_dashboard(user_id=user_id, useAbi=useAbi)

@app.get("/api/users")
def get_all_users():
    return load_data()

@app.get("/api/user/{user_id}")
def get_user(user_id):
    data = load_data()

    for user in data:
        if user['user_id'] == user_id:
            return user
    return {"error": "User not found"}

@app.get("/api/refresh")
def run_extract():
    try:
        metadata = extract_data()
        logger.info("Extract completed successfully.")
        return {
            "status": "success",
            "message": "Refresh completed successfully",
            "refresh_metadata": metadata,
        }
    except Exception as e:
        logger.exception(f"Extract failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Refresh failed",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


# Registered last so every /api route above takes precedence.
if STATIC_DIR:
    static_root = Path(STATIC_DIR).resolve()
    index_file = static_root / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail={"error": "Not found"})

        candidate = (static_root / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(static_root):
            return FileResponse(candidate)
        # Anything else is a client-side route (e.g. /user/<id>); the React router handles it.
        return FileResponse(index_file)