from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from logger import logging
from extract_chats import main as extract_data
from scoring_service import (
    score_questions,
    OPENAI_MODEL,
    MOCK_SCORER,
    init_score_cache,
    get_cache_stats,
    clear_score_cache,
)
import uvicorn
import json
import os
import uuid
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

logger = logging.getLogger("professor_dashboard")

DATA_PATH = os.getenv("DATA_PATH")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_score_cache()
    yield


app = FastAPI(title="Professor Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RefreshJobManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.current_job: dict[str, Any] = {
            "job_id": None,
            "status": "idle",  # "idle", "in_progress", "completed", "failed"
            "stage": "idle",   # "idle", "extracting", "scoring", "completed", "failed"
            "progress": 0,     # 0 to 100
            "message": "Ready",
            "started_at": None,
            "completed_at": None,
            "error": None,
            "refresh_metadata": None,
            "scoring_stats": None,
        }

    def get_status(self) -> dict[str, Any]:
        return dict(self.current_job)

    def is_in_flight(self) -> bool:
        return self.current_job.get("status") == "in_progress"

    async def start_job(self) -> dict[str, Any]:
        async with self._lock:
            if self.is_in_flight():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "status": "in_progress",
                        "message": "A refresh job is already in flight",
                        "job": self.get_status(),
                    },
                )

            job_id = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            self.current_job = {
                "job_id": job_id,
                "status": "in_progress",
                "stage": "extracting",
                "progress": 5,
                "message": "Starting chat extraction...",
                "started_at": now_iso,
                "completed_at": None,
                "error": None,
                "refresh_metadata": None,
                "scoring_stats": None,
            }
            asyncio.create_task(self._run_job(job_id))
            return dict(self.current_job)

    async def _run_job(self, job_id: str):
        try:
            self.current_job["stage"] = "extracting"
            self.current_job["progress"] = 15
            self.current_job["message"] = "Extracting chats from OpenWebUI..."

            # Run extraction in worker thread
            metadata = await asyncio.to_thread(extract_data)
            self.current_job["refresh_metadata"] = metadata
            self.current_job["progress"] = 45
            self.current_job["stage"] = "scoring"
            self.current_job["message"] = "Extraction complete. Scoring student questions..."

            # Load the extracted data to find questions
            data = load_data()
            users = [u for u in data if isinstance(u, dict)]
            questions = extract_questions_for_users(users)

            total_q = len(questions)
            if total_q == 0:
                self.current_job["progress"] = 100
                self.current_job["stage"] = "completed"
                self.current_job["status"] = "completed"
                self.current_job["message"] = "Refresh completed successfully (no questions to score)."
                self.current_job["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.current_job["scoring_stats"] = {"total_questions": 0, "scored": 0}
                return

            def on_scoring_progress(completed: int, total: int):
                pct = 45 + int((completed / total) * 50)
                self.current_job["progress"] = min(pct, 98)
                self.current_job["message"] = f"Scoring questions ({completed}/{total})..."

            scored, aggregate = await score_questions(
                questions,
                use_abi=True,
                progress_callback=on_scoring_progress,
            )

            self.current_job["progress"] = 100
            self.current_job["stage"] = "completed"
            self.current_job["status"] = "completed"
            self.current_job["message"] = "Refresh completed successfully."
            self.current_job["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.current_job["scoring_stats"] = {
                "total_questions": total_q,
                "scored": len(scored),
                "aggregate_overall": aggregate.get("overall_0_100") if aggregate else None,
            }
            logger.info("Refresh job %s completed successfully.", job_id)
        except Exception as e:
            logger.exception("Refresh job %s failed: %s", job_id, e)
            self.current_job["status"] = "failed"
            self.current_job["stage"] = "failed"
            self.current_job["progress"] = 100
            self.current_job["error"] = str(e)
            self.current_job["message"] = f"Refresh failed: {e}"
            self.current_job["completed_at"] = datetime.now(timezone.utc).isoformat()


refresh_manager = RefreshJobManager()


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
    bypassCache: bool = False


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
        results, aggregate = await score_questions(payload.questions, payload.useAbi, bypass_cache=payload.bypassCache)
        return {
            "ok": True,
            "results": results,
            "aggregate": aggregate,
        }
    except Exception as e:
        logger.exception("Scoring failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/cache/stats")
def cache_stats_endpoint():
    return {
        "ok": True,
        "cache": get_cache_stats(),
    }


@app.post("/api/cache/clear")
def cache_clear_endpoint():
    clear_score_cache()
    return {
        "ok": True,
        "message": "Score cache cleared successfully",
    }


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
        logger.exception("Student feedback scoring failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/student-feedback/{user_id}")
async def student_feedback(user_id: str, useAbi: bool = True):
    return await student_feedback_dashboard(user_id=user_id, useAbi=useAbi)


@app.get("/users")
def get_all_users():
    return load_data()


@app.get("/user/{user_id}")
def get_user(user_id):
    data = load_data()

    for user in data:
        if user['user_id'] == user_id:
            return user
    return {"error": "User not found"}


@app.get("/refresh/status")
@app.get("/api/refresh/status")
def get_refresh_status():
    return refresh_manager.get_status()


@app.post("/refresh", status_code=status.HTTP_202_ACCEPTED)
@app.post("/api/refresh", status_code=status.HTTP_202_ACCEPTED)
async def trigger_refresh_job():
    job = await refresh_manager.start_job()
    return {
        "status": "in_progress",
        "message": "Refresh started in background",
        "job_id": job.get("job_id"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
    }


@app.get("/refresh")
async def refresh_endpoint(sync: bool = Query(False, description="Run synchronously for legacy callers")):
    if sync:
        try:
            metadata = extract_data()
            logger.info("Extract completed synchronously.")
            return {
                "status": "success",
                "message": "Refresh completed successfully",
                "refresh_metadata": metadata,
            }
        except Exception as e:
            logger.exception("Synchronous extract failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Refresh failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Asynchronous path
    if refresh_manager.is_in_flight():
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "in_progress",
                "message": "A refresh job is already in flight",
                **refresh_manager.get_status(),
            },
        )

    job = await refresh_manager.start_job()
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "in_progress",
            "message": "Refresh started in background",
            "job_id": job.get("job_id"),
            "stage": job.get("stage"),
            "progress": job.get("progress"),
        },
    )
