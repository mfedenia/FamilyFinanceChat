from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from logger import logging
from extract_chats import main as extract_data
from scoring_service import score_questions, OPENAI_MODEL, MOCK_SCORER
import uvicorn
import json
import os
import base64
import hashlib
import hmac
import time
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("professor_dashboard")

DATA_PATH = os.getenv("DATA_PATH")
STUDENT_FEEDBACK_ENABLED = os.getenv("STUDENT_FEEDBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
OPENWEBUI_HMAC_SECRET = os.getenv("OPENWEBUI_HMAC_SECRET", "")
ALLOW_INSECURE_STUDENT_ID = os.getenv("ALLOW_INSECURE_STUDENT_ID", "false").lower() in {"1", "true", "yes", "on"}

app = FastAPI(title = "Professor Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
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


def _b64url_decode(raw: str) -> bytes:
    padded = raw + "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _verify_student_token(token: str) -> dict[str, Any]:
    if not OPENWEBUI_HMAC_SECRET:
        raise HTTPException(status_code=500, detail={"error": "OPENWEBUI_HMAC_SECRET is not configured"})

    if "." not in token:
        raise HTTPException(status_code=401, detail={"error": "Malformed student token"})

    payload_b64, sig_hex = token.split(".", 1)
    expected = hmac.new(
        OPENWEBUI_HMAC_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, sig_hex):
        raise HTTPException(status_code=401, detail={"error": "Invalid token signature"})

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail={"error": "Invalid token payload"})

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail={"error": "Token missing user_id"})

    exp = payload.get("exp")
    if exp is not None:
        try:
            if int(exp) < int(time.time()):
                raise HTTPException(status_code=401, detail={"error": "Token expired"})
        except ValueError:
            raise HTTPException(status_code=401, detail={"error": "Invalid token exp"})

    return payload


def _get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()

    alt = request.headers.get("X-Student-Token")
    return alt.strip() if alt else None


def resolve_student_identity(request: Request, expected_user_id: str | None = None) -> str:
    if not STUDENT_FEEDBACK_ENABLED:
        raise HTTPException(status_code=403, detail={"error": "Student feedback is disabled"})

    token = _get_bearer_token(request)
    authenticated_user_id = None

    if token and OPENWEBUI_HMAC_SECRET:
        payload = _verify_student_token(token)
        authenticated_user_id = payload["user_id"]
    elif OPENWEBUI_HMAC_SECRET and not token:
        raise HTTPException(status_code=401, detail={"error": "Missing student token"})
    elif ALLOW_INSECURE_STUDENT_ID:
        authenticated_user_id = request.headers.get("X-Student-User-Id") or expected_user_id
    else:
        raise HTTPException(status_code=401, detail={"error": "Student authentication is required"})

    if not authenticated_user_id:
        raise HTTPException(status_code=401, detail={"error": "Unable to resolve student identity"})

    if expected_user_id and authenticated_user_id != expected_user_id:
        raise HTTPException(status_code=403, detail={"error": "Forbidden: can only access your own feedback"})

    return authenticated_user_id


def log_student_feedback_event(request: Request, user_id: str, status: str, message: str = ""):
    request_id = request.headers.get("X-Request-Id", "n/a")
    logger.info(
        f"student_feedback request_id={request_id} user_id={user_id} status={status} path={request.url.path} message={message}"
    )


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


@app.get("/api/health")
def score_health():
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "mock": MOCK_SCORER,
        "student_feedback_enabled": STUDENT_FEEDBACK_ENABLED,
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


@app.get("/api/student-feedback/me")
async def student_feedback_me(request: Request, useAbi: bool = True):
    user_id = resolve_student_identity(request)

    data = load_data()
    selected_user = next((u for u in data if u.get("user_id") == user_id), None)

    if not selected_user:
        log_student_feedback_event(request, user_id, "not_found", "User not found in extracted dataset")
        raise HTTPException(status_code=404, detail={"error": "User not found"})

    questions = extract_questions_for_user(selected_user)
    if not questions:
        log_student_feedback_event(request, user_id, "empty", "No question-like prompts found")
        return {
            "ok": True,
            "user": {
                "user_id": selected_user.get("user_id"),
                "name": selected_user.get("name"),
                "email": selected_user.get("email"),
            },
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
        log_student_feedback_event(request, user_id, "success", f"questions={len(questions)} useAbi={useAbi}")
        return {
            "ok": True,
            "user": {
                "user_id": selected_user.get("user_id"),
                "name": selected_user.get("name"),
                "email": selected_user.get("email"),
            },
            "results": results,
            "aggregate": aggregate,
        }
    except Exception as e:
        log_student_feedback_event(request, user_id, "error", str(e))
        logger.exception(f"Student feedback scoring failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/student-feedback/{user_id}")
async def student_feedback_by_id(user_id: str, request: Request, useAbi: bool = True):
    """Compatibility route. Enforces self-only access when token authentication is enabled."""
    resolved_user_id = resolve_student_identity(request, expected_user_id=user_id)

    data = load_data()
    selected_user = next((u for u in data if u.get("user_id") == resolved_user_id), None)

    if not selected_user:
        log_student_feedback_event(request, resolved_user_id, "not_found", "User not found in extracted dataset")
        raise HTTPException(status_code=404, detail={"error": "User not found"})

    questions = extract_questions_for_user(selected_user)
    if not questions:
        log_student_feedback_event(request, resolved_user_id, "empty", "No question-like prompts found")
        return {
            "ok": True,
            "user": {
                "user_id": selected_user.get("user_id"),
                "name": selected_user.get("name"),
                "email": selected_user.get("email"),
            },
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
        log_student_feedback_event(request, resolved_user_id, "success", f"questions={len(questions)} useAbi={useAbi}")
        return {
            "ok": True,
            "user": {
                "user_id": selected_user.get("user_id"),
                "name": selected_user.get("name"),
                "email": selected_user.get("email"),
            },
            "results": results,
            "aggregate": aggregate,
        }
    except Exception as e:
        log_student_feedback_event(request, resolved_user_id, "error", str(e))
        logger.exception(f"Student feedback scoring failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

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

@app.get("/refresh")
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