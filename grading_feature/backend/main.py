from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
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
        extract_data()
        logger.info("Extract completed successfully.")
    except Exception as e:
        logger.error(f"Extract failed: {e}")