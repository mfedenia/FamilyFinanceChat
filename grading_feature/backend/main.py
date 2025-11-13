from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from logger import logging
from extract_chats import main as extract_data
import uvicorn
import json

logger = logging.getLogger("professor_dashboard")

DATA_PATH = "data/extracted_chats.json"

app = FastAPI(title = "Professor Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)
 
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
def refresh_data():
    logger.info("Refreshing the data")
    extract_data()
    return {"message": "Data refreshed"}

uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)