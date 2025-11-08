from fastapi import FastAPI 
from logger import logging
from extract_chats import main as extract_data
import uvicorn
import json

logger = logging.getLogger("professor_dashboard")

DATA_PATH = "data/extracted_chats.json"

app = FastAPI(title = "Professor Dashboard")


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
        if user['email'] == user_id:
            return user
    return {"error": "User not found"}

@app.get("/refresh")
def refresh_data():
    logger.info("Refreshing the data")
    extract_data()
    return {"message": "Data refreshed"}

uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)