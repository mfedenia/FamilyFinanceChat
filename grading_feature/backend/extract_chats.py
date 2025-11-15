import sqlite3
import pandas as pd 
import json
from logger import logging
import datetime


'''
OUTPUT_JSON_SHAPE = {
    "user_id" : "",
    "name" : "" ,
    "email": "" ,
    "role" : "" ,
    "join_date": "",
    "chats" : [ {
        "title" : "" ,
        "message_pairs" : [
            {
                "timestamp": "...", # in format "MM/dd/YYYY HH/MM"
                "question" : "...",
                "answer" : "..." 
            } ,  
            ... 
            {

            }
        ]
    ]
}

---------------------------------------


NOTE: To make this better so we don't have to dump large json files over and over again, we can change the query
      so we get the most recent timestamp per user chat title and query the data with the timestamp being 
      greater than that (will need to figure out how to do this)

'''

logger = logging.getLogger("professor_dashboard")

DB_PATH = "/opt/openwebui/data/webui.db"
EMAILS_PATH = "data/test_emails.csv"
OUTPUT_PATH = "data/extracted_chats.json"

all_users = []

def get_connection():
    """Creates a connection with dict like rows """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logger.debug("Created database connection")
    return conn

def get_all_users(conn):
    """Queries the db and gives back one corresponding tuple back"""
    query = "SELECT id, email, name, role, created_at FROM user WHERE role = 'user';"
    cur = conn.execute(query)

    return cur.fetchall()

def get_chats_by_user(conn, user_id):
    query = "SELECT chat FROM chat WHERE user_id = ?;"
    cur = conn.execute(query, (user_id,))
    return cur.fetchall() 

def parse_json(json_string):
    try:
        data = json.loads(json_string)
        return data
    except Exception as e:
        logger.error("Can't parse json string")
        return None

def get_timestamp(ts):
    ts_format = datetime.datetime.fromtimestamp(ts)
    date_formatted = ts_format.strftime("%m/%d/%Y")
    time_formatted = ts_format.strftime("%H:%M")

    return f"{date_formatted} {time_formatted}"

def build_hieracrchy(conn):
    """Builds hieractchy like the shape above"""

    logger.info("Building logger hierarchy")

    try:
        users = get_all_users(conn)
        
        if not users:
            logger.warning(f"No users found in DB")
            return
        
        for user in users:
            user_id, email, name, role, created_date = user[0], user[1], user[2], user[3], user[4] 
            
            join_date = get_timestamp(created_date)

            json_structure = {
                "user_id": user_id,
                "name": name,
                "email": email,
                "role": role,
                "join_date": join_date,
                "chats": []
            }

            user_chats = get_chats_by_user(conn, user_id)
            if not user_chats:
                logger.warning(f"No chats associated with {name}({email}), going to next user")
                
            
            for idx, row in enumerate(user_chats):
                chats_json = row[0] # get first val from tuple
                processed_json = parse_json(chats_json)
                if not processed_json:
                    logger.warning("Skipping broken json")
                chat_entry = { 
                    "title": processed_json.get('title', 'Unknown'),
                    "message_pairs": []
                }
                messages = processed_json.get('messages', [])
                for j in range(0, len(messages), 2):
                    ts = get_timestamp(messages[j].get('timestamp', 0))
                    q = messages[j].get("content", "")
                    a = messages[j+1].get("content", "") if j+1 < len(messages) else None
                    chat_entry["message_pairs"].append({"timestamp": ts, "question": q, "answer": a})

                json_structure["chats"].append(chat_entry)
                logger.info(f"Added a chat entry")

            all_users.append(json_structure)  
            logger.info(f"Created hierarchy for {name} ({email}))")     
    except Exception as e:
        logger.error(f"Can't query the data with email {email}: {e}")

    return None


def export_json():
    # add to json file
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_users, f, indent=4)
    logger.info("Exported User JSON file")


def main():
    try: 
        conn = get_connection()
        build_hieracrchy(conn)
        export_json()

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.debug("DB Connection closed")


if __name__ == "__main__":
    main()
    