import sqlite3
import os
import pandas as pd 
import json
from logger import logging
import datetime
import tempfile
from dotenv import load_dotenv

load_dotenv()

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


NOTE: To make this more efficient so we don't have to dump large json files over and over again, we can change the query
      so we get the most recent timestamp per user chat title and query the data with the timestamp being 
      greater than that (will need to figure out how to do this)

'''

logger = logging.getLogger("professor_dashboard")

DB_PATH = os.getenv("DB_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
EXTRACT_USER_ROLES = os.getenv("EXTRACT_USER_ROLES", "user")


class ExtractionError(Exception):
    pass

def get_connection():
    """Creates a connection with dict like rows """
    if not DB_PATH:
        raise ExtractionError("DB_PATH is not set")
    if not os.path.exists(DB_PATH):
        raise ExtractionError(f"DB_PATH does not exist: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logger.debug("Created database connection")
    return conn

def get_all_users(conn):
    """Queries the db and gives back one corresponding tuple back"""
    base_query = "SELECT id, email, name, role, created_at FROM user"

    raw_roles = (EXTRACT_USER_ROLES or "user").strip()
    if raw_roles.lower() in {"all", "*"}:
        query = f"{base_query};"
        cur = conn.execute(query)
        logger.info("Extracting chats for all user roles")
        return cur.fetchall()

    roles = [r.strip() for r in raw_roles.split(",") if r.strip()]
    if not roles:
        roles = ["user"]

    placeholders = ",".join(["?"] * len(roles))
    query = f"{base_query} WHERE role IN ({placeholders});"
    cur = conn.execute(query, roles)
    logger.info(f"Extracting chats for roles: {roles}")

    return cur.fetchall()

def get_chats_by_user(conn, user_id):
    query = "SELECT chat FROM chat WHERE user_id = ?;"
    cur = conn.execute(query, (user_id,))
    return cur.fetchall() 

def parse_json(json_string):
    try:
        data = json.loads(json_string)
        return data
    except Exception:
        logger.warning("Can't parse chat JSON row; skipping malformed row")
        return None


def normalize_timestamp(ts):
    try:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return int(ts)
        if isinstance(ts, str) and ts.strip() != "":
            return int(float(ts))
    except Exception:
        return None
    return None

def get_timestamp(ts):
    # Returns in format "mm/dd/yyyy" (0 padded if its a single digit)
    ts_format = datetime.datetime.fromtimestamp(ts)
    date_formatted = ts_format.strftime("%m/%d/%Y")
    time_formatted = ts_format.strftime("%H:%M")

    return f"{date_formatted} {time_formatted}"

def build_hieracrchy(conn):
    """Builds hieractchy like the shape above"""

    logger.info("Building logger hierarchy")
    
    all_users = []
    users_processed = 0
    chat_entries_processed = 0
    message_pairs_processed = 0
    malformed_chat_rows = 0
    latest_message_epoch = None

    try:
        users = get_all_users(conn)
        
        if not users:
            logger.warning(f"No users found in DB")
            return [], {
                "users_processed": 0,
                "chat_entries_processed": 0,
                "message_pairs_processed": 0,
                "latest_message_timestamp_found": None,
                "malformed_chat_rows_skipped": 0,
            }
        
        for user in users:
            user_id, email, name, role, created_date = user[0], user[1], user[2], user[3], user[4] 
            users_processed += 1
            
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
                    malformed_chat_rows += 1
                    continue

                messages = processed_json.get('messages', [])
                if not isinstance(messages, list):
                    logger.warning("Skipping malformed chat row where messages is not a list")
                    malformed_chat_rows += 1
                    continue

                chat_entry = { 
                    "title": processed_json.get('title', 'Unknown'),
                    "message_pairs": []
                }
                chat_entries_processed += 1

                for j in range(0, len(messages), 2):
                    if not isinstance(messages[j], dict):
                        logger.warning("Skipping malformed message entry that is not an object")
                        continue

                    ts_epoch = normalize_timestamp(messages[j].get('timestamp'))
                    if ts_epoch is not None:
                        latest_message_epoch = max(latest_message_epoch, ts_epoch) if latest_message_epoch is not None else ts_epoch

                    ts = get_timestamp(ts_epoch if ts_epoch is not None else 0)
                    q = messages[j].get("content", "")
                    if j + 1 < len(messages) and isinstance(messages[j + 1], dict):
                        a = messages[j + 1].get("content", "")
                    else:
                        a = None

                    chat_entry["message_pairs"].append({"timestamp": ts, "question": q, "answer": a})
                    message_pairs_processed += 1

                json_structure["chats"].append(chat_entry)
                logger.info(f"Added a chat entry")

            all_users.append(json_structure)  
            logger.info(f"Created hierarchy for {name} ({email}))")     
    except Exception as e:
        raise ExtractionError(f"Failed while building hierarchy: {e}") from e

    latest_message_timestamp_found = get_timestamp(latest_message_epoch) if latest_message_epoch is not None else None
    metadata = {
        "users_processed": users_processed,
        "chat_entries_processed": chat_entries_processed,
        "message_pairs_processed": message_pairs_processed,
        "latest_message_timestamp_found": latest_message_timestamp_found,
        "malformed_chat_rows_skipped": malformed_chat_rows,
    }

    return all_users, metadata


def export_json(all_users):
    if not OUTPUT_PATH:
        raise ExtractionError("OUTPUT_PATH is not set")

    output_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_file_path = None
    try:
        fd, temp_file_path = tempfile.mkstemp(prefix=".grading-refresh-", suffix=".tmp", dir=output_dir or None)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(all_users, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file_path, OUTPUT_PATH)
        logger.info("Exported User JSON file atomically")
        return os.path.abspath(OUTPUT_PATH)
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                logger.warning("Failed to remove temp file after export failure")
        raise ExtractionError(f"Failed to export JSON: {e}") from e


def main():
    try: 
        conn = get_connection()
        all_users, metadata = build_hieracrchy(conn)
        output_file_path = export_json(all_users)

        metadata["output_file_path"] = output_file_path
        return metadata

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.debug("DB Connection closed")


if __name__ == "__main__":
    main()
    