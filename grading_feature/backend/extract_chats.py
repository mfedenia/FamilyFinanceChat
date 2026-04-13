import os
import json
from collections import defaultdict
from typing import Any

import requests
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

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:8080").rstrip("/")
OPENWEBUI_USERS_PATH = os.getenv("OPENWEBUI_USERS_PATH", "/api/v1/users/")
OPENWEBUI_CHATS_PATH = os.getenv("OPENWEBUI_CHATS_PATH", "/api/v1/chats/all")
OPENWEBUI_API_TOKEN = (
    os.getenv("OPENWEBUI_API_TOKEN")
    or os.getenv("OPENWEBUI_API_KEY")
    or ""
)
OPENWEBUI_TIMEOUT_SEC = float(os.getenv("OPENWEBUI_TIMEOUT_SEC", "30"))

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
EXTRACT_USER_ROLES = os.getenv("EXTRACT_USER_ROLES", "user")


class ExtractionError(Exception):
    pass

def normalize_api_path(path: str) -> str:
    if not path.startswith("/"):
        return f"/{path}"
    return path


def build_api_path_candidates(path: str) -> list[str]:
    normalized = normalize_api_path(path).rstrip("/")
    candidates = [normalized]

    if normalized.startswith("/api/v1/"):
        candidates.append(normalized.replace("/api/v1/", "/api/", 1))
    elif normalized.startswith("/api/"):
        candidates.append(f"/api/v1/{normalized.removeprefix('/api/')}")

    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def get_api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = (OPENWEBUI_API_TOKEN or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-Key"] = token
    return headers


def fetch_api_json(path: str) -> Any:
    last_error = None

    for candidate_path in build_api_path_candidates(path):
        url = f"{OPENWEBUI_BASE_URL}{candidate_path}"
        try:
            response = requests.get(url, headers=get_api_headers(), timeout=OPENWEBUI_TIMEOUT_SEC)
        except requests.RequestException as e:
            last_error = ExtractionError(f"Failed to connect to OpenWebUI API at {url}: {e}")
            continue

        if response.status_code >= 400:
            snippet = response.text[:400]
            unsupported_version = "unsupported api version" in snippet.lower()
            if unsupported_version and candidate_path.startswith("/api/v1/"):
                last_error = ExtractionError(
                    f"OpenWebUI API request failed ({response.status_code}) at {url}: {snippet}"
                )
                continue

            raise ExtractionError(f"OpenWebUI API request failed ({response.status_code}) at {url}: {snippet}")

        try:
            return response.json()
        except ValueError as e:
            snippet = response.text[:400]
            raise ExtractionError(
                f"OpenWebUI API returned non-JSON response at {url} (status {response.status_code}): {snippet}"
            ) from e

    if last_error is not None:
        raise last_error

    raise ExtractionError(f"Failed to resolve a working OpenWebUI API endpoint for {path}")


def coerce_list(payload: Any, preferred_keys: list[str]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


def get_all_users():
    payload = fetch_api_json(OPENWEBUI_USERS_PATH)
    users = coerce_list(payload, ["data", "users", "items", "results"])

    raw_roles = (EXTRACT_USER_ROLES or "user").strip()
    if raw_roles.lower() in {"all", "*"}:
        logger.info("Extracting chats for all user roles")
        return users

    roles = [r.strip() for r in raw_roles.split(",") if r.strip()]
    if not roles:
        roles = ["user"]

    role_set = set(roles)
    filtered_users = [u for u in users if str(u.get("role", "")).strip() in role_set]
    logger.info(f"Extracting chats for roles: {roles}")
    return filtered_users


def get_all_chats():
    payload = fetch_api_json(OPENWEBUI_CHATS_PATH)
    return coerce_list(payload, ["data", "chats", "items", "results"])


def get_chat_details(chat_id: str) -> dict[str, Any]:
    if not chat_id:
        raise ExtractionError("Cannot load chat details without a chat_id")

    payload = fetch_api_json(f"/api/v1/chats/{chat_id}")
    if isinstance(payload, dict):
        return payload

    raise ExtractionError(f"OpenWebUI chat detail endpoint returned non-object payload for chat_id={chat_id}")


def parse_chat_payload(chat_item: dict[str, Any]):
    raw = chat_item.get("chat")
    if raw is None:
        raw = chat_item.get("payload")
    if raw is None:
        raw = chat_item.get("data")
    if raw is None:
        raw = chat_item

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return parse_json(raw)
    return None


def get_chat_id(chat_item: dict[str, Any], chat_payload: dict[str, Any] | None = None) -> str | None:
    candidates = [
        chat_item.get("chat_id"),
        chat_item.get("chatId"),
        chat_item.get("id"),
    ]

    if isinstance(chat_payload, dict):
        candidates.extend([
            chat_payload.get("chat_id"),
            chat_payload.get("chatId"),
            chat_payload.get("id"),
        ])

    for candidate in candidates:
        if candidate:
            return str(candidate)

    return None


def resolve_user_id(chat_item: dict[str, Any], chat_payload: dict[str, Any] | None):
    direct = chat_item.get("user_id") or chat_item.get("userId")
    if direct is not None:
        return str(direct)

    user_obj = chat_item.get("user")
    if isinstance(user_obj, dict) and user_obj.get("id") is not None:
        return str(user_obj.get("id"))

    if isinstance(chat_payload, dict):
        nested_uid = chat_payload.get("user_id")
        if nested_uid is not None:
            return str(nested_uid)

    return None

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

def build_hieracrchy():
    """Builds hieractchy like the shape above"""

    logger.info("Building logger hierarchy")
    
    all_users = []
    users_processed = 0
    chat_entries_processed = 0
    message_pairs_processed = 0
    malformed_chat_rows = 0
    latest_message_epoch = None

    try:
        users = get_all_users()
        all_chats = get_all_chats()

        chats_by_user = defaultdict(list)
        for chat_item in all_chats:
            if not isinstance(chat_item, dict):
                malformed_chat_rows += 1
                continue

            chat_payload = parse_chat_payload(chat_item)
            if not isinstance(chat_payload, dict):
                malformed_chat_rows += 1
                continue

            if "messages" not in chat_payload:
                chat_id = get_chat_id(chat_item, chat_payload)
                if not chat_id:
                    malformed_chat_rows += 1
                    logger.warning("Skipping shallow chat payload without chat_id")
                    continue

                deep_chat_payload = get_chat_details(chat_id)
                chat_payload = parse_chat_payload(deep_chat_payload)
                if not isinstance(chat_payload, dict):
                    malformed_chat_rows += 1
                    logger.warning(f"Skipping unreadable deep chat payload for chat_id={chat_id}")
                    continue

            user_id = resolve_user_id(chat_item, chat_payload)
            if not user_id:
                malformed_chat_rows += 1
                continue

            chats_by_user[user_id].append(chat_payload)
        
        if not users:
            logger.warning("No users found from OpenWebUI API")
            return [], {
                "users_processed": 0,
                "chat_entries_processed": 0,
                "message_pairs_processed": 0,
                "latest_message_timestamp_found": None,
                "malformed_chat_rows_skipped": 0,
            }
        
        for user in users:
            user_id = str(user.get("id", ""))
            email = user.get("email")
            name = user.get("name")
            role = user.get("role")
            created_date = normalize_timestamp(user.get("created_at"))

            if not user_id:
                logger.warning("Skipping user without id from OpenWebUI users payload")
                continue

            users_processed += 1
            
            join_date = get_timestamp(created_date if created_date is not None else 0)

            json_structure = {
                "user_id": user_id,
                "name": name,
                "email": email,
                "role": role,
                "join_date": join_date,
                "chats": []
            }

            user_chats = chats_by_user.get(user_id, [])
            if not user_chats:
                logger.warning(f"No chats associated with {name}({email}), going to next user")
                
            
            for processed_json in user_chats:
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

                open_message_pair = None

                for message in messages:
                    if not isinstance(message, dict):
                        logger.warning("Skipping malformed message entry that is not an object")
                        continue

                    ts_epoch = normalize_timestamp(message.get("timestamp"))
                    if ts_epoch is not None:
                        latest_message_epoch = max(latest_message_epoch, ts_epoch) if latest_message_epoch is not None else ts_epoch

                    ts = get_timestamp(ts_epoch if ts_epoch is not None else 0)
                    role = str(message.get("role", "")).strip().lower()
                    content = message.get("content", "")

                    if role == "user":
                        if open_message_pair is not None:
                            chat_entry["message_pairs"].append(open_message_pair)
                            message_pairs_processed += 1

                        open_message_pair = {
                            "timestamp": ts,
                            "question": content,
                            "answer": None,
                        }
                    elif role == "assistant":
                        if open_message_pair is None:
                            logger.warning("Skipping assistant message without an open user question")
                            continue

                        open_message_pair["answer"] = content
                        chat_entry["message_pairs"].append(open_message_pair)
                        message_pairs_processed += 1
                        open_message_pair = None

                if open_message_pair is not None:
                    chat_entry["message_pairs"].append(open_message_pair)
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
        all_users, metadata = build_hieracrchy()
        output_file_path = export_json(all_users)

        metadata["output_file_path"] = output_file_path
        return metadata

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
    