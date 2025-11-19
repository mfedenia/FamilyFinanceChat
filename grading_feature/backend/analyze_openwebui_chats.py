#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import io
import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timezone

# -----------------------
# Utility Functions
# -----------------------

def iter_json_records(path):
    """
    Supports both JSON and JSONL formats:
    - If JSONL: parse line by line
    - If JSON: parse as list or dict
    Uniformly yield as Python objects
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        # Roughly determine if jsonl
        is_jsonl = "\n" in head and head.strip().startswith("{") and not head.strip().endswith("}")
        if is_jsonl:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        # Fault tolerance: some lines are comma-separated or invalid json
                        try:
                            yield json.loads(line.rstrip(","))
                        except Exception:
                            continue
        else:
            data = json.load(f)
            # Could be list or dict
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data

def try_get(dct, path_list, default=None):
    """Safely get nested field: path_list=['source','user','name']"""
    cur = dct
    for key in path_list:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def ts_to_iso(ts):
    """Convert seconds/milliseconds/ISO/string time to ISO8601 (UTC) if possible"""
    if ts is None:
        return ""
    # Number (seconds or milliseconds)
    if isinstance(ts, (int, float)):
        # Roughly determine if seconds or milliseconds
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            return str(ts)
    # String: try to return directly
    if isinstance(ts, str):
        return ts
    return str(ts)

def first_n(s, n=120):
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    return s[:n]

def infer_account_from_path(path):
    """
    Try to infer account name from file path (e.g., export directory organized by user)
    You can customize this based on your directory structure.
    Rule: take the second-to-last directory name as account hint (common for multi-user export directories).
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return ""

def guess_account(node, meta, path):
    """
    Try to infer account name (user display name) from node/metadata/path
    Priority: node.source.user.name / node.user.name / node.author.name / node.sources[].source.user.name / meta / path
    """
    # Common locations in single node
    cand = try_get(node, ["source", "user", "name"]) or try_get(node, ["user", "name"]) or try_get(node, ["author", "name"])
    if cand:
        return cand
    # List-type sources
    if isinstance(node.get("sources"), list):
        for s in node["sources"]:
            cand = try_get(s, ["source", "user", "name"]) or try_get(s, ["user", "name"])
            if cand:
                return cand
    # Conversation-level meta
    cand = try_get(meta, ["account"]) or try_get(meta, ["user", "name"])
    if cand:
        return cand
    # Path fallback
    return infer_account_from_path(path) or "(unknown)"

def find_user_info(obj):
    """
    Recursively find user_id and user_name in object
    Return the first one found
    """
    if isinstance(obj, dict):
        # Direct user field
        if "user" in obj and isinstance(obj["user"], dict):
            uid = obj["user"].get("id")
            uname = obj["user"].get("name")
            if uid or uname:
                return uid, uname
        # Direct id/name fields
        if "id" in obj and "name" in obj:
            return obj.get("id"), obj.get("name")
        # Recursive search
        for v in obj.values():
            uid, uname = find_user_info(v)
            if uid or uname:
                return uid, uname
    elif isinstance(obj, list):
        for item in obj:
            uid, uname = find_user_info(item)
            if uid or uname:
                return uid, uname
    return None, None

def extract_identity(node, meta, path):
    """
    Prioritize recursively finding user_id and user_name
    """
    uid, uname = find_user_info(node)
    if not uid or not uname:
        uid2, uname2 = find_user_info(meta)
        uid = uid or uid2
        uname = uname or uname2
    # Path fallback
    hint = infer_account_from_path(path) or uname or "(unknown)"
    key = uid or hint
    return key, (uid or ""), (uname or ""), hint

# -----------------------
# Parse Different Formats
# -----------------------

def extract_messages_from_record(record):
    """
    Adapt to various common export structures, uniformly extract as "message node list".
    Returns: nodes, meta
    nodes: list(dict) containing at least: id, parentId, role, content, model, timestamp
    meta: dict containing conversation-level metadata (optional)
    """
    nodes = []
    meta = {}

    # 1) Your sample: top level is a node (or group of nodes):
    #   { "id": ..., "parentId": ..., "role": "user", "content": "...", "timestamp": ... , "models": [...] , "source": {...} }
    # Or contains "childrenIds"
    possible_node_keys = ["id", "role", "content", "timestamp"]
    if any(k in record for k in possible_node_keys):
        nodes.append(record)
        return nodes, meta

    # 2) Some OpenWebUI export formats: {"messages": [...]} or {"items":[...]} or {"conversations":[...] }
    for key in ["messages", "items", "conversations", "nodes", "data"]:
        if key in record and isinstance(record[key], list):
            return record[key], {k: v for k, v in record.items() if k != key}

    # 3) LangChain/custom structure: {"conversation": {"messages":[...]}, "meta":{...}}
    for key in ["conversation", "thread", "chat"]:
        if key in record and isinstance(record[key], dict):
            inner = record[key]
            for msg_key in ["messages", "nodes", "items"]:
                if msg_key in inner and isinstance(inner[msg_key], list):
                    return inner[msg_key], {k: v for k, v in record.items() if k != key}

    # 4) Fallback: record is list?
    if isinstance(record, list):
        return record, {}

    # Cannot recognize
    return [], {}

def normalize_node(node, fallback_conv_id=""):
    """
    Normalize node fields: return standardized dictionary
    Standard fields:
      conv_id, node_id, parent_id, role, content, model, model_list, timestamp
    """
    node_id = node.get("id") or node.get("_id") or node.get("message_id") or node.get("uuid")
    parent_id = node.get("parentId") or node.get("parent_id") or node.get("reply_to")
    role = node.get("role") or node.get("speaker") or ""
    content = node.get("content") or node.get("text") or node.get("message") or ""
    model = node.get("model") or ""
    # Some structures have "models": [...]
    model_list = []
    if "models" in node and isinstance(node["models"], list):
        model_list = node["models"]
        if not model and model_list:
            model = model_list[0]

    timestamp = node.get("timestamp") or node.get("created_at") or node.get("time")
    conv_id = node.get("conversation_id") or node.get("thread_id") or fallback_conv_id

    return {
        "conv_id": conv_id,
        "node_id": node_id,
        "parent_id": parent_id,
        "role": role,
        "content": content,
        "model": model,
        "model_list": model_list,
        "timestamp": timestamp
    }

# -----------------------
# Main Process
# -----------------------

def analyze(paths, outdir):
    os.makedirs(outdir, exist_ok=True)

    messages_rows = []
    unanswered_rows = []
    users_rows_map = defaultdict(lambda: {
        "user_id": "",
        "user_name": "",
        "account_hint": "",
        "files": set(),
        "conversations": set(),
        "messages": 0,
        "user_msgs": 0,
        "assistant_msgs": 0,
        "other_msgs": 0
    })
    global_models = Counter()
    total_conversations = set()

    for path in paths:
        for rec in iter_json_records(path):
            nodes, meta = extract_messages_from_record(rec)
            if not nodes:
                continue

            fallback_conv = try_get(meta, ["id"]) or try_get(meta, ["conversation_id"]) or os.path.basename(path)
            std_nodes = [normalize_node(n, fallback_conv_id=fallback_conv) for n in nodes]

            children_map = defaultdict(list)
            id_map = {}
            for n in std_nodes:
                if n["node_id"]:
                    id_map[n["node_id"]] = n
                if n["parent_id"]:
                    children_map[n["parent_id"]].append(n["node_id"])

            for n in std_nodes:
                total_conversations.add(n["conv_id"] or fallback_conv)

                if n["model"]:
                    global_models[n["model"]] += 1
                for m in n["model_list"]:
                    global_models[m] += 1

                key, uid, uname, hint = extract_identity(n, meta, path)

                # User dimension aggregation (prioritize grouping by user_id)
                users_rows_map[key]["user_id"] = uid
                users_rows_map[key]["user_name"] = uname
                users_rows_map[key]["account_hint"] = hint
                users_rows_map[key]["files"].add(path)
                if n["conv_id"]:
                    users_rows_map[key]["conversations"].add(n["conv_id"])
                users_rows_map[key]["messages"] += 1
                if n["role"] == "user":
                    users_rows_map[key]["user_msgs"] += 1
                elif n["role"] == "assistant":
                    users_rows_map[key]["assistant_msgs"] += 1
                else:
                    users_rows_map[key]["other_msgs"] += 1

                messages_rows.append({
                    "file": path,
                    "user_id": uid,
                    "user_name": uname,
                    "account_hint": hint,
                    "conversation_id": n["conv_id"] or "",
                    "node_id": n["node_id"] or "",
                    "parent_id": n["parent_id"] or "",
                    "role": n["role"],
                    "timestamp_iso": ts_to_iso(n["timestamp"]),
                    "model": n["model"],
                    "models_all": ",".join(map(str, n["model_list"])) if n["model_list"] else "",
                    "content_snippet": first_n(n["content"], 160)
                })

            # Identify "questions not replied by assistant"
            for n in std_nodes:
                if n["role"] != "user":
                    continue
                child_ids = children_map.get(n["node_id"] or "", [])
                has_assistant_reply = any((id_map.get(cid) or {}).get("role") == "assistant" for cid in child_ids)
                if not has_assistant_reply:
                    key, uid, uname, hint = extract_identity(n, meta, path)
                    unanswered_rows.append({
                        "file": path,
                        "user_id": uid,
                        "user_name": uname,
                        "account_hint": hint,
                        "conversation_id": n["conv_id"] or "",
                        "node_id": n["node_id"] or "",
                        "timestamp_iso": ts_to_iso(n["timestamp"]),
                        "question_snippet": first_n(n["content"], 200)
                    })

    # Write messages.csv (with user_id / user_name)
    messages_csv = os.path.join(outdir, "messages.csv")
    with open(messages_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "user_id", "user_name", "account_hint",
            "conversation_id", "node_id", "parent_id",
            "role", "timestamp_iso", "model", "models_all", "content_snippet"
        ])
        writer.writeheader()
        writer.writerows(messages_rows)

    # Write unanswered.csv (with user_id / user_name)
    unanswered_csv = os.path.join(outdir, "unanswered.csv")
    with open(unanswered_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "user_id", "user_name", "account_hint",
            "conversation_id", "node_id", "timestamp_iso", "question_snippet"
        ])
        writer.writeheader()
        writer.writerows(unanswered_rows)

    # Write users.csv (with user_id as primary key)
    users_csv = os.path.join(outdir, "users.csv")
    with open(users_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id", "user_name", "account_hint",
            "files_count", "conversations_count", "messages",
            "user_msgs", "assistant_msgs", "other_msgs"
        ])
        writer.writeheader()
        for k, v in sorted(users_rows_map.items(), key=lambda x: (x[1]["user_id"] or x[0])):
            writer.writerow({
                "user_id": v["user_id"],
                "user_name": v["user_name"],
                "account_hint": v["account_hint"],
                "files_count": len(v["files"]),
                "conversations_count": len(v["conversations"]),
                "messages": v["messages"],
                "user_msgs": v["user_msgs"],
                "assistant_msgs": v["assistant_msgs"],
                "other_msgs": v["other_msgs"],
            })

    # Terminal output
    print("\n=== Summary ===")
    print(f"Conversations (unique): {len(total_conversations)}")
    print(f"Messages total: {len(messages_rows)}")
    print(f"Unanswered user questions: {len(unanswered_rows)}")
    print(f"Users (unique by user_id/fallback): {len(users_rows_map)}")

    # Count messages by user_id
    user_id_counts = defaultdict(int)
    for v in users_rows_map.values():
        uid = v["user_id"] or v["account_hint"]
        user_id_counts[uid] += v["messages"]

    print("\nUser message counts:")
    for uid, count in sorted(user_id_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {uid}: {count} messages")

    print("Top models:")
    for m, c in global_models.most_common(10):
        print(f"  {m}: {c}")
    print(f"\nCSV written to: {os.path.abspath(outdir)}")
    print(f" - {messages_csv}")
    print(f" - {unanswered_csv}")
    print(f" - {users_csv}")

def collect_paths(input_path):
    input_path = os.path.abspath(input_path)
    paths = []
    if os.path.isdir(input_path):
        # Recursively collect .json / .jsonl
        for ext in ("**/*.json", "**/*.jsonl"):
            paths.extend(glob.glob(os.path.join(input_path, ext), recursive=True))
    else:
        paths.append(input_path)
    # Deduplicate & filter non-existent
    paths = [p for p in sorted(set(paths)) if os.path.exists(p)]
    return paths

def select_json_files():
    """
    Search for JSON/JSONL files in current directory and let user select files to analyze
    """
    current_dir = os.getcwd()
    json_files = []
    
    # Search for .json and .jsonl files
    for ext in ["*.json", "*.jsonl"]:
        json_files.extend(glob.glob(os.path.join(current_dir, ext)))
        json_files.extend(glob.glob(os.path.join(current_dir, "**", ext), recursive=True))
    
    # Deduplicate and sort
    json_files = sorted(set(json_files))
    
    if not json_files:
        print("No JSON/JSONL files found in current directory and subdirectories.")
        return []
    
    print(f"Found the following JSON/JSONL files in {current_dir}:")
    print("-" * 60)
    for i, file in enumerate(json_files, 1):
        rel_path = os.path.relpath(file, current_dir)
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"{i:2d}. {rel_path} ({file_size:.1f} KB)")
    
    print("-" * 60)
    print("Please select files to analyze (enter numbers separated by commas, or 'all' for all files):")
    
    while True:
        try:
            user_input = input("Selection: ").strip()
            
            if user_input.lower() == 'all':
                return json_files
            
            # Parse user input numbers
            selected_indices = []
            for part in user_input.split(','):
                part = part.strip()
                if '-' in part:  # Support range selection like "1-3"
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    selected_indices.append(int(part))
            
            # Validate index range
            selected_files = []
            for idx in selected_indices:
                if 1 <= idx <= len(json_files):
                    selected_files.append(json_files[idx - 1])
                else:
                    print(f"Invalid index: {idx}")
                    continue
            
            if selected_files:
                print(f"Selected {len(selected_files)} file(s):")
                for file in selected_files:
                    print(f"  - {os.path.relpath(file, current_dir)}")
                return selected_files
            else:
                print("No valid files selected. Please try again.")
                
        except ValueError:
            print("Invalid input format. Please enter numbers, ranges (e.g., 1-3), or 'all'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return []

def main():
    parser = argparse.ArgumentParser(description="Analyze OpenWebUI exported chats (multi-user, JSON/JSONL).")
    parser.add_argument("--input", help="Input file or directory (optional - will search current dir if not provided)")
    parser.add_argument("--outdir", default="./out", help="Output directory for CSVs")
    args = parser.parse_args()

    if args.input:
        # Use path provided via command line
        paths = collect_paths(args.input)
        if not paths:
            print("No JSON/JSONL files found.")
            return
    else:
        # Interactive file selection in current directory
        paths = select_json_files()
        if not paths:
            return
    analyze(paths, args.outdir)    # Start analysis

if __name__ == "__main__":
    main()
