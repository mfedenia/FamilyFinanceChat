"""Compare two grading extractor JSON files without printing student content.

During an upgrade window, writers are blocked, so each user's saved chats and
ordered question/answer pairs should remain identical. Chat order in the API is
not stable and is ignored; duplicate chats are still counted.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def load_snapshot(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("extraction must be a non-empty list of users")
    users = {}
    pair_count = 0
    for user in data:
        if not isinstance(user, dict) or not isinstance(user.get("user_id"), str):
            raise ValueError("each user needs a string user_id")
        user_id = user["user_id"]
        if not user_id or user_id in users:
            raise ValueError("user_id is empty or duplicated")
        chats = user.get("chats")
        if not isinstance(chats, list):
            raise ValueError("each user needs a chats list")
        records = Counter()
        for chat in chats:
            if not isinstance(chat, dict) or not isinstance(chat.get("title"), str):
                raise ValueError("each chat needs a title")
            pairs = chat.get("message_pairs")
            if not isinstance(pairs, list):
                raise ValueError("each chat needs a message_pairs list")
            normalized = []
            for pair in pairs:
                if not isinstance(pair, dict) or not all(
                    key in pair for key in ("timestamp", "question", "answer")
                ):
                    raise ValueError("each pair needs timestamp, question and answer")
                normalized.append({key: pair[key] for key in ("timestamp", "question", "answer")})
            pair_count += len(normalized)
            records[json.dumps([chat["title"], normalized], sort_keys=True, ensure_ascii=False)] += 1
        users[user_id] = records
    if pair_count == 0:
        raise ValueError("extraction has no saved message pairs to compare")
    return users, sum(sum(chats.values()) for chats in users.values()), pair_count


def compare(before, after):
    old_users, old_chats, old_pairs = load_snapshot(before)
    new_users, new_chats, new_pairs = load_snapshot(after)
    changed_users = sum(old_users.get(user_id) != new_users.get(user_id) for user_id in old_users.keys() & new_users.keys())
    matched = old_users == new_users
    print(
        f"baseline users/chats/pairs={len(old_users)}/{old_chats}/{old_pairs}; "
        f"after={len(new_users)}/{new_chats}/{new_pairs}; "
        f"missing users={len(old_users.keys() - new_users.keys())}; "
        f"new users={len(new_users.keys() - old_users.keys())}; changed users={changed_users}"
    )
    return matched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("after")
    args = parser.parse_args(argv)
    try:
        if compare(args.baseline, args.after):
            print("Extraction comparison passed")
            return 0
        print("Extraction comparison failed", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Extraction comparison invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
