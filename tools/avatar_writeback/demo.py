"""B13: safe synthetic end-to-end write-back test (creates a student-owned chat)."""
from __future__ import annotations

import argparse
import os
import sys
from .writer import AvatarWriteback, JsonAPI, WritebackError


def main() -> int:
    p = argparse.ArgumentParser(description="B13 synthetic transcript write-back to Open WebUI")
    p.add_argument("--base-url", default="http://localhost:3000")
    p.add_argument("--model", required=True, help="Real configured model name; only stored as chat metadata")
    p.add_argument("--student-id", required=True, help="ID of the Open WebUI user who owns the bearer token")
    p.add_argument("--confirm-synthetic-write", action="store_true", help="Create a synthetic chat in Open WebUI")
    args = p.parse_args()
    if not args.confirm_synthetic_write:
        p.error("Pass --confirm-synthetic-write to create a synthetic chat")
    token = os.environ.get("AVATAR_STUDENT_API_KEY", "")
    try:
        writer = AvatarWriteback(JsonAPI(args.base_url, token), model=args.model, expected_user_id=args.student_id)
        chat_id = writer.start_session(title="B13 verification — synthetic avatar practice")
        writer.append_turn(chat_id, turn_id="demo-1", student_text="What financial goals are most important for your family?", avatar_text="We want to save for college and build an emergency fund.")
        writer.append_turn(chat_id, turn_id="demo-2", student_text="How much do you currently have in emergency savings?", avatar_text="We have approximately two months of essential expenses saved.")
        stored = writer.get_chat(chat_id)
        msgs = stored["chat"]["messages"]
        if len(msgs) != 4:
            raise WritebackError("Expected exactly four saved messages")
        print("B13 synthetic write-back PASS")
        print(f"chat_id={chat_id} owner_user_id={stored['user_id']} message_count={len(msgs)}")
        print("NEXT: grading dashboard /refresh and verify 2 pairs for the same student.")
        return 0
    except (WritebackError, ValueError) as exc:
        print(f"B13 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
