"""Focused checks for upgrade-window extraction comparisons."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_extractions import compare, load_snapshot


def user(user_id, question="Question", answer="Answer"):
    return {
        "user_id": user_id,
        "chats": [{
            "title": "Session",
            "message_pairs": [{"timestamp": "10/01/2026 10:00", "question": question, "answer": answer}],
        }],
    }


class CompareExtractionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, name, data):
        path = Path(self.temp.name) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_chat_and_user_order_does_not_change_result(self):
        before = self.write("before.json", [user("1"), user("2")])
        after = self.write("after.json", [user("2"), user("1")])
        self.assertTrue(compare(before, after))

    def test_changed_owner_or_content_fails(self):
        before = self.write("before.json", [user("1"), user("2")])
        self.assertFalse(compare(before, self.write("owner.json", [user("1"), user("3")])))
        self.assertFalse(compare(before, self.write("content.json", [user("1", answer="Different"), user("2")])))

    def test_missing_or_empty_data_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            load_snapshot(self.write("empty.json", []))
        with self.assertRaisesRegex(ValueError, "no saved message pairs"):
            load_snapshot(self.write("pairs.json", [{"user_id": "1", "chats": []}]))


if __name__ == "__main__":
    unittest.main()
