"""Run: python -m unittest discover -s tests -p 'test_avatar_writeback.py' -v"""
import copy
import pathlib
import sys
import unittest
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.avatar_writeback.writer import AvatarWriteback, JsonAPI, WritebackError


class FakeAPI:
    """Simulates v0.11.4's student-scoped public chat endpoints."""
    def __init__(self, user_id="student-123"):
        self.user_id = user_id
        self.chats = {}
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        if method == "POST" and path == "/api/v1/chats/new":
            chat_id = str(uuid.uuid4())
            self.chats[chat_id] = {"id": chat_id, "user_id": self.user_id, "chat": copy.deepcopy(body["chat"])}
            return copy.deepcopy(self.chats[chat_id])
        if method == "GET" and path.startswith("/api/v1/chats/"):
            return copy.deepcopy(self.chats[path.rsplit("/", 1)[-1]])
        if method == "POST" and path.startswith("/api/v1/chats/"):
            chat_id = path.rsplit("/", 1)[-1]
            self.chats[chat_id]["chat"] = copy.deepcopy(body["chat"])
            return copy.deepcopy(self.chats[chat_id])
        raise AssertionError(f"Unexpected API call {method} {path}")


class B13Tests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.writer = AvatarWriteback(self.api, model="avatar-test", expected_user_id="student-123")
        self.chat_id = self.writer.start_session(title="Avatar Practice")

    def test_two_turns_persist_in_flat_list_and_tree(self):
        self.assertTrue(self.writer.append_turn(self.chat_id, turn_id="turn-1", student_text="How is your budget?", avatar_text="Tight.", timestamp=1000))
        self.assertTrue(self.writer.append_turn(self.chat_id, turn_id="turn-2", student_text="What about debt?", avatar_text="We have loans.", timestamp=1010))
        chat = self.writer.get_chat(self.chat_id)["chat"]
        messages = chat["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m["content"] for m in messages], ["How is your budget?", "Tight.", "What about debt?", "We have loans."])
        self.assertEqual([m["timestamp"] for m in messages], [1000, 1000, 1010, 1010])
        tree = chat["history"]["messages"]
        for msg in messages:
            self.assertEqual(tree[msg["id"]]["content"], msg["content"])
        self.assertEqual(tree[messages[0]["id"]]["childrenIds"], [messages[1]["id"]])
        self.assertIn(messages[2]["id"], tree[messages[1]["id"]]["childrenIds"])
        self.assertEqual(chat["history"]["currentId"], messages[3]["id"])
        self.assertEqual(self.api.chats[self.chat_id]["user_id"], "student-123")

    def test_idempotent_turn_retry(self):
        self.writer.append_turn(self.chat_id, turn_id="once", student_text="Question?", avatar_text="Answer")
        self.assertFalse(self.writer.append_turn(self.chat_id, turn_id="once", student_text="Question?", avatar_text="Answer"))
        self.assertEqual(len(self.writer.get_chat(self.chat_id)["chat"]["messages"]), 2)
        self.assertEqual(len([c for c in self.api.calls if c[0] == "POST" and c[1].endswith(self.chat_id)]), 1)

    def test_wrong_student_cannot_be_attributed(self):
        other = AvatarWriteback(self.api, model="avatar-test", expected_user_id="not-this-student")
        with self.assertRaisesRegex(WritebackError, "owner"):
            other.append_turn(self.chat_id, turn_id="wrong", student_text="Q", avatar_text="A")
        self.assertEqual(self.writer.get_chat(self.chat_id)["chat"]["messages"], [])

    def test_rejects_empty_and_unknown_data(self):
        for kwargs in [
            {"turn_id": "", "student_text": "Q", "avatar_text": "A"},
            {"turn_id": "x", "student_text": "  ", "avatar_text": "A"},
            {"turn_id": "x", "student_text": "Q", "avatar_text": ""},
        ]:
            with self.assertRaises(ValueError):
                self.writer.append_turn(self.chat_id, **kwargs)

    def test_fake_extractor_contract(self):
        """Existing extract_chats.py needs top-level user_id and chat.messages[]."""
        self.writer.append_turn(self.chat_id, turn_id="t1", student_text="Question?", avatar_text="Answer.")
        row = self.writer.get_chat(self.chat_id)
        self.assertIn("user_id", row)
        self.assertEqual(row["user_id"], "student-123")
        self.assertIsInstance(row["chat"]["messages"], list)
        self.assertEqual([(m["role"], m["content"]) for m in row["chat"]["messages"]], [("user", "Question?"), ("assistant", "Answer.")])

    def test_missing_or_dummy_token_rejected(self):
        with self.assertRaises(ValueError):
            JsonAPI("http://localhost:3000", "REPLACE_ME")
        with self.assertRaises(ValueError):
            JsonAPI("http://localhost:3000", "")

    def test_detects_failed_write_when_not_persisted(self):
        class BrokenAPI(FakeAPI):
            def request(self, method, path, body=None):
                if method == "POST" and path.startswith("/api/v1/chats/") and path != "/api/v1/chats/new":
                    return {"id": path.rsplit("/", 1)[-1], "user_id": self.user_id, "chat": body["chat"]}
                return super().request(method, path, body)
        broken = BrokenAPI()
        writer = AvatarWriteback(broken, model="m", expected_user_id="student-123")
        chat_id = writer.start_session()
        with self.assertRaisesRegex(WritebackError, "did not persist"):
            writer.append_turn(chat_id, turn_id="t", student_text="Q", avatar_text="A")


if __name__ == "__main__":
    unittest.main()
