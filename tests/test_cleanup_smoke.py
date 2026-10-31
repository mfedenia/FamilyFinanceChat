"""Cleanup refuses unrelated resources and deletes only after an explicit flag."""

import os
import unittest
from unittest.mock import Mock, patch

from scripts import cleanup_smoke


RUN_ID = "1234567890abcdef"
KB_ID = "kb-id"
FILE_ID = "file-id"
ARGS = ["--base-url", "http://example.invalid", "--run-id", RUN_ID,
        "--knowledge-id", KB_ID, "--file-id", FILE_ID, "--token-env", "CLEANUP_TOKEN"]


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        session = Mock()
        session.headers = {}
        payloads = {
            f"/api/v1/knowledge/{KB_ID}": {"id": KB_ID, "name": f"CI Smoke {RUN_ID}", "user_id": "owner", "files": [{"id": FILE_ID}]},
            f"/api/v1/files/{FILE_ID}": {"id": FILE_ID, "filename": f"smoke-{RUN_ID}.txt", "user_id": "owner"},
        }

        def request(method, url, **kwargs):
            path = url.removeprefix("http://example.invalid")
            self.calls.append((method, path))
            return Mock(ok=True, json=Mock(return_value=payloads.get(path, {})))

        session.request.side_effect = request
        self.payloads = payloads
        self.factory = patch.object(cleanup_smoke.requests, "Session", return_value=session)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.env = patch.dict(os.environ, {"CLEANUP_TOKEN": "fixture-token"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_dry_run_only_reads(self):
        cleanup_smoke.main(ARGS)
        self.assertEqual([method for method, _ in self.calls], ["GET", "GET"])

    def test_execute_deletes_exact_file_and_knowledge_base(self):
        cleanup_smoke.main(ARGS + ["--execute"])
        self.assertEqual(self.calls[-2:], [
            ("DELETE", f"/api/v1/files/{FILE_ID}"),
            ("DELETE", f"/api/v1/knowledge/{KB_ID}/delete"),
        ])

    def test_name_or_owner_mismatch_prevents_deletion(self):
        self.payloads[f"/api/v1/files/{FILE_ID}"]["user_id"] = "other"
        with self.assertRaisesRegex(RuntimeError, "owners do not match"):
            cleanup_smoke.main(ARGS + ["--execute"])
        self.assertFalse(any(method == "DELETE" for method, _ in self.calls))


if __name__ == "__main__":
    unittest.main()
