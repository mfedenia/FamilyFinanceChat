"""Smoke-client behavior with a simulated Open WebUI API; no server or secrets needed."""

import contextlib
import io
import os
import unittest
from unittest.mock import Mock, patch

import requests

from scripts import smoke_test


class SmokeTestClientTests(unittest.TestCase):
    def setUp(self):
        self.run_id = "test-run-a13"
        self.existing_token = "unit-test-existing-bearer"
        self.signup_token = "unit-test-signup-bearer"
        self.output = io.StringIO()
        self.session = Mock()
        self.session.headers = {}
        self.session.get.return_value = self.response({})
        self.payloads = {
            "/api/v1/auths/signup": {"id": "new-user", "role": "admin", "token": self.signup_token},
            "/api/chat/completions": {"choices": [{"message": {"content": "Hello"}}]},
            "/api/v1/knowledge/create": {"id": "new-kb"},
            "/api/v1/files/?process=true&process_in_background=false": {"id": "new-file"},
            "/api/v1/files/new-file/process/status": {"status": "completed"},
            "/api/v1/knowledge/new-kb/file/add": {"files": [{"id": "new-file"}]},
            "/api/v1/retrieval/query/collection": {"documents": [[f"CI-RETRIEVAL-{self.run_id}"]]},
        }
        self.failures = {}
        self.calls = []

        def request(method, url, **kwargs):
            path = url.removeprefix("http://localhost:3003")
            self.calls.append((method, path, dict(self.session.headers), kwargs))
            if path in self.failures:
                result = self.failures[path]
                if isinstance(result, Exception):
                    raise result
                return result
            return self.response(self.payloads[path])

        self.session.request.side_effect = request
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        self.session_factory = contexts.enter_context(patch.object(smoke_test.requests, "Session", return_value=self.session))
        contexts.enter_context(patch.object(smoke_test.secrets, "token_hex", return_value=self.run_id))
        contexts.enter_context(patch.dict(os.environ, {"A13_TEST_TOKEN": self.existing_token}))
        contexts.enter_context(contextlib.redirect_stdout(self.output))

    @staticmethod
    def response(payload, status=200, text=""):
        return Mock(ok=status < 400, status_code=status, text=text, json=Mock(return_value=payload))

    def run_existing(self):
        smoke_test.main(["--existing-token-env", "A13_TEST_TOKEN"])

    def test_fresh_mode_keeps_signup_and_full_retrieval(self):
        smoke_test.main([])
        self.assertEqual(self.calls[0][1], "/api/v1/auths/signup")
        self.assertEqual(self.calls[1][2]["Authorization"], f"Bearer {self.signup_token}")
        self.assertEqual(self.calls[-1][1], "/api/v1/retrieval/query/collection")
        self.assertIn('"id": "new-user"', self.output.getvalue())
        self.assertIn("Smoke test passed", self.output.getvalue())
        self.assertNotIn(self.signup_token, self.output.getvalue())

    def test_existing_mode_skips_signup_and_records_resources(self):
        self.run_existing()
        self.assertFalse(any("signup" in path for _, path, _, _ in self.calls))
        for _, _, headers, _ in self.calls:
            self.assertEqual(headers["Authorization"], f"Bearer {self.existing_token}")
        for kind, resource_id in (("knowledge", "new-kb"), ("file", "new-file")):
            self.assertIn(f'"kind": "{kind}", "id": "{resource_id}"', self.output.getvalue())
        self.assertIn("Smoke test passed", self.output.getvalue())
        self.assertNotIn(self.existing_token, self.output.getvalue())

    def test_missing_or_blank_token_fails_before_any_network(self):
        for value in (None, "", " \t\n"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["A13_TEST_TOKEN"] = value
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    self.run_existing()
                self.session_factory.assert_not_called()

    def test_auth_failure_never_falls_back_to_signup_or_echoes_response(self):
        for status in (401, 403):
            with self.subTest(status=status):
                self.calls.clear()
                self.failures["/api/chat/completions"] = self.response({}, status, self.existing_token)
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as caught:
                    self.run_existing()
                self.assertEqual([call[1] for call in self.calls], ["/api/chat/completions"])
                self.assertNotIn(self.existing_token, str(caught.exception) + self.output.getvalue())

    def test_fresh_mode_requires_admin(self):
        self.payloads["/api/v1/auths/signup"]["role"] = "pending"
        with self.assertRaisesRegex(RuntimeError, "admin bearer token"):
            smoke_test.main([])
        self.assertEqual(len(self.calls), 1)

    def test_chat_error_stops_before_upload(self):
        self.failures["/api/chat/completions"] = self.response({}, 500)
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self.run_existing()
        self.assertEqual(len(self.calls), 1)

    def test_empty_chat_fails(self):
        self.payloads["/api/chat/completions"] = {"choices": []}
        with self.assertRaisesRegex(RuntimeError, "no assistant text"):
            self.run_existing()
        self.assertEqual(len(self.calls), 1)

    def test_upload_failure_preserves_knowledge_id_for_cleanup(self):
        self.failures["/api/v1/files/?process=true&process_in_background=false"] = self.response({}, 500)
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self.run_existing()
        self.assertIn('"id": "new-kb"', self.output.getvalue())
        self.assertNotIn('"id": "new-file"', self.output.getvalue())

    def test_processing_failure_preserves_file_id(self):
        self.payloads["/api/v1/files/new-file/process/status"] = {"status": "failed"}
        with self.assertRaisesRegex(RuntimeError, "processing did not complete"):
            self.run_existing()
        self.assertIn('"id": "new-file"', self.output.getvalue())

    def test_retrieval_must_return_this_runs_marker(self):
        self.payloads["/api/v1/retrieval/query/collection"] = {"documents": [["unrelated document"]]}
        with self.assertRaisesRegex(RuntimeError, "uploaded document marker"):
            self.run_existing()
        self.assertNotIn("Smoke test passed", self.output.getvalue())

    def test_transport_error_does_not_echo_credentials(self):
        self.failures["/api/chat/completions"] = requests.ConnectionError(self.existing_token)
        with self.assertRaisesRegex(RuntimeError, "request failed") as caught:
            self.run_existing()
        self.assertNotIn(self.existing_token, str(caught.exception) + self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
