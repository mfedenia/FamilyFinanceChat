"""Exercise the actual smoke payload through the repository Filter hooks.

This checks client/Filter compatibility, not server hook invocation.
"""

import unittest
from unittest.mock import AsyncMock

from monitoring.chat_metrics_filter import Filter
import test_smoke_test


class SmokeMetricsFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_payload_emits_metric_with_both_identifier_shapes(self):
        client = test_smoke_test.SmokeTestClientTests()
        try:
            client.setUp()
            client.run_existing()
            body = next(call[3]["json"] for call in client.calls
                        if call[1] == "/api/chat/completions")
        finally:
            client.doCleanups()
        self.assertEqual(body["id"], f"smoke-message-{client.run_id}")
        for metadata_shape in (False, True):
            with self.subTest(metadata_shape=metadata_shape):
                payload = dict(body)
                metadata = None
                if metadata_shape:
                    metadata = {"chat_id": payload.pop("chat_id"),
                                "message_id": payload.pop("id")}
                function = Filter()
                function._push_metrics_async = AsyncMock()
                await function.inlet(payload, __metadata__=metadata)
                self.assertEqual(len(function._state), 1)
                await function.outlet(payload, __metadata__=metadata)
                function._push_metrics_async.assert_awaited_once()
                self.assertIn("openwebui_chat_completion_seconds",
                              function._push_metrics_async.call_args.args[1])
                self.assertEqual(function._state, {})
