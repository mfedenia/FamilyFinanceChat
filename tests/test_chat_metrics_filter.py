import asyncio
import inspect
import time
import unittest
from aiohttp import web

from monitoring.chat_metrics_filter import Filter


class TestChatMetricsFilter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.filter = Filter()
        self.pushed_payloads = []

        async def fake_push(push_url: str, payload: str):
            self.pushed_payloads.append((push_url, payload))

        self.filter._push_metrics_async = fake_push

    def test_inlet_and_outlet_are_async_coroutines(self):
        """Verify that inlet and outlet are coroutine functions (async def)."""
        self.assertTrue(inspect.iscoroutinefunction(self.filter.inlet))
        self.assertTrue(inspect.iscoroutinefunction(self.filter.outlet))

    async def test_upstream_v095_request_shape_and_metadata(self):
        """
        In Open WebUI v0.9.5, main.py moves chat_id and message_id into metadata
        (and passes __metadata__ to filter hooks).
        Verify inlet and outlet correctly resolve them from metadata.
        """
        # Upstream v0.9.5 inlet shape: chat_id & id removed from top level, moved to metadata
        inlet_body = {
            "model": "model-v095",
            "messages": [{"role": "user", "content": "Hello from 0.9.5"}],
            "metadata": {
                "chat_id": "chat-95",
                "message_id": "msg-95-req",
            },
        }
        await self.filter.inlet(
            inlet_body,
            __metadata__={"chat_id": "chat-95", "message_id": "msg-95-req"},
        )
        self.assertIn(("chat-95", "msg-95-req"), self.filter._state)

        # Upstream v0.9.5 outlet shape
        outlet_body = {
            "model": "model-v095",
            "usage": {"prompt_tokens": 12, "completion_tokens": 24},
            "metadata": {
                "chat_id": "chat-95",
                "message_id": "msg-95-req",
            },
        }
        await self.filter.outlet(
            outlet_body,
            __metadata__={"chat_id": "chat-95", "message_id": "msg-95-req"},
        )

        self.assertNotIn(("chat-95", "msg-95-req"), self.filter._state)
        self.assertEqual(len(self.pushed_payloads), 1)
        payload = self.pushed_payloads[0][1]
        self.assertIn('openwebui_chat_completion_seconds{model="model-v095"}', payload)
        self.assertIn('openwebui_llm_prompt_tokens{model="model-v095"} 12', payload)
        self.assertIn('openwebui_llm_completion_tokens{model="model-v095"} 24', payload)

    async def test_two_simultaneous_chats_with_asyncio_gather(self):
        """
        Acceptance requirement:
        Two simultaneous chats executed concurrently on the event loop via asyncio.gather
        produce two correct, non-interleaved metric records.
        """
        async def run_chat(chat_id: str, message_id: str, model: str, duration: float, prompt_tokens: int, completion_tokens: int):
            # Upstream v0.9.5 request shape
            inlet_body = {
                "model": model,
                "messages": [{"role": "user", "content": f"Message in {chat_id}"}],
                "metadata": {
                    "chat_id": chat_id,
                    "message_id": message_id,
                },
            }
            await self.filter.inlet(
                inlet_body,
                __metadata__={"chat_id": chat_id, "message_id": message_id},
            )

            # Simulate LLM inference duration concurrently
            await asyncio.sleep(duration)

            outlet_body = {
                "model": model,
                "id": message_id,
                "chat_id": chat_id,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
            return await self.filter.outlet(outlet_body)

        # Run Chat A (longer delay) and Chat B (shorter delay) simultaneously
        await asyncio.gather(
            run_chat("chat-A", "msg-A1", "model-alpha", duration=0.04, prompt_tokens=20, completion_tokens=40),
            run_chat("chat-B", "msg-B1", "model-beta", duration=0.01, prompt_tokens=55, completion_tokens=75),
        )

        # State must be completely drained
        self.assertEqual(len(self.filter._state), 0)
        self.assertEqual(len(self.pushed_payloads), 2)

        # Because Chat B had shorter duration, it should finish and push first
        first_payload = self.pushed_payloads[0][1]
        second_payload = self.pushed_payloads[1][1]

        self.assertIn('model="model-beta"', first_payload)
        self.assertIn('openwebui_llm_prompt_tokens{model="model-beta"} 55', first_payload)
        self.assertIn('openwebui_llm_completion_tokens{model="model-beta"} 75', first_payload)
        self.assertNotIn('model-alpha', first_payload)

        self.assertIn('model="model-alpha"', second_payload)
        self.assertIn('openwebui_llm_prompt_tokens{model="model-alpha"} 20', second_payload)
        self.assertIn('openwebui_llm_completion_tokens{model="model-alpha"} 40', second_payload)
        self.assertNotIn('model-beta', second_payload)

    async def test_missing_identifiers_safely_ignored(self):
        """
        If either chat_id or message_id is absent, inlet should not record state
        and outlet should return without pushing to prevent colliding keys or leaks.
        """
        # Missing message_id
        body_no_msg = {
            "model": "model-x",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"chat_id": "c1"},
        }
        await self.filter.inlet(body_no_msg)
        self.assertEqual(len(self.filter._state), 0)

        # Missing chat_id
        body_no_chat = {
            "model": "model-x",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"message_id": "m1"},
        }
        await self.filter.inlet(body_no_chat)
        self.assertEqual(len(self.filter._state), 0)

        # Outlet with missing IDs should return early
        await self.filter.outlet(body_no_msg)
        self.assertEqual(len(self.pushed_payloads), 0)

    async def test_valves_disabled(self):
        """When valves.enabled is False, inlet/outlet should do nothing."""
        self.filter.valves.enabled = False
        body = {
            "model": "model-off",
            "messages": [],
            "metadata": {"chat_id": "c1", "message_id": "m1"},
        }
        ret_inlet = await self.filter.inlet(body)
        self.assertEqual(len(self.filter._state), 0)
        self.assertEqual(ret_inlet, body)

        ret_outlet = await self.filter.outlet(body)
        self.assertEqual(len(self.pushed_payloads), 0)
        self.assertEqual(ret_outlet, body)

    async def test_stale_state_cleanup(self):
        """Ensure state older than max_age_seconds is pruned."""
        self.filter._state[("old-chat", "old-msg")] = {
            "start": time.perf_counter() - 400.0,
            "msg_count": 1,
            "estimated_tokens": 5,
            "model": "old-model",
        }
        self.filter._state[("fresh-chat", "fresh-msg")] = {
            "start": time.perf_counter(),
            "msg_count": 1,
            "estimated_tokens": 5,
            "model": "fresh-model",
        }

        self.filter._cleanup_stale_state(max_age_seconds=300.0)

        self.assertNotIn(("old-chat", "old-msg"), self.filter._state)
        self.assertIn(("fresh-chat", "fresh-msg"), self.filter._state)

    async def test_real_http_push_to_mock_pushgateway_server(self):
        """
        Brings up an in-process HTTP Pushgateway server on a local port,
        runs the real _push_metrics_async over aiohttp, and asserts
        that Prometheus metrics arrive intact over HTTP.
        """
        received_requests = []

        async def handle_post(request):
            body = await request.text()
            received_requests.append({
                "path": request.path,
                "content_type": request.headers.get("Content-Type"),
                "body": body,
            })
            return web.Response(text="OK")

        app = web.Application()
        app.router.add_post("/metrics/job/{job_name}", handle_post)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)  # Random available port
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        try:
            real_filter = Filter()
            real_filter.valves.pushgateway_url = f"http://127.0.0.1:{port}"
            real_filter.valves.job_name = "test_job"

            # Execute real inlet and outlet with real HTTP push
            inlet_body = {
                "model": "model-http-test",
                "messages": [{"role": "user", "content": "ping"}],
                "metadata": {"chat_id": "chat-http", "message_id": "msg-http"},
            }
            await real_filter.inlet(inlet_body)

            outlet_body = {
                "id": "msg-http",
                "chat_id": "chat-http",
                "model": "model-http-test",
                "usage": {"prompt_tokens": 11, "completion_tokens": 22},
            }
            await real_filter.outlet(outlet_body)

            # Assert HTTP transmission succeeded
            self.assertEqual(len(received_requests), 1)
            req = received_requests[0]
            self.assertEqual(req["path"], "/metrics/job/test_job")
            self.assertEqual(req["content_type"], "text/plain; version=0.0.4")
            self.assertIn('openwebui_chat_completion_seconds{model="model-http-test"}', req["body"])
            self.assertIn('openwebui_llm_prompt_tokens{model="model-http-test"} 11', req["body"])
            self.assertIn('openwebui_llm_completion_tokens{model="model-http-test"} 22', req["body"])
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
