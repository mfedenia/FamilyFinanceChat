import asyncio
import inspect
import time
import unittest
from unittest.mock import AsyncMock, patch

from monitoring.chat_metrics_filter import Filter


class TestChatMetricsFilter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.filter = Filter()
        # Mock the async push method to capture payloads
        self.pushed_payloads = []

        async def fake_push(push_url: str, payload: str):
            self.pushed_payloads.append((push_url, payload))

        self.filter._push_metrics_async = fake_push

    def test_inlet_and_outlet_are_async_coroutines(self):
        """Verify that inlet and outlet are coroutine functions (async def)."""
        self.assertTrue(inspect.iscoroutinefunction(self.filter.inlet))
        self.assertTrue(inspect.iscoroutinefunction(self.filter.outlet))

    async def test_two_simultaneous_chats_non_interleaved(self):
        """
        Acceptance requirement:
        Two simultaneous chats produce two correct, non-interleaved metric records.
        """
        chat_a_inlet_body = {
            "chat_id": "chat-AAA",
            "id": "msg-101",
            "model": "model-alpha",
            "messages": [{"role": "user", "content": "Hello alpha"}],
        }
        chat_b_inlet_body = {
            "chat_id": "chat-BBB",
            "id": "msg-202",
            "model": "model-beta",
            "messages": [
                {"role": "user", "content": "Hello beta 1"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Hello beta 2"},
            ],
        }

        # 1. Inlet for Chat A
        await self.filter.inlet(
            chat_a_inlet_body,
            __user__={"id": "user-1"},
            __chat_id__="chat-AAA",
        )
        self.assertIn(("chat-AAA", "msg-101"), self.filter._state)

        # 2. Inlet for Chat B (concurrent chat before Chat A has completed outlet)
        await self.filter.inlet(
            chat_b_inlet_body,
            __user__={"id": "user-2"},
            __chat_id__="chat-BBB",
        )
        self.assertIn(("chat-BBB", "msg-202"), self.filter._state)

        # Verify state is isolated per (chat_id, message_id)
        state_a = self.filter._state[("chat-AAA", "msg-101")]
        state_b = self.filter._state[("chat-BBB", "msg-202")]
        self.assertEqual(state_a["model"], "model-alpha")
        self.assertEqual(state_a["msg_count"], 1)
        self.assertEqual(state_b["model"], "model-beta")
        self.assertEqual(state_b["msg_count"], 3)

        # 3. Simulate delay and outlet for Chat A
        await asyncio.sleep(0.02)
        chat_a_outlet_body = {
            "chat_id": "chat-AAA",
            "id": "msg-101",
            "usage": {"prompt_tokens": 15, "completion_tokens": 30},
        }
        await self.filter.outlet(
            chat_a_outlet_body,
            __user__={"id": "user-1"},
            __chat_id__="chat-AAA",
        )

        # State A must be popped; State B must remain intact
        self.assertNotIn(("chat-AAA", "msg-101"), self.filter._state)
        self.assertIn(("chat-BBB", "msg-202"), self.filter._state)

        # 4. Simulate delay and outlet for Chat B
        await asyncio.sleep(0.02)
        chat_b_outlet_body = {
            "chat_id": "chat-BBB",
            "id": "msg-202",
            "usage": {"prompt_tokens": 50, "completion_tokens": 90},
        }
        await self.filter.outlet(
            chat_b_outlet_body,
            __user__={"id": "user-2"},
            __chat_id__="chat-BBB",
        )

        # State B must be popped; state dictionary is now empty
        self.assertNotIn(("chat-BBB", "msg-202"), self.filter._state)
        self.assertEqual(len(self.filter._state), 0)

        # 5. Verify captured metrics for both chats
        self.assertEqual(len(self.pushed_payloads), 2)
        payload_a = self.pushed_payloads[0][1]
        payload_b = self.pushed_payloads[1][1]

        # Assert Chat A payload correctness
        self.assertIn('openwebui_chat_completion_seconds{model="model-alpha"}', payload_a)
        self.assertIn('openwebui_chat_context_length{model="model-alpha"} 1', payload_a)
        self.assertIn('openwebui_llm_prompt_tokens{model="model-alpha"} 15', payload_a)
        self.assertIn('openwebui_llm_completion_tokens{model="model-alpha"} 30', payload_a)
        self.assertNotIn('model-beta', payload_a)

        # Assert Chat B payload correctness
        self.assertIn('openwebui_chat_completion_seconds{model="model-beta"}', payload_b)
        self.assertIn('openwebui_chat_context_length{model="model-beta"} 3', payload_b)
        self.assertIn('openwebui_llm_prompt_tokens{model="model-beta"} 50', payload_b)
        self.assertIn('openwebui_llm_completion_tokens{model="model-beta"} 90', payload_b)
        self.assertNotIn('model-alpha', payload_b)

    async def test_fallback_matching_when_outlet_message_id_differs(self):
        """If outlet generates a different response ID, match by chat_id."""
        inlet_body = {
            "chat_id": "chat-xyz",
            "id": "prompt-msg-1",
            "model": "model-gamma",
            "messages": [{"role": "user", "content": "Question"}],
        }
        await self.filter.inlet(inlet_body)

        outlet_body = {
            "chat_id": "chat-xyz",
            "id": "assistant-msg-2",  # Different message id
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        await self.filter.outlet(outlet_body)

        self.assertEqual(len(self.pushed_payloads), 1)
        self.assertIn('model="model-gamma"', self.pushed_payloads[0][1])

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

    async def test_valves_disabled(self):
        """When valves.enabled is False, inlet/outlet should do nothing."""
        self.filter.valves.enabled = False
        body = {"chat_id": "c1", "id": "m1", "messages": []}
        ret_inlet = await self.filter.inlet(body)
        self.assertEqual(len(self.filter._state), 0)
        self.assertEqual(ret_inlet, body)

        ret_outlet = await self.filter.outlet(body)
        self.assertEqual(len(self.pushed_payloads), 0)
        self.assertEqual(ret_outlet, body)

    async def test_real_push_metrics_async_silently_ignores_connection_error(self):
        """Verify real _push_metrics_async handles unreachable Pushgateway gracefully."""
        real_filter = Filter()
        # Should not raise exception even when pointing to invalid host
        await real_filter._push_metrics_async(
            "http://127.0.0.1:65534/metrics/job/test", "metric_test 1\n"
        )


if __name__ == "__main__":
    unittest.main()

