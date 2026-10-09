"""
title: Chat Metrics Filter
description: Captures chat-stage timing metrics (payload processing, LLM inference, token counts) and pushes them to a Prometheus Pushgateway. Install via Open WebUI Admin > Workspace > Functions.
author: FamilyFinanceChat
version: 1.1.0
license: MIT
"""

import asyncio
import time
from typing import Any, Dict, Optional, Tuple
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default: Any = None, description: str = "") -> Any:  # type: ignore
        return default

# Support non-blocking HTTP via aiohttp or httpx; fallback to threadpool if needed
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import httpx
except ImportError:
    httpx = None

import urllib.request


class Filter:
    """
    OpenWebUI Filter Function that captures per-chat metrics and pushes them
    to a Prometheus Pushgateway asynchronously without blocking the event loop.

    Metrics pushed (as Prometheus exposition format):
      - openwebui_chat_completion_seconds   — total round-trip time (inlet→outlet)
      - openwebui_chat_context_length       — number of messages sent to LLM
      - openwebui_context_tokens_estimated  — estimated token count (chars / 4)
      - openwebui_llm_prompt_tokens         — actual prompt tokens from LLM response
      - openwebui_llm_completion_tokens     — actual completion tokens from LLM response

    Configuration (via Valves in OW Admin UI):
      - pushgateway_url: URL of the Pushgateway (default: http://pushgateway:9091)
      - job_name: Prometheus job label (default: openwebui_chat_metrics)
      - enabled: Toggle metrics collection on/off
    """

    class Valves(BaseModel):
        pushgateway_url: str = Field(
            default="http://pushgateway:9091",
            description="Prometheus Pushgateway URL (reachable from the OW container)",
        )
        job_name: str = Field(
            default="openwebui_chat_metrics",
            description="Prometheus job label for pushed metrics",
        )
        enabled: bool = Field(
            default=True,
            description="Enable or disable metrics collection",
        )

    def __init__(self):
        self.valves = self.Valves()
        # Internal state to store timing across inlet/outlet.
        # Keyed by (chat_id, message_id) rather than user_id to prevent state leaks
        # across concurrent sessions in async Open WebUI.
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _extract_identifiers(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Extracts chat_id and message_id from request/response metadata or body.
        Falls back safely if individual fields are omitted.
        """
        # 1. Determine chat_id
        chat_id = __chat_id__ or body.get("chat_id")
        if not chat_id and isinstance(body.get("metadata"), dict):
            chat_id = body["metadata"].get("chat_id")
        if not chat_id and __user__:
            chat_id = f"user_{__user__.get('id', 'anonymous')}"
        if not chat_id:
            chat_id = "default_chat"

        # 2. Determine message_id
        message_id = __message_id__ or body.get("message_id") or body.get("id")
        if not message_id and isinstance(body.get("metadata"), dict):
            message_id = body["metadata"].get("message_id")
        if not message_id:
            messages = body.get("messages", [])
            if messages and isinstance(messages[-1], dict) and messages[-1].get("id"):
                message_id = str(messages[-1].get("id"))
        if not message_id:
            message_id = "latest"

        return str(chat_id), str(message_id)

    def _cleanup_stale_state(self, max_age_seconds: float = 300.0) -> None:
        """Prunes orphaned entries if requests timed out or failed before outlet."""
        now = time.perf_counter()
        stale_keys = [
            k for k, v in self._state.items()
            if now - v.get("start", now) > max_age_seconds
        ]
        for k in stale_keys:
            self._state.pop(k, None)

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Runs before the LLM call.
        Records start time and context size in internal state, keyed by (chat_id, message_id).
        """
        if not self.valves.enabled:
            return body

        self._cleanup_stale_state()

        chat_id, message_id = self._extract_identifiers(
            body, __user__=__user__, __chat_id__=__chat_id__
        )
        messages = body.get("messages", [])
        msg_count = len(messages)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
        estimated_tokens = total_chars // 4

        state_key = (chat_id, message_id)
        self._state[state_key] = {
            "start": time.perf_counter(),
            "msg_count": msg_count,
            "estimated_tokens": estimated_tokens,
            "model": body.get("model", "unknown"),
        }

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Runs after the LLM response is assembled.
        Calculates elapsed time, extracts token usage, and pushes all metrics
        to the Pushgateway asynchronously without blocking the event loop.
        """
        if not self.valves.enabled:
            return body

        chat_id, message_id = self._extract_identifiers(
            body, __user__=__user__, __chat_id__=__chat_id__
        )
        state_key = (chat_id, message_id)

        # Retrieve and pop the state
        metrics_meta = self._state.pop(state_key, None)

        # Fallback: if message_id generated during outlet differs from inlet,
        # match the most recent pending session for this chat_id.
        if metrics_meta is None:
            matching_keys = [k for k in self._state if k[0] == chat_id]
            if matching_keys:
                target_key = max(matching_keys, key=lambda k: self._state[k].get("start", 0))
                metrics_meta = self._state.pop(target_key, None)

        if metrics_meta is None:
            return body

        elapsed = time.perf_counter() - metrics_meta["start"]
        model = metrics_meta.get("model", "unknown")
        msg_count = metrics_meta.get("msg_count", 0)
        estimated_tokens = metrics_meta.get("estimated_tokens", 0)

        # Extract actual token usage from the LLM response if available.
        usage = {}
        if isinstance(body, dict):
            # Non-streaming responses may have usage at top level
            usage = body.get("usage", {}) or {}
            # Or nested in messages
            messages = body.get("messages", [])
            if messages and isinstance(messages[-1], dict):
                usage = usage or messages[-1].get("usage", {}) or {}

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Build Prometheus exposition format payload.
        lines = [
            f'# HELP openwebui_chat_completion_seconds Total chat round-trip time',
            f'# TYPE openwebui_chat_completion_seconds gauge',
            f'openwebui_chat_completion_seconds{{model="{model}"}} {elapsed:.4f}',
            f'# HELP openwebui_chat_context_length Number of messages in context',
            f'# TYPE openwebui_chat_context_length gauge',
            f'openwebui_chat_context_length{{model="{model}"}} {msg_count}',
            f'# HELP openwebui_context_tokens_estimated Estimated token count',
            f'# TYPE openwebui_context_tokens_estimated gauge',
            f'openwebui_context_tokens_estimated{{model="{model}"}} {estimated_tokens}',
        ]

        if prompt_tokens:
            lines.extend([
                f'# HELP openwebui_llm_prompt_tokens Prompt tokens from LLM',
                f'# TYPE openwebui_llm_prompt_tokens gauge',
                f'openwebui_llm_prompt_tokens{{model="{model}"}} {prompt_tokens}',
            ])

        if completion_tokens:
            lines.extend([
                f'# HELP openwebui_llm_completion_tokens Completion tokens from LLM',
                f'# TYPE openwebui_llm_completion_tokens gauge',
                f'openwebui_llm_completion_tokens{{model="{model}"}} {completion_tokens}',
            ])

        # Push to Pushgateway via non-blocking HTTP POST.
        payload = "\n".join(lines) + "\n"
        push_url = (
            f"{self.valves.pushgateway_url.rstrip('/')}"
            f"/metrics/job/{self.valves.job_name}"
        )

        await self._push_metrics_async(push_url, payload)

        return body

    async def _push_metrics_async(self, push_url: str, payload: str) -> None:
        """
        Asynchronously sends metrics to the Pushgateway via non-blocking HTTP POST.
        Ensures the asyncio event loop is never blocked.
        """
        data = payload.encode("utf-8")
        headers = {"Content-Type": "text/plain; version=0.0.4"}

        try:
            if aiohttp is not None:
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(push_url, data=data, headers=headers) as resp:
                        await resp.read()
            elif httpx is not None:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(push_url, content=data, headers=headers)
            else:
                # Non-blocking threadpool execution if neither aiohttp nor httpx is installed
                def _sync_request():
                    req = urllib.request.Request(
                        push_url, data=data, headers=headers, method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=5):
                        pass

                await asyncio.to_thread(_sync_request)
        except Exception:
            # Silently ignore push failures — metrics are best-effort.
            # Logging here would spam the OW logs on every chat if pushgateway is down.
            pass

