"""
title: Chat Metrics Filter
description: Captures chat-stage timing metrics (payload processing, LLM inference, token counts) and pushes them to a Prometheus Pushgateway. Install via Open WebUI Admin > Workspace > Functions.
author: FamilyFinanceChat
version: 1.0.0
license: MIT
"""

import time
import urllib.request
import json
from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    """
    OpenWebUI Filter Function that captures per-chat metrics and pushes them
    to a Prometheus Pushgateway.

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
        self._state = {}  # Internal state to store timing across inlet/outlet

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Runs before the LLM call.
        Records start time and context size in internal state.
        """
        if not self.valves.enabled:
            return body

        user_id = __user__.get("id") if __user__ else "anonymous"
        messages = body.get("messages", [])
        msg_count = len(messages)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = total_chars // 4

        # Store timing and context metadata internally, indexed by user_id.
        self._state[user_id] = {
            "start": time.perf_counter(),
            "msg_count": msg_count,
            "estimated_tokens": estimated_tokens,
            "model": body.get("model", "unknown"),
        }

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Runs after the LLM response is assembled.
        Calculates elapsed time, extracts token usage, and pushes all metrics
        to the Pushgateway.
        """
        if not self.valves.enabled:
            return body

        user_id = __user__.get("id") if __user__ else "anonymous"
        metrics_meta = self._state.pop(user_id, None)
        
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

        # Push to Pushgateway via HTTP POST (no prometheus_client dependency).
        payload = "\n".join(lines) + "\n"
        push_url = (
            f"{self.valves.pushgateway_url.rstrip('/')}"
            f"/metrics/job/{self.valves.job_name}"
        )

        try:
            req = urllib.request.Request(
                push_url,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "text/plain; version=0.0.4"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            # Silently ignore push failures — metrics are best-effort.
            # Logging here would spam the OW logs on every chat if pushgateway is down.
            pass

        return body
