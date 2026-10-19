"""Minimal HTTP clients for structured LLM output (OpenAI and Anthropic).

Implemented directly on HTTPX (no vendor SDKs) so transports can be mocked in tests and
no request is ever made unless the LLM is explicitly enabled with an approved key.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from pit.data.providers.http import redact


class LLMError(Exception):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class LLMResponse:
    content: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    raw_model: str | None = None


class LLMClient(ABC):
    provider: str = "abstract"

    def __init__(self, model: str, timeout: float = 45.0, max_output_tokens: int = 900,
                 transport: httpx.BaseTransport | None = None, max_retries: int = 1, sleep=time.sleep):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.transport = transport
        self.max_retries = max_retries
        self.sleep = sleep

    @abstractmethod
    def _request(self, client: httpx.Client, system: str, user: str, schema: dict) -> httpx.Response: ...

    @abstractmethod
    def _parse(self, payload: dict) -> tuple[dict, int | None, int | None, str | None]: ...

    def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        attempt = 0
        while True:
            t0 = time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    resp = self._request(client, system, user, schema)
            except httpx.HTTPError as exc:
                err = LLMError(f"network error: {exc}", retryable=True)
            else:
                latency = (time.monotonic() - t0) * 1000
                if resp.status_code < 400:
                    try:
                        content, tin, tout, model = self._parse(resp.json())
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise LLMError(f"unparseable response: {exc}") from exc
                    return LLMResponse(content, tin, tout, latency, model)
                retryable = resp.status_code == 429 or resp.status_code >= 500
                err = LLMError(redact(f"HTTP {resp.status_code}: {resp.text[:300]}"), resp.status_code, retryable)
            if not err.retryable or attempt >= self.max_retries:
                raise err
            self.sleep(2.0 * (2**attempt))
            attempt += 1


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", **kw):
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(self, client, system, user, schema):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "forecast_review", "strict": True, "schema": schema}},
            "max_completion_tokens": self.max_output_tokens,
        }
        return client.post(f"{self.base_url}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {self.api_key}"})

    def _parse(self, payload):
        msg = payload["choices"][0]["message"]
        if msg.get("refusal"):
            raise ValueError(f"model refused: {msg['refusal']}")
        content = json.loads(msg["content"])
        usage = payload.get("usage") or {}
        return content, usage.get("prompt_tokens"), usage.get("completion_tokens"), payload.get("model")


class AnthropicClient(LLMClient):
    """Anthropic Messages API; structured output via a single forced tool call."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.anthropic.com", **kw):
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(self, client, system, user, schema):
        body = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": "submit_review", "description": "Submit the structured forecast review.",
                       "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "submit_review"},
        }
        return client.post(f"{self.base_url}/v1/messages", json=body,
                           headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})

    def _parse(self, payload):
        block = next(b for b in payload["content"] if b.get("type") == "tool_use")
        usage = payload.get("usage") or {}
        return block["input"], usage.get("input_tokens"), usage.get("output_tokens"), payload.get("model")
