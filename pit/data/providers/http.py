"""HTTP plumbing shared by REST providers: rate limiting, retries, error logging."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable

import httpx

from pit.data.providers.base import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)

log = logging.getLogger(__name__)

_SECRET_PARAM = re.compile(r"(apikey|api_key|apiKey|token|access_token)=([^&\s]+)", re.IGNORECASE)


def redact(text: str) -> str:
    """Remove credentials from URLs/messages before they are logged or stored."""
    return _SECRET_PARAM.sub(r"\1=***", text)


class RateLimiter:
    """Token bucket limiting requests per minute. Thread-safe; clock and sleep injectable."""

    def __init__(
        self,
        requests_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.capacity = max(1.0, float(requests_per_minute))
        self.rate_per_sec = float(requests_per_minute) / 60.0
        self.tokens = self.capacity
        self.clock = clock
        self.sleep = sleep
        self.updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be made. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate_per_sec)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return waited
                need = (1.0 - self.tokens) / self.rate_per_sec
            self.sleep(need)
            waited += need


class HttpJsonClient:
    """GET-only JSON client with bounded exponential backoff.

    * 401/403 -> :class:`AuthenticationError` (not retried)
    * 429     -> retried, honouring ``Retry-After``
    * 5xx / network errors -> retried
    * other 4xx -> :class:`ProviderError`
    """

    def __init__(
        self,
        provider: str,
        base_url: str,
        requests_per_minute: float,
        headers: dict[str, str] | None = None,
        timeout: float = 20.0,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        transport: httpx.BaseTransport | None = None,
        ops: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.provider = provider
        self.ops = ops
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.sleep = sleep
        self.limiter = RateLimiter(requests_per_minute, clock=clock, sleep=sleep)
        self.client = httpx.Client(
            base_url=base_url, headers=headers or {}, timeout=timeout, transport=transport, follow_redirects=True
        )

    def close(self) -> None:
        self.client.close()

    def _log_error(self, endpoint: str, message: str, status: int | None) -> None:
        log.warning("%s %s failed (%s): %s", self.provider, redact(endpoint), status, redact(message))
        if self.ops is not None:
            try:
                self.ops.log_api_error("provider", redact(message), self.provider, redact(endpoint), status)
            except Exception:  # pragma: no cover - logging must never break ingestion
                log.exception("failed to record provider error")

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        attempt = 0
        while True:
            self.limiter.acquire()
            try:
                resp = self.client.get(path, params=params)
            except httpx.HTTPError as exc:
                err: ProviderError = ProviderUnavailableError(f"network error: {exc}")
                retry_after = None
            else:
                if self.ops is not None:
                    try:
                        self.ops.log_api_usage("provider", self.provider, path)
                    except Exception:  # pragma: no cover
                        log.exception("failed to record provider usage")
                if resp.status_code < 400:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        err = ProviderUnavailableError(f"invalid JSON: {exc}", resp.status_code)
                        retry_after = None
                elif resp.status_code in (401, 403):
                    self._log_error(path, f"authentication failed: {resp.text[:300]}", resp.status_code)
                    raise AuthenticationError(f"{self.provider}: authentication failed", resp.status_code)
                elif resp.status_code == 429:
                    err = RateLimitError(f"rate limited: {resp.text[:300]}", 429)
                    retry_after = _retry_after(resp)
                elif resp.status_code >= 500:
                    err = ProviderUnavailableError(f"server error: {resp.text[:300]}", resp.status_code)
                    retry_after = None
                else:
                    self._log_error(path, f"client error: {resp.text[:300]}", resp.status_code)
                    raise ProviderError(f"{self.provider}: HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)

            self._log_error(path, str(err), err.status_code)
            if attempt >= self.max_retries:
                raise err
            delay = retry_after if retry_after is not None else self.backoff_base * (2**attempt)
            self.sleep(delay)
            attempt += 1


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
