"""Optional LLM review producing a separate, linked ``llm_assisted`` forecast.

Policy (prompt ``llm-v1``):

* The deterministic forecast is stored first and is never modified.
* Deterministic WAIT -> the LLM-assisted forecast is also WAIT (reason ``deterministic_wait``)
  and no API call is made. In particular the LLM can never override data-integrity WAITs.
* Deterministic UP/DOWN -> the LLM may ACCEPT (assisted forecast keeps the direction,
  targets and confidence) or REJECT (assisted forecast is WAIT with ``llm_rejected``).
  An ACCEPT whose own assessment points the opposite way is treated as REJECT.
* If the call is skipped (frequency setting), over budget, unpriced, fails, or returns an
  invalid response, no assisted forecast is recorded for that slot - its outcome would be
  unknown - but the attempt is logged in ``llm_calls``.
* Every call records provider, model, prompt version, input snapshot, response, tokens,
  estimated cost, latency and errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Callable

from pydantic import ValidationError

from pit.app.context import AppContext
from pit.common.timeutils import ceil_minute, iso, iso_min, to_epoch_min, utcnow
from pit.config.settings import LLMConfig
from pit.forecasting.events import EventCalendar
from pit.forecasting.llm import prompt as P
from pit.forecasting.llm.providers import AnthropicClient, LLMClient, LLMError, OpenAIClient
from pit.forecasting.llm.schema import JSON_SCHEMA, LLMAssessment
from pit.forecasting.models import Direction, Forecast, ForecastResult, ForecastTarget, WaitReason, forecast_id_for

log = logging.getLogger(__name__)
MODEL_TYPE = "llm_assisted"


def estimate_cost(cfg: LLMConfig, model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    price = cfg.pricing_usd_per_mtok.get(model)
    if price is None:
        return None
    return ((input_tokens or 0) * price.input + (output_tokens or 0) * price.output) / 1_000_000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough pre-call estimate (~4 characters per token)


class LLMReviewer:
    def __init__(self, ctx: AppContext, client: LLMClient, events: EventCalendar | None = None,
                 clock: Callable[[], datetime] = utcnow, record_clock: Callable[[], datetime] | None = None):
        self.ctx = ctx
        self.cfg = ctx.settings.llm
        self.client = client
        self.clock = clock  # wall clock: call log timestamps and monthly budget
        self.record_clock = record_clock or clock  # forecast record time (simulated in backtests)
        self.events = events or EventCalendar.from_file(ctx.settings.resolve_path(ctx.settings.events.file),
                                                        ctx.settings.events)
        self.config_version = ctx.settings.forecast_config_version

    @classmethod
    def from_context(cls, ctx: AppContext) -> "LLMReviewer | None":
        cfg = ctx.settings.llm
        if not cfg.enabled:
            return None
        env = "OPENAI_API_KEY" if cfg.provider == "openai" else "ANTHROPIC_API_KEY"
        key = os.environ.get(env, "").strip()
        if not key:
            log.error("LLM enabled but %s is not set; continuing without LLM review", env)
            ctx.ops.log_api_error("llm", f"{env} not set; LLM review skipped", cfg.provider)
            return None
        kw = dict(timeout=cfg.timeout_seconds, max_output_tokens=cfg.max_output_tokens)
        client = OpenAIClient(key, cfg.model, **kw) if cfg.provider == "openai" else AnthropicClient(key, cfg.model, **kw)
        return cls(ctx, client)

    # ------------------------------------------------------------------ helpers
    def _log_call(self, f: Forecast, status: str, cache_key: str, payload: dict, response: dict | None = None,
                  error: str | None = None, tokens: tuple[int | None, int | None] = (None, None),
                  cost: float = 0.0, latency: float | None = None, cached: bool = False) -> str:
        call_id = "llm_" + uuid.uuid4().hex[:16]
        self.ctx.llm_calls.insert({
            "call_id": call_id, "forecast_id": f.forecast_id, "created_at": iso(self.clock()),
            "provider": self.client.provider, "model": self.client.model, "prompt_version": P.PROMPT_VERSION,
            "cache_key": cache_key, "cached": int(cached), "input_json": json.dumps(payload, sort_keys=True, default=str),
            "response_json": json.dumps(response) if response is not None else None, "status": status,
            "error": error, "input_tokens": tokens[0], "output_tokens": tokens[1], "cost_usd": cost, "latency_ms": latency,
        })
        if status not in ("ok",):
            self.ctx.ops.log_api_error("llm", f"{status}: {error or ''}"[:500], self.client.provider)
        return call_id

    def _month_spend(self) -> float:
        now = self.clock()
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.ctx.llm_calls.spend_since(iso(start))

    def _assisted(self, f: Forecast, targets: list[ForecastTarget], direction: str, reasons: list[str],
                  extra: dict, call_id: str | None) -> ForecastResult:
        created = self.record_clock()
        start_min = max(f.slot_min, to_epoch_min(ceil_minute(created)), f.start_min)
        fid = forecast_id_for(f.origin, f.run_id, f.instrument, f.slot_min, f.horizon_minutes, MODEL_TYPE,
                              self.config_version)
        comps = f.components
        comps["llm"] = extra
        accepted = direction in ("UP", "DOWN")
        record = f.model_copy(update={
            "forecast_id": fid,
            "model_type": MODEL_TYPE,
            "parent_forecast_id": f.forecast_id,
            "created_at": iso(created),
            "start_ts": iso_min(start_min),
            "start_min": start_min,
            "expiry_ts": iso_min(start_min + f.horizon_minutes),
            "expiry_min": start_min + f.horizon_minutes,
            "direction": direction,
            "confidence": f.confidence if accepted else "none",
            "wait_reasons_json": json.dumps(reasons),
            "primary_wait_reason": reasons[0] if not accepted and reasons else None,
            "components_json": json.dumps(comps, sort_keys=True),
            "llm_provider": self.client.provider,
            "llm_model": self.client.model,
            "llm_prompt_version": P.PROMPT_VERSION,
            "llm_call_id": call_id,
            "content_hash": "",
        })
        new_targets = [t.model_copy(update={"forecast_id": fid}) for t in targets]
        return ForecastResult(forecast=record, targets=new_targets, snapshot=None)

    # ------------------------------------------------------------------ main entry
    def review(self, det: ForecastResult, frames=None) -> ForecastResult | None:
        f = det.forecast
        if f.model_type != "deterministic":
            return None
        targets = det.targets or self.ctx.forecasts.targets(f.forecast_id)
        if f.direction == Direction.WAIT.value:
            # V1 never consults the LLM on WAITs: it may filter signals, never create them.
            return self._assisted(f, targets, Direction.WAIT.value,
                                  [WaitReason.DETERMINISTIC_WAIT.value] + f.wait_reasons, {"consulted": False}, None)
        interval = self.ctx.settings.forecasting.interval_minutes
        if (f.slot_min // interval) % self.cfg.every_n_slots != 0:
            return None

        payload = P.build_input(f, det.snapshot, self.events)
        user = P.render_user_message(payload)
        cache_key = hashlib.sha256(
            json.dumps([self.client.provider, self.client.model, P.PROMPT_VERSION, payload], sort_keys=True,
                       default=str).encode()
        ).hexdigest()

        content = self.ctx.llm_calls.cached_response(cache_key) if self.cfg.cache_responses else None
        cached = content is not None
        tokens: tuple[int | None, int | None] = (None, None)
        cost, latency = 0.0, None
        if not cached:
            est = estimate_cost(self.cfg, self.client.model, estimate_tokens(P.SYSTEM_PROMPT + user),
                                self.cfg.max_output_tokens)
            if est is None:
                self._log_call(f, "pricing_unknown", cache_key, payload,
                               error=f"no pricing configured for model {self.client.model}; budget cannot be enforced")
                return None
            if self._month_spend() + est > self.cfg.monthly_budget_usd:
                self._log_call(f, "budget_exceeded", cache_key, payload,
                               error=f"monthly budget {self.cfg.monthly_budget_usd} USD would be exceeded")
                return None
            try:
                resp = self.client.complete(P.SYSTEM_PROMPT, user, JSON_SCHEMA)
            except LLMError as exc:
                self._log_call(f, "error", cache_key, payload, error=str(exc))
                return None
            content = resp.content
            tokens = (resp.input_tokens, resp.output_tokens)
            cost = estimate_cost(self.cfg, self.client.model, *tokens) or 0.0
            latency = resp.latency_ms
        try:
            assessment = LLMAssessment.model_validate(content)
        except ValidationError as exc:
            self._log_call(f, "invalid_response", cache_key, payload, response=content, error=str(exc)[:500],
                           tokens=tokens, cost=cost, latency=latency, cached=cached)
            return None
        call_id = self._log_call(f, "ok", cache_key, payload, response=assessment.model_dump(), tokens=tokens,
                                 cost=cost, latency=latency, cached=cached)
        opposite = "DOWN" if f.direction == "UP" else "UP"
        accepted = assessment.recommendation == "ACCEPT" and assessment.directional_assessment != opposite
        extra = {"consulted": True, "recommendation": assessment.recommendation,
                 "assessment": assessment.directional_assessment, "confidence": assessment.confidence,
                 "accepted": accepted}
        if accepted:
            return self._assisted(f, targets, f.direction, [], extra, call_id)
        return self._assisted(f, targets, Direction.WAIT.value, [WaitReason.LLM_REJECTED.value], extra, call_id)
