import json
from datetime import datetime, timezone

import httpx
import pytest

from pit.forecasting.events import EconomicEvent, EventCalendar
from pit.forecasting.llm import prompt as P
from pit.forecasting.llm.providers import AnthropicClient, LLMError, OpenAIClient
from pit.forecasting.llm.reviewer import LLMReviewer
from pit.forecasting.models import ForecastResult
from tests.helpers import make_forecast

GOOD = {
    "directional_assessment": "UP", "confidence": 0.7, "supporting_observations": ["momentum aligned"],
    "contradictory_evidence": [], "regime_interpretation": "uptrend", "risk_factors": ["spread"],
    "recommendation": "ACCEPT", "rationale": "consistent multi-timeframe momentum",
}
NOW = datetime(2026, 1, 5, 10, 0, 5, tzinfo=timezone.utc)


def openai_transport(content, calls, status=200):
    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={
            "model": "gpt-5-mini",
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300},
        })
    return httpx.MockTransport(handler)


def reviewer(ctx, content=GOOD, calls=None, status=200, events=None):
    calls = calls if calls is not None else []
    client = OpenAIClient("sk-test", "gpt-5-mini", transport=openai_transport(content, calls, status), sleep=lambda s: None)
    return LLMReviewer(ctx, client, events=events or EventCalendar([]), clock=lambda: NOW), calls


@pytest.fixture
def llm_ctx(make_ctx):
    return make_ctx({"llm": {"enabled": True, "monthly_budget_usd": 5.0}})


def det(direction="UP", **kw):
    f, targets = make_forecast(direction, **kw)
    return ForecastResult(forecast=f.with_hash(), targets=targets)


def test_accept_creates_linked_assisted_forecast_and_logs_call(llm_ctx):
    r, calls = reviewer(llm_ctx)
    d = det("UP")
    out = r.review(d)
    a = out.forecast
    assert a.model_type == "llm_assisted" and a.parent_forecast_id == d.forecast.forecast_id
    assert a.direction == "UP" and a.target_price == d.forecast.target_price
    assert a.llm_model == "gpt-5-mini" and a.llm_prompt_version == P.PROMPT_VERSION
    assert {t.forecast_id for t in out.targets} == {a.forecast_id}
    row = llm_ctx.db.query_one("SELECT * FROM llm_calls WHERE call_id = ?", (a.llm_call_id,))
    assert row["status"] == "ok" and row["input_tokens"] == 1200 and row["output_tokens"] == 300
    assert row["cost_usd"] == pytest.approx((1200 * 0.25 + 300 * 2.0) / 1e6)
    body = calls[0]
    assert body["response_format"]["json_schema"]["strict"] is True
    # The deterministic record is untouched.
    assert d.forecast.content_hash == d.forecast.computed_hash()


def test_reject_and_contradictory_accept_become_wait(llm_ctx):
    r, _ = reviewer(llm_ctx, dict(GOOD, recommendation="REJECT"))
    out = r.review(det("UP"))
    assert out.forecast.direction == "WAIT" and out.forecast.wait_reasons == ["llm_rejected"]
    assert out.forecast.confidence == "none"
    r2, _ = reviewer(llm_ctx, dict(GOOD, directional_assessment="DOWN"))
    out2 = r2.review(det("UP", start_min=det().forecast.start_min + 15, forecast_id="other"))
    assert out2.forecast.direction == "WAIT"


def test_deterministic_wait_never_calls_llm(llm_ctx):
    r, calls = reviewer(llm_ctx)
    out = r.review(det("WAIT", candidate="UP"))
    assert calls == [] and out.forecast.direction == "WAIT"
    assert out.forecast.wait_reasons[0] == "deterministic_wait"
    assert llm_ctx.db.scalar("SELECT COUNT(*) FROM llm_calls") == 0


def test_invalid_response_is_logged_and_produces_no_record(llm_ctx):
    bad = dict(GOOD)
    bad.pop("rationale")
    r, _ = reviewer(llm_ctx, bad)
    assert r.review(det("UP")) is None
    assert llm_ctx.db.scalar("SELECT status FROM llm_calls") == "invalid_response"


def test_budget_and_unknown_pricing_block_calls(make_ctx):
    ctx = make_ctx({"llm": {"enabled": True, "monthly_budget_usd": 0.0}})
    r, calls = reviewer(ctx)
    assert r.review(det("UP")) is None and calls == []
    assert ctx.db.scalar("SELECT status FROM llm_calls") == "budget_exceeded"
    ctx2 = make_ctx({"llm": {"enabled": True, "model": "unpriced-model"}})
    client = OpenAIClient("k", "unpriced-model", transport=openai_transport(GOOD, calls))
    r2 = LLMReviewer(ctx2, client, events=EventCalendar([]), clock=lambda: NOW)
    assert r2.review(det("UP")) is None and calls == []
    assert ctx2.db.scalar("SELECT status FROM llm_calls") == "pricing_unknown"


def test_identical_input_served_from_cache(llm_ctx):
    calls = []
    r, _ = reviewer(llm_ctx, calls=calls)
    d = det("UP")
    r.review(d)
    # A second reviewer for a different run gets the identical payload -> cache hit, no API call.
    r2, _ = reviewer(llm_ctx, calls=calls)
    f2 = d.forecast.model_copy(update={"forecast_id": "fc_other", "run_id": "t2"})
    out = r2.review(ForecastResult(forecast=f2.with_hash(), targets=[t.model_copy(update={"forecast_id": "fc_other"}) for t in d.targets]))
    assert len(calls) == 1 and out is not None
    assert llm_ctx.db.scalar("SELECT SUM(cached) FROM llm_calls") == 1


def test_server_error_retried_then_logged(llm_ctx):
    calls = []
    r, _ = reviewer(llm_ctx, calls=calls, status=500)
    assert r.review(det("UP")) is None
    assert len(calls) == 2  # one retry
    assert llm_ctx.db.scalar("SELECT status FROM llm_calls") == "error"
    assert llm_ctx.db.scalar("SELECT COUNT(*) FROM api_errors WHERE component = 'llm'") == 1


def test_anthropic_client_forced_tool_use():
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "claude-test", "content": [{"type": "text", "text": "ok"},
                                                {"type": "tool_use", "name": "submit_review", "input": GOOD}],
            "usage": {"input_tokens": 900, "output_tokens": 200},
        })

    c = AnthropicClient("key", "claude-test", transport=httpx.MockTransport(handler))
    resp = c.complete("sys", "user", {"type": "object"})
    assert resp.content == GOOD and resp.input_tokens == 900
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "submit_review"}


def test_client_error_not_retried():
    calls = []
    c = OpenAIClient("k", "m", transport=openai_transport(GOOD, calls, status=401), sleep=lambda s: None)
    with pytest.raises(LLMError):
        c.complete("s", "u", {})
    assert len(calls) == 1


def test_prompt_input_is_structured_and_only_verified_events(llm_ctx):
    d = det("UP")
    events = EventCalendar([EconomicEvent("US CPI", d.forecast.slot_min + 120, 30, 30)])
    payload = P.build_input(d.forecast, None, events)
    assert payload["verified_economic_events"] == [{"name": "US CPI", "time_utc": "2026-01-05T12:00:00Z"}]
    assert payload["deterministic_signal"]["heuristic_p_hit_uncalibrated"] == d.forecast.p_hit
    text = P.render_user_message(payload)
    assert len(text) < 6000  # compact derived information, not raw history


def test_from_context_without_key_disables_gracefully(llm_ctx):
    assert LLMReviewer.from_context(llm_ctx) is None
    assert llm_ctx.db.scalar("SELECT COUNT(*) FROM api_errors WHERE component = 'llm'") == 1
