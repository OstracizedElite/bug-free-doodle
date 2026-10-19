"""Strict response schema for the optional LLM reviewer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directional_assessment: Literal["UP", "DOWN", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_observations: list[str] = Field(max_length=8)
    contradictory_evidence: list[str] = Field(max_length=8)
    regime_interpretation: str = Field(max_length=600)
    risk_factors: list[str] = Field(max_length=8)
    recommendation: Literal["ACCEPT", "REJECT"]
    rationale: str = Field(max_length=1200)


# JSON Schema in the strict subset accepted by structured-output APIs
# (every property required, no additional properties).
JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "directional_assessment",
        "confidence",
        "supporting_observations",
        "contradictory_evidence",
        "regime_interpretation",
        "risk_factors",
        "recommendation",
        "rationale",
    ],
    "properties": {
        "directional_assessment": {"type": "string", "enum": ["UP", "DOWN", "NEUTRAL"]},
        "confidence": {"type": "number", "description": "0 to 1"},
        "supporting_observations": {"type": "array", "items": {"type": "string"}},
        "contradictory_evidence": {"type": "array", "items": {"type": "string"}},
        "regime_interpretation": {"type": "string"},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
        "rationale": {"type": "string"},
    },
}
