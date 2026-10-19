"""PIT — Predictive Intelligence Trading.

Deterministic XAU/USD forecasting and research platform. Research and paper trading only:
no brokerage integration and no live order placement exist in this codebase.
"""

__version__ = "1.0.0"

# Version of the deterministic forecasting logic. Bump whenever code changes could alter a
# forecast for identical inputs and configuration; it is stored on every forecast record.
MODEL_VERSION = "pit-det-1.0.0"
