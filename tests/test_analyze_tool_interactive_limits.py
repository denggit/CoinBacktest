from __future__ import annotations

from analyze_tool.data_service import (
    LoadRequest,
    estimate_requested_bars,
    resolve_adaptive_timeframe,
)


def test_two_year_1m_request_uses_adaptive_15m_overview_without_rejection() -> None:
    req = LoadRequest(
        data_type="trade_bar",
        timeframe="1m",
        start="2024-01-01 00:00:00",
        end="2025-12-31 23:59:00",
        limit=2_000_000,
    )
    effective, source_rows, display_rows = resolve_adaptive_timeframe(req)
    assert source_rows is not None and source_rows > 1_000_000
    assert effective.timeframe == "15m"
    assert display_rows is not None and display_rows < 100_000
    assert effective.start == req.start and effective.end == req.end


def test_short_1m_request_keeps_requested_resolution() -> None:
    req = LoadRequest(
        data_type="trade_bar",
        timeframe="1m",
        start="2024-01-01 00:00:00",
        end="2024-01-31 23:59:00",
        limit=2_000_000,
    )
    effective, source_rows, display_rows = resolve_adaptive_timeframe(req)
    assert estimate_requested_bars(req) == source_rows == display_rows
    assert effective.timeframe == "1m"
