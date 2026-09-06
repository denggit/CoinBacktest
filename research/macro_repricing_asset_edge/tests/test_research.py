from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.macro_repricing_asset_edge.config import SignalThresholds
from research.macro_repricing_asset_edge.data_sources import (
    build_monitor_panel,
    parse_cnbc_chart_payload,
    parse_okx_candles,
    parse_yahoo_chart_payload,
)
from research.macro_repricing_asset_edge.event_study import (
    align_intraday_events,
    benjamini_hochberg,
    daily_forward_returns,
)
from research.macro_repricing_asset_edge.features import (
    asof_change,
    build_intraday_features,
    classify_intraday_row,
    select_intraday_events,
)
from research.macro_repricing_asset_edge.scheduled_events import (
    align_scheduled_asset_responses,
    build_scheduled_macro_signals,
    leave_one_event_out_summary,
)


FIXTURES = Path(__file__).with_name("fixtures")


def _observation(
    timestamp: str,
    metric: str,
    value: float,
    *,
    target_range: str | None = None,
    meeting: str | None = "2026-09-16",
) -> dict[str, object]:
    return {
        "timestamp_utc": pd.Timestamp(timestamp),
        "source": "fixture",
        "metric": metric,
        "meeting_date": meeting if metric.startswith("fedwatch_") else None,
        "target_range": target_range,
        "value": value,
        "status": "ok",
    }


def test_expected_rate_is_rebuilt_from_full_fedwatch_distribution() -> None:
    rows = [
        _observation("2026-08-28T14:00:00Z", "fedwatch_cut_probability", 20.0),
        _observation("2026-08-28T14:00:00Z", "fedwatch_hold_probability", 70.0),
        _observation("2026-08-28T14:00:00Z", "fedwatch_hike_probability", 10.0),
        _observation("2026-08-28T14:00:00Z", "fedwatch_target_probability", 20.0, target_range="3.25-3.50"),
        _observation("2026-08-28T14:00:00Z", "fedwatch_target_probability", 70.0, target_range="3.50-3.75"),
        _observation("2026-08-28T14:00:00Z", "fedwatch_target_probability", 10.0, target_range="3.75-4.00"),
    ]
    panel = build_monitor_panel(pd.DataFrame(rows))
    expected = 3.375 * 0.20 + 3.625 * 0.70 + 3.875 * 0.10
    assert panel.iloc[0]["fedwatch_expected_rate_pct"] == pytest.approx(expected)
    assert panel.iloc[0]["fedwatch_policy_bias_pct"] == pytest.approx(10.0)


def test_asof_change_never_uses_future_of_requested_lookback() -> None:
    index = pd.to_datetime(
        ["2026-08-28T13:44:50Z", "2026-08-28T13:45:10Z", "2026-08-28T14:00:00Z"],
        utc=True,
    )
    series = pd.Series([4.00, 4.50, 4.10], index=index)
    change = asof_change(series, 15, tolerance_minutes=1)
    # Target is 13:45:00. The 13:45:10 observation is in its future and must
    # not be selected; the valid prior observation is 13:44:50.
    assert change.iloc[-1] == pytest.approx(0.10)


def test_hawkish_classification_uses_bias_and_us2y_direction() -> None:
    row = pd.Series(
        {
            "fedwatch_bias_change_15m_pct": -7.0,
            "us2y_change_15m_bp": 6.0,
            "us10y_change_15m_bp": 5.5,
        }
    )
    result = classify_intraday_row(row, SignalThresholds())
    assert result["regime"] == "hawkish"
    assert result["severity"] == 3


def test_event_cooldown_preserves_severity_breakthrough() -> None:
    index = pd.to_datetime(
        ["2026-08-28T14:00:00Z", "2026-08-28T14:10:00Z", "2026-08-28T14:20:00Z"], utc=True
    )
    features = pd.DataFrame(
        {
            "us2y_change_5m_bp": [3.1, 3.2, 3.3],
            "fedwatch_bias_change_15m_pct": [np.nan, np.nan, -7.0],
            "us2y_change_15m_bp": [np.nan, np.nan, 6.0],
        },
        index=index,
    )
    events = select_intraday_events(features, SignalThresholds(), cooldown_minutes=30)
    assert len(events) == 2
    assert events.iloc[0]["severity"] == 1
    assert events.iloc[1]["severity"] == 3


def test_intraday_alignment_enters_strictly_after_signal() -> None:
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=pd.to_datetime(
            ["2026-08-28T14:00:00Z", "2026-08-28T14:01:00Z", "2026-08-28T14:05:00Z"], utc=True
        ),
    )
    events = pd.DataFrame(
        [{"timestamp_utc": pd.Timestamp("2026-08-28T14:00:00Z"), "regime": "hawkish", "severity": 1, "score": 1.0}]
    )
    aligned = align_intraday_events(events, {"SOXX": bars}, horizons_minutes=(5,))
    assert aligned.iloc[0]["entry_time_utc"] == pd.Timestamp("2026-08-28T14:01:00Z")
    assert aligned.iloc[0]["entry_price"] == pytest.approx(101.0)


def test_daily_alignment_rejects_signals_before_asset_history() -> None:
    macro = pd.DataFrame(
        [
            {"regime": "hawkish", "severity": 1, "score": 1.0},
            {"regime": "dovish", "severity": 1, "score": 1.0},
        ],
        index=pd.to_datetime(["1976-01-05", "2020-01-02"]),
    )
    asset = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    aligned = daily_forward_returns(macro, {"SOXX": asset}, horizons_sessions=(1,))
    assert len(aligned) == 1
    assert aligned.iloc[0]["signal_date"] == pd.Timestamp("2020-01-02")
    assert aligned.iloc[0]["entry_session"] == pd.Timestamp("2020-01-03")


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    p_values = pd.Series([0.001, 0.02, 0.04, np.nan])
    adjusted = benjamini_hochberg(p_values)
    valid = adjusted.dropna().to_numpy()
    assert np.all(np.diff(valid) >= 0)
    assert adjusted.iloc[0] == pytest.approx(0.003)


def test_free_intraday_payload_parsers_use_fixed_fixtures() -> None:
    yahoo_payload = json.loads((FIXTURES / "yahoo_chart_5m.json").read_text(encoding="utf-8"))
    cnbc_payload = json.loads((FIXTURES / "cnbc_chart_5m.json").read_text(encoding="utf-8"))
    okx_payload = json.loads((FIXTURES / "okx_candles.json").read_text(encoding="utf-8"))

    yahoo = parse_yahoo_chart_payload(yahoo_payload, "SOXX")
    cnbc = parse_cnbc_chart_payload(cnbc_payload, "US2Y")
    okx = parse_okx_candles(okx_payload["data"], "ETH-USDT-SWAP")

    assert len(yahoo) == 2
    assert yahoo.index.tz is not None
    assert yahoo.iloc[-1]["close"] == pytest.approx(100.3)
    assert len(cnbc) == 2
    assert cnbc.iloc[-1]["close"] == pytest.approx(4.218)
    assert len(okx) == 2
    assert okx.index.is_monotonic_increasing
    assert okx.iloc[-1]["close"] == pytest.approx(3005.0)


def test_scheduled_proxy_signal_and_execution_delay_are_causal() -> None:
    event_time = pd.Timestamp("2026-08-28T14:00:00Z")
    events = pd.DataFrame(
        [
            {
                "event_id": "fixture_event",
                "event_type": "CPI",
                "event_time_utc": event_time,
                "event_time_bjt": event_time.tz_convert("Asia/Shanghai"),
            }
        ]
    )
    index = pd.to_datetime(
        ["2026-08-28T13:55:00Z", "2026-08-28T14:00:00Z", "2026-08-28T14:05:00Z"],
        utc=True,
    )

    def bars(closes: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes}, index=index
        )

    macro = {
        "ZQ_POST_FOMC": bars([96.00, 95.96, 95.95]),
        "ZT": bars([102.0, 101.9, 101.8]),
        "US10Y_YAHOO": bars([4.20, 4.24, 4.25]),
        "DXY": bars([100.0, 100.2, 100.3]),
        "US2Y_EXACT": pd.DataFrame(),
        "US10Y_EXACT": pd.DataFrame(),
    }
    signals = build_scheduled_macro_signals(events, macro, signal_delays_minutes=(5,))
    assert signals.iloc[0]["regime"] == "proxy_hawkish"
    assert signals.iloc[0]["zq_post_fomc_implied_rate_change_bp"] == pytest.approx(4.0)

    asset_index = pd.to_datetime(
        [
            "2026-08-28T14:05:00Z",
            "2026-08-28T14:10:00Z",
            "2026-08-28T14:15:00Z",
            "2026-08-28T14:20:00Z",
        ],
        utc=True,
    )
    asset = pd.DataFrame(
        {
            "open": [100.0, 99.0, 98.0, 97.0],
            "high": [100.2, 99.2, 98.2, 97.2],
            "low": [99.5, 98.5, 97.5, 96.5],
            "close": [99.8, 98.8, 97.8, 96.8],
        },
        index=asset_index,
    )
    aligned = align_scheduled_asset_responses(
        signals,
        {"SOXX": asset},
        execution_delays_minutes=(0,),
        horizons_minutes=(5,),
    )
    # The signal is known at 14:05; strict-after entry must be 14:10.
    assert aligned.iloc[0]["entry_time_utc"] == pd.Timestamp("2026-08-28T14:10:00Z")
    assert aligned.iloc[0]["entry_price"] == pytest.approx(99.0)


def test_leave_one_event_out_reports_sign_stability() -> None:
    responses = pd.DataFrame(
        {
            "asset": ["QQQ"] * 3,
            "regime": ["proxy_dovish"] * 3,
            "signal_delay_minutes": [5] * 3,
            "execution_delay_minutes": [0] * 3,
            "horizon_minutes": [60] * 3,
            "event_id": ["a", "b", "c"],
            "net_return_5bp_pct": [0.2, 0.3, 0.4],
        }
    )
    summary = leave_one_event_out_summary(responses)
    assert bool(summary.iloc[0]["same_sign_all_loo"])
    assert summary.iloc[0]["loo_min_mean_pct"] > 0
