from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from macro_monitor.alerts import (
    FedWatchState,
    bp_change,
    classify_macro,
    fedwatch_repricing_alerts,
    severity_for,
    threshold_triggered,
    yield_alerts,
)
from macro_monitor.config import EmailConfig, Thresholds
from macro_monitor.models import Observation
from macro_monitor.parsers import parse_dxy_index_html, parse_fedwatch_html, parse_treasury_yield_html
from macro_monitor.storage import MacroStore


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_fedwatch_parser_extracts_nearest_meeting_and_full_distribution() -> None:
    snapshot = parse_fedwatch_html(
        _fixture("fedwatch.html"),
        today=datetime(2026, 8, 28).date(),
        timestamp_utc="2026-08-28T01:02:03.000Z",
    )
    assert snapshot.meeting_date == "2026-09-16"
    assert snapshot.current_target_range == "4.25-4.50"
    assert [(x.target_range, x.probability) for x in snapshot.probabilities] == [
        ("3.75-4.00", 12.4),
        ("4.00-4.25", 61.8),
        ("4.25-4.50", 25.3),
        ("4.50-4.75", 0.5),
    ]
    assert snapshot.cut_probability == pytest.approx(74.2)
    assert snapshot.hold_probability == pytest.approx(25.3)
    expected_rate = (3.875 * 12.4 + 4.125 * 61.8 + 4.375 * 25.3 + 4.625 * 0.5) / 100.0
    assert snapshot.expected_rate == pytest.approx(expected_rate)
    expected_rows = [row for row in snapshot.observations() if row.metric == "fedwatch_expected_rate"]
    assert len(expected_rows) == 1
    assert expected_rows[0].value == pytest.approx(expected_rate)


def test_treasury_parser_supports_investing_and_legacy_dom() -> None:
    assert parse_treasury_yield_html(_fixture("treasury_2y.html"), term="2y") == pytest.approx(4.171)
    assert parse_treasury_yield_html(_fixture("treasury_10y.html"), term="10y") == pytest.approx(4.261)


def test_dxy_parser_extracts_public_quote_value() -> None:
    assert parse_dxy_index_html(_fixture("dxy.html")) == pytest.approx(101.437)


def test_bp_calculation() -> None:
    assert bp_change(4.18, 4.23) == pytest.approx(-5.0)


def test_threshold_detection_is_inclusive() -> None:
    assert threshold_triggered(-5.0, 5.0)
    assert not threshold_triggered(4.99, 5.0)


def test_yield_threshold_alert_key() -> None:
    thresholds = Thresholds()
    y_alerts = yield_alerts("us2y_yield", 4.171, {15: 4.232}, thresholds)
    assert y_alerts[0].key == "US2Y_15M_DOWN"
    assert y_alerts[0].detail.endswith("-6.1 bp, 15m)")


def test_hawkish_fedwatch_alert_uses_hike_driver_and_expected_rate_delta() -> None:
    previous = FedWatchState(0.0, 66.3, 33.7, 3.709)
    current = FedWatchState(0.0, 43.0, 57.0, 3.768)

    alerts = fedwatch_repricing_alerts(current, {15: previous}, Thresholds())

    assert len(alerts) == 1
    assert alerts[0].key == "FEDWATCH_HIKE_15M_UP"
    assert alerts[0].direction == "hawkish"
    assert alerts[0].title == "FedWatch hawkish repricing: Hike up"
    assert "Driver Hike: 33.7% -> 57.0% (+23.3 pct, 15m)" in alerts[0].detail
    assert "Cut / Hold / Hike: 0.0% / 66.3% / 33.7% -> 0.0% / 43.0% / 57.0%" in alerts[0].detail
    assert "Expected Rate: 3.709% -> 3.768%" in alerts[0].detail
    assert "ΔExpected Rate: +5.9 bp" in alerts[0].detail


def test_later_hawkish_breakthrough_escalates_and_bypasses_cooldown(tmp_path: Path) -> None:
    first = datetime(2026, 8, 28, 14, 26, tzinfo=timezone.utc)
    continuation = datetime(2026, 8, 28, 14, 44, tzinfo=timezone.utc)
    early_severity = severity_for(-7.8, 5.0)
    continuation_severity = severity_for(-11.7, 5.0)

    assert early_severity == 2
    assert continuation_severity == 3
    with MacroStore(tmp_path / "macro.sqlite") as store:
        assert store.cooldown_allows("FEDWATCH_HIKE_15M_UP", early_severity, first, 1800)
        assert store.cooldown_allows(
            "FEDWATCH_HIKE_15M_UP",
            continuation_severity,
            continuation,
            1800,
        )


def test_dovish_fedwatch_alert_uses_cut_driver() -> None:
    previous = FedWatchState(20.0, 60.0, 20.0, 3.625)
    current = FedWatchState(30.0, 60.0, 10.0, 3.575)

    alerts = fedwatch_repricing_alerts(current, {15: previous}, Thresholds())

    assert alerts[0].key == "FEDWATCH_CUT_15M_UP"
    assert alerts[0].direction == "dovish"
    assert "Driver Cut: 20.0% -> 30.0% (+10.0 pct, 15m)" in alerts[0].detail
    assert "ΔExpected Rate: -5.0 bp" in alerts[0].detail


def test_dovish_classification_and_confirmation() -> None:
    assert classify_macro(
        fedwatch_change_pct=8.6,
        fedwatch_threshold_pct=5.0,
        us2y_change_bp=-6.1,
        us2y_threshold_bp=5.0,
    ) == "STRONG DOVISH REPRICING"
    assert classify_macro(
        fedwatch_change_pct=None,
        fedwatch_threshold_pct=5.0,
        us2y_change_bp=-6.1,
        us2y_threshold_bp=5.0,
    ) == "DOVISH REPRICING"


def test_hawkish_classification_and_confirmation() -> None:
    assert classify_macro(
        fedwatch_change_pct=-7.0,
        fedwatch_threshold_pct=5.0,
        us2y_change_bp=6.0,
        us2y_threshold_bp=5.0,
    ) == "STRONG HAWKISH REPRICING"
    assert classify_macro(
        fedwatch_change_pct=-7.0,
        fedwatch_threshold_pct=5.0,
        us2y_change_bp=None,
        us2y_threshold_bp=5.0,
    ) == "HAWKISH REPRICING"


def test_sqlite_persistence_and_time_window_lookup(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    rows = [
        Observation(_iso(now - timedelta(minutes=15, seconds=10)), "fixture", "us2y_yield", None, None, 4.232),
        Observation(_iso(now - timedelta(minutes=14, seconds=50)), "fixture", "us2y_yield", None, None, 4.229),
        Observation(_iso(now), "fixture", "us2y_yield", None, None, 4.171),
    ]
    with MacroStore(tmp_path / "macro.sqlite", retention_days=365) as store:
        assert store.insert_observations(rows) == 3
        assert store.count_observations() == 3
        previous = store.window_value("us2y_yield", _iso(now), 15, tolerance_seconds=30)
        assert previous is not None
        assert previous.value == pytest.approx(4.229)


def test_time_window_lookup_respects_meeting_date(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    rows = [
        Observation(_iso(now - timedelta(minutes=15)), "fixture", "fedwatch_cut_probability", "2026-09-16", None, 65.6),
        Observation(_iso(now - timedelta(minutes=15)), "fixture", "fedwatch_cut_probability", "2026-10-28", None, 80.0),
    ]
    with MacroStore(tmp_path / "macro.sqlite") as store:
        store.insert_observations(rows)
        previous = store.window_value(
            "fedwatch_cut_probability", _iso(now), 15, meeting_date="2026-09-16", tolerance_seconds=5
        )
        assert previous is not None and previous.value == pytest.approx(65.6)


def test_cooldown_deduplicates_but_allows_severity_breakthrough(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    with MacroStore(tmp_path / "macro.sqlite") as store:
        assert store.cooldown_allows("US2Y_15M_DOWN", 1, now, 1800)
        assert not store.cooldown_allows("US2Y_15M_DOWN", 1, now + timedelta(minutes=10), 1800)
        assert store.cooldown_allows("US2Y_15M_DOWN", 2, now + timedelta(minutes=11), 1800)
        assert store.cooldown_allows("US2Y_15M_DOWN", 2, now + timedelta(minutes=42), 1800)


def test_existing_email_env_names_are_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL_SENDER", "monitor@qq.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "smtp-authorization-code")
    monkeypatch.setenv("EMAIL_RECEIVER", "one@example.com;two@example.com")
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    monkeypatch.delenv("EMAIL_USERNAME", raising=False)
    monkeypatch.delenv("EMAIL_TO", raising=False)
    config = EmailConfig.from_env(cli_enabled=True)
    assert config.configured
    assert config.smtp_host == "smtp.qq.com"
    assert config.smtp_port == 465
    assert config.recipients == ("one@example.com", "two@example.com")
