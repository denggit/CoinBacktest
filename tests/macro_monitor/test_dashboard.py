from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from macro_monitor.dashboard import DashboardDataService, MacroDashboardServer
from macro_monitor.models import Observation
from macro_monitor.storage import MacroStore


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _seed_dashboard(path: Path, now: datetime) -> None:
    old = now - timedelta(minutes=15)
    meeting = "2026-09-16"
    rows = [
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 65.6),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 29.4),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 5.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 74.2),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 25.8),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 0.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.50-3.75", 25.8),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.25-3.50", 74.2),
        Observation(_iso(old), "cnbc_us2y", "us2y_yield", None, None, 4.232),
        Observation(_iso(now), "cnbc_us2y", "us2y_yield", None, None, 4.171),
        Observation(_iso(old), "cnbc_us10y", "us10y_yield", None, None, 4.500),
        Observation(_iso(now), "cnbc_us10y", "us10y_yield", None, None, 4.450),
        Observation(_iso(old), "cnbc_us10y-cnbc_us2y", "us10y_2y_spread", None, None, 0.268),
        Observation(_iso(now), "cnbc_us10y-cnbc_us2y", "us10y_2y_spread", None, None, 0.279),
        Observation(_iso(old), "cnbc_dxy", "dxy_index", None, None, 101.000),
        Observation(_iso(now), "cnbc_dxy", "dxy_index", None, None, 100.600),
    ]
    with MacroStore(path) as store:
        store.insert_observations(rows)


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_macro_dashboard")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_dashboard_snapshot_builds_strong_dovish_live_guidance(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "macro.sqlite"
    _seed_dashboard(db_path, now)
    snapshot = DashboardDataService(db_path, now_provider=lambda: now).snapshot()

    assert snapshot["database"]["ready"] is True
    assert snapshot["regime"]["label"] == "STRONG DOVISH"
    assert snapshot["regime"]["severity"] == 3
    assert snapshot["metrics"]["us2y_yield"]["changes"]["15m"] == pytest.approx(-6.1)
    assert snapshot["metrics"]["fedwatch_cut_probability"]["changes"]["15m"] == pytest.approx(8.6)
    assert snapshot["fedwatch"]["bias_changes"]["15m"] == pytest.approx(13.6)
    assert snapshot["fedwatch"]["current_target_range"] == "3.50-3.75"
    assert snapshot["fedwatch"]["expected_move_bp"] == pytest.approx(-18.55)
    assert snapshot["metrics"]["dxy_index"]["changes"]["15m"] == pytest.approx(-0.3960396)
    assert snapshot["meeting"]["most_likely_range"] == "3.25-3.50"
    assert len(snapshot["guidance"]) >= 3


def test_dashboard_detects_hawkish_repricing_when_cut_stays_zero_and_hike_rises(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    old = now - timedelta(minutes=15)
    meeting = "2026-09-16"
    db_path = tmp_path / "macro.sqlite"
    rows = [
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 0.0),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 80.0),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 20.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 0.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 65.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 35.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.50-3.75", 65.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.75-4.00", 35.0),
        Observation(_iso(old), "cnbc_us2y", "us2y_yield", None, None, 4.232),
        Observation(_iso(now), "cnbc_us2y", "us2y_yield", None, None, 4.232),
        Observation(_iso(old), "cnbc_us10y", "us10y_yield", None, None, 4.500),
        Observation(_iso(now), "cnbc_us10y", "us10y_yield", None, None, 4.500),
    ]
    with MacroStore(db_path) as store:
        store.insert_observations(rows)

    snapshot = DashboardDataService(db_path, now_provider=lambda: now).snapshot()

    assert snapshot["metrics"]["fedwatch_cut_probability"]["changes"]["15m"] == pytest.approx(0.0)
    assert snapshot["fedwatch"]["hike_probability"] == pytest.approx(35.0)
    assert snapshot["fedwatch"]["policy_bias"] == pytest.approx(-35.0)
    assert snapshot["fedwatch"]["bias_changes"]["15m"] == pytest.approx(-15.0)
    assert snapshot["fedwatch"]["skew_label"] == "HAWKISH SKEW"
    assert snapshot["fedwatch"]["expected_move_bp"] == pytest.approx(8.75)
    assert snapshot["regime"]["label"] == "HAWKISH REPRICING"
    assert snapshot["regime"]["window"] == "15m"


def test_automatic_regime_outputs_hawkish_flattening_with_dxy_pending(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    old = now - timedelta(minutes=15)
    meeting = "2026-09-16"
    db_path = tmp_path / "macro.sqlite"
    rows = [
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 0.0),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 60.0),
        Observation(_iso(old), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 40.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_cut_probability", meeting, None, 0.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hold_probability", meeting, None, 64.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_hike_probability", meeting, None, 36.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.50-3.75", 64.0),
        Observation(_iso(now), "cme_fedwatch_en", "fedwatch_target_probability", meeting, "3.75-4.00", 36.0),
        Observation(_iso(old), "cnbc_us2y", "us2y_yield", None, None, 4.200),
        Observation(_iso(now), "cnbc_us2y", "us2y_yield", None, None, 4.260),
        Observation(_iso(old), "cnbc_us10y", "us10y_yield", None, None, 4.650),
        Observation(_iso(now), "cnbc_us10y", "us10y_yield", None, None, 4.680),
        Observation(_iso(old), "cnbc_us10y-cnbc_us2y", "us10y_2y_spread", None, None, 0.450),
        Observation(_iso(now), "cnbc_us10y-cnbc_us2y", "us10y_2y_spread", None, None, 0.420),
        Observation(_iso(old), "cnbc_dxy", "dxy_index", None, None, 101.000),
        Observation(_iso(now), "cnbc_dxy", "dxy_index", None, None, 101.100),
    ]
    with MacroStore(db_path) as store:
        store.insert_observations(rows)

    snapshot = DashboardDataService(db_path, now_provider=lambda: now).snapshot()
    automatic = snapshot["automatic_regime"]
    states = {item["key"]: item["state"] for item in automatic["signals"]}

    assert automatic["label"] == "HAWKISH FLATTENING"
    assert automatic["window"] == "15m"
    assert states == {
        "fedwatch": "hawkish",
        "us2y": "hawkish",
        "us10y": "neutral",
        "curve": "hawkish",
        "dxy": "pending",
    }


def test_dashboard_waits_cleanly_when_collector_database_is_absent(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    snapshot = DashboardDataService(tmp_path / "missing.sqlite", now_provider=lambda: now).snapshot()
    assert snapshot["database"]["ready"] is False
    assert snapshot["connection"]["state"] == "waiting"
    assert snapshot["regime"]["code"] == "waiting"


def test_history_builds_fedwatch_bias_and_market_metric_changes(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "macro.sqlite"
    _seed_dashboard(db_path, now)
    service = DashboardDataService(db_path, now_provider=lambda: now)

    fedwatch = service.history("fedwatch_bias", "24h")
    us2y = service.history("us2y_yield", "24h")
    dxy = service.history("dxy_index", "24h")

    assert fedwatch["ready"] is True
    assert fedwatch["current"] == pytest.approx(74.2)
    assert fedwatch["minimum"] == pytest.approx(60.6)
    assert fedwatch["period_change"] == pytest.approx(13.6)
    assert fedwatch["change_unit"] == "pct"
    assert fedwatch["meeting_date"] == "2026-09-16"
    assert [point["value"] for point in fedwatch["points"]] == pytest.approx([60.6, 74.2])

    assert us2y["current"] == pytest.approx(4.171)
    assert us2y["period_change"] == pytest.approx(-6.1)
    assert us2y["change_unit"] == "bp"

    assert dxy["current"] == pytest.approx(100.6)
    assert dxy["period_change"] == pytest.approx((100.6 / 101.0 - 1.0) * 100.0)
    assert dxy["change_unit"] == "%"


def test_history_validates_metric_and_range_and_limits_points(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "macro.sqlite"
    rows = [
        Observation(
            _iso(now - timedelta(seconds=(1800 - index) * 48)),
            "cnbc_us2y",
            "us2y_yield",
            None,
            None,
            4.0 + index / 100_000,
        )
        for index in range(1801)
    ]
    with MacroStore(db_path) as store:
        store.insert_observations(rows)

    service = DashboardDataService(db_path, now_provider=lambda: now)
    history = service.history("us2y_yield", "24h")

    assert history["raw_count"] == 1801
    assert history["returned_count"] <= 1800
    assert history["points"][-1]["value"] == pytest.approx(4.018)
    with pytest.raises(ValueError, match="Unsupported history metric"):
        service.history("unknown", "24h")
    with pytest.raises(ValueError, match="Unsupported history range"):
        service.history("us2y_yield", "1y")


def test_http_api_and_sse_deliver_snapshots_without_page_refresh(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "macro.sqlite"
    _seed_dashboard(db_path, now)
    server = MacroDashboardServer(
        DashboardDataService(db_path, now_provider=lambda: now),
        _logger(),
        host="127.0.0.1",
        port=0,
    )
    server.start()
    try:
        with urllib.request.urlopen(f"{server.url}/healthz", timeout=3) as response:
            assert json.loads(response.read())["status"] == "ok"
        with urllib.request.urlopen(f"{server.url}/api/snapshot", timeout=3) as response:
            payload = json.loads(response.read())
            assert payload["regime"]["label"] == "STRONG DOVISH"
        with urllib.request.urlopen(f"{server.url}/events", timeout=3) as response:
            lines = [response.readline().decode("utf-8").strip() for _ in range(4)]
            assert lines[0] == "event: snapshot"
            assert lines[2].startswith("data: ")
            event_payload = json.loads(lines[2].removeprefix("data: "))
            assert event_payload["revision"] == payload["revision"]
        with urllib.request.urlopen(f"{server.url}/", timeout=3) as response:
            html = response.read().decode("utf-8")
            assert "MACRO PULSE" in html
            assert "/assets/dashboard.js" in html
            assert "/history?metric=us2y_yield" in html
        with urllib.request.urlopen(f"{server.url}/history?metric=us2y_yield", timeout=3) as response:
            html = response.read().decode("utf-8")
            assert "LONG HISTORY VIEW" in html
            assert "/assets/history.js" in html
        with urllib.request.urlopen(f"{server.url}/api/history?metric=us2y_yield&range=24h", timeout=3) as response:
            history = json.loads(response.read())
            assert history["ready"] is True
            assert history["period_change"] == pytest.approx(-6.1)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{server.url}/api/history?metric=nope&range=24h", timeout=3)
        assert error.value.code == 400
    finally:
        server.stop()


def test_stopping_dashboard_does_not_stop_or_lock_collector_storage(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "macro.sqlite"
    _seed_dashboard(db_path, now)
    server = MacroDashboardServer(DashboardDataService(db_path, now_provider=lambda: now), _logger(), port=0)
    server.start()
    server.stop()

    with MacroStore(db_path) as collector_store:
        before = collector_store.count_observations()
        collector_store.insert_observations(
            [Observation(_iso(now + timedelta(seconds=15)), "cnbc_us2y", "us2y_yield", None, None, 4.170)]
        )
        assert collector_store.count_observations() == before + 1
