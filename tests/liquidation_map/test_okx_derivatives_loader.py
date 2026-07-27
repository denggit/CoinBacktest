from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from src.data_feed.okx_derivatives_loader import OKXAPIError, OKXDerivativesLoader


class _Response:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    def __init__(self, payloads: list[dict[str, Any] | _Response]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> _Response:
        self.calls.append((url, params))
        item = self.payloads.pop(0) if self.payloads else {"code": "0", "data": []}
        return item if isinstance(item, _Response) else _Response(item)


def _utc_ms(local_time: pd.Timestamp) -> int:
    return int((local_time - pd.Timedelta(hours=8)).timestamp() * 1000)


def test_open_interest_uses_contract_history_inst_id_and_persists(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    ts0 = _utc_ms(base)
    ts1 = _utc_ms(base + pd.Timedelta(minutes=5))
    session = _Session([
        {
            "code": "0",
            "data": [
                {"ts": str(ts1), "oi": "210", "oiCcy": "4200", "oiUsd": "8400000"},
                {"ts": str(ts0), "oi": "200", "oiCcy": "4000", "oiUsd": "8000000"},
            ],
        },
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_open_interest_history(base, base + pd.Timedelta(minutes=5), sleep_seconds=0, reference_time=base + pd.Timedelta(days=1))

    assert len(frame) == 2
    url, params = session.calls[0]
    assert url.endswith("/api/v5/rubik/stat/contracts/open-interest-history")
    assert params["instId"] == "ETH-USDT-SWAP"
    assert params["period"] == "5m"
    assert "ccy" not in params
    assert frame.attrs["oi_source"] == "okx_contract_history"
    loaded = loader.load_open_interest(base, base + pd.Timedelta(minutes=5))
    assert list(loaded["oi_usd"]) == [8000000.0, 8400000.0]


def test_open_interest_paginates_backward_by_oldest_timestamp(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    ts0 = _utc_ms(base)
    ts1 = _utc_ms(base + pd.Timedelta(minutes=5))
    ts2 = _utc_ms(base + pd.Timedelta(minutes=10))
    session = _Session([
        {"code": "0", "data": [[str(ts2), "3", "30", "300"], [str(ts1), "2", "20", "200"]]},
        {"code": "0", "data": [[str(ts0), "1", "10", "100"]]},
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_open_interest_history(base, base + pd.Timedelta(minutes=10), sleep_seconds=0, reference_time=base + pd.Timedelta(days=1))

    assert len(frame) == 3
    assert len(session.calls) == 2
    assert int(session.calls[1][1]["end"]) == ts1
    assert all(call[0].endswith("/open-interest-history") for call in session.calls)


def test_open_interest_empty_response_is_explicit_and_does_not_use_aggregate(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    session = _Session([{"code": "0", "data": []}])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_open_interest_history(
        base,
        base + pd.Timedelta(hours=1),
        sleep_seconds=0,
        reference_time=base + pd.Timedelta(days=1),
    )

    assert frame.empty
    assert len(session.calls) == 1
    assert "begin" not in session.calls[0][1]
    assert "end" in session.calls[0][1]
    assert "open-interest-volume" not in session.calls[0][0]
    assert "latest" in frame.attrs["availability_note"]


def test_open_interest_auto_selects_one_hour_for_old_range(tmp_path: Path) -> None:
    start = pd.Timestamp("2026-06-01 00:00")
    end = pd.Timestamp("2026-06-30 23:59:59")
    ts = _utc_ms(pd.Timestamp("2026-06-30 23:00"))
    session = _Session([
        {"code": "0", "data": [[str(ts), "10", "1", "2000"]]},
        {"code": "0", "data": []},
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_open_interest_history(
        start,
        end,
        period="5m",
        sleep_seconds=0,
        reference_time=pd.Timestamp("2026-07-17 00:00"),
    )

    assert len(frame) == 1
    assert session.calls[0][1]["period"] == "1H"
    assert "begin" not in session.calls[0][1]
    assert frame.attrs["requested_period"] == "5m"
    assert frame.attrs["effective_period"] == "1H"
    assert frame.attrs["auto_period_changed"] is True



def test_rate_limit_retries_with_retry_after_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    ts = _utc_ms(base)
    sleeps: list[float] = []
    monkeypatch.setattr("src.data_feed.okx_derivatives_loader.time.sleep", sleeps.append)
    session = _Session([
        _Response(
            {"code": "50011", "msg": "Too Many Requests"},
            status_code=429,
            headers={"Retry-After": "2"},
        ),
        {"code": "0", "data": [{"ts": str(ts), "oi": "1", "oiCcy": "10", "oiUsd": "100"}]},
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_open_interest_history(base, base, sleep_seconds=0, reference_time=base + pd.Timedelta(days=1))

    assert len(frame) == 1
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_rate_limit_exhaustion_preserves_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    monkeypatch.setattr("src.data_feed.okx_derivatives_loader.time.sleep", lambda _: None)
    session = _Session([
        _Response({"code": "50011", "msg": "Too Many Requests"}, status_code=429)
        for _ in range(6)
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)

    with pytest.raises(OKXAPIError) as exc_info:
        loader.fetch_open_interest_history(base, base, sleep_seconds=0, reference_time=base + pd.Timedelta(days=1))

    assert "HTTP 429" in str(exc_info.value)
    assert "50011" in str(exc_info.value)
    assert len(session.calls) == 6


def test_okx_non_rate_limit_error_is_not_retried(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    session = _Session([_Response({"code": "50030", "msg": "Illegal time range"}, status_code=200)])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)

    with pytest.raises(OKXAPIError) as exc_info:
        loader.fetch_open_interest_history(base, base, sleep_seconds=0, reference_time=base + pd.Timedelta(days=1))

    assert "50030" in str(exc_info.value)
    assert len(session.calls) == 1



def test_load_open_interest_prefers_full_range_period_over_partial_finer_period(tmp_path: Path) -> None:
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=_Session([]))
    start = pd.Timestamp("2026-06-01 00:00")
    end = pd.Timestamp("2026-06-30 23:00")

    hourly = pd.DataFrame(
        {
            "oi_contracts": [1.0, 2.0],
            "oi_ccy": [1.0, 2.0],
            "oi_usd": [100.0, 200.0],
            "period": ["1H", "1H"],
        },
        index=[start, end],
    )
    five_min = pd.DataFrame(
        {
            "oi_contracts": [3.0, 4.0],
            "oi_ccy": [3.0, 4.0],
            "oi_usd": [300.0, 400.0],
            "period": ["5m", "5m"],
        },
        index=[end - pd.Timedelta(minutes=5), end],
    )
    loader._save_open_interest(hourly)
    loader._save_open_interest(five_min)

    loaded = loader.load_open_interest(start, end)
    assert set(loaded["period"]) == {"1H"}
    assert loaded.attrs["selected_period"] == "1H"

def test_mark_candle_timestamp_is_shifted_to_available_time(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    ts = _utc_ms(base)
    session = _Session([
        {"code": "0", "data": [[str(ts), "2000", "2001", "1999", "2000.5", "1"]]},
        {"code": "0", "data": []},
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_mark_price_history(base, base + pd.Timedelta(minutes=2), timeframe="1m", sleep_seconds=0)
    assert frame.index[0] == base + pd.Timedelta(minutes=1)


def test_mark_price_uses_official_history_endpoint(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    ts = _utc_ms(base)
    session = _Session([
        {"code": "0", "data": [[str(ts), "2000", "2001", "1999", "2000.5", "1"]]},
        {"code": "0", "data": []},
    ])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    loader.fetch_mark_price_history(base, base + pd.Timedelta(minutes=2), timeframe="1m", sleep_seconds=0)
    assert session.calls[0][0].endswith("/api/v5/market/history-mark-price-candles")


def test_liquidation_history_does_not_call_delisted_rest_endpoint(tmp_path: Path) -> None:
    base = pd.Timestamp("2026-06-01 00:00")
    session = _Session([])
    loader = OKXDerivativesLoader(data_dir=tmp_path, session=session)
    frame = loader.fetch_liquidation_orders(base, base + pd.Timedelta(days=1), sleep_seconds=0)

    assert frame.empty
    assert session.calls == []
    assert frame.attrs["remote_history_available"] is False
    assert "historical liquidation REST" in frame.attrs["availability_note"]
