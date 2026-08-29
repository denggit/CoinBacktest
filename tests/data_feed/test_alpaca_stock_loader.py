from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader


class _Response:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, headers, timeout):
        self.calls.append((url, dict(params), dict(headers), timeout))
        return self.responses.pop(0)


def _bar(ts: str, price: float):
    return {"t": ts, "o": price, "h": price + 1, "l": price - 1, "c": price + 0.5, "v": 1000, "n": 10, "vw": price + 0.25}


def test_fetch_remote_paginates_sip_minute_bars(tmp_path: Path) -> None:
    session = _Session(
        [
            _Response({"bars": {"SOXL": [_bar("2026-06-01T08:00:00Z", 10)]}, "next_page_token": "NEXT"}),
            _Response({"bars": {"SOXL": [_bar("2026-06-01T08:01:00Z", 11)]}, "next_page_token": None}),
        ]
    )
    loader = AlpacaStockLoader(
        symbol="SOXL",
        data_dir=tmp_path,
        api_key_id="key",
        api_secret_key="secret",
        session=session,
    )
    frame = loader.fetch_remote("2026-06-01T08:00:00Z", "2026-06-01T08:01:00Z")
    assert len(frame) == 2
    assert str(frame.index.tz) == "UTC"
    assert session.calls[0][1]["feed"] == "sip"
    assert session.calls[0][1]["timeframe"] == "1Min"
    assert session.calls[1][1]["page_token"] == "NEXT"


def test_local_cache_round_trip_is_utc_aware(tmp_path: Path) -> None:
    loader = AlpacaStockLoader(symbol="SOXL", data_dir=tmp_path, api_key_id="key", api_secret_key="secret", session=_Session([]))
    idx = pd.date_range("2026-06-01 08:00", periods=2, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000.0, 1200.0],
            "trade_count": [10, 12],
            "vwap": [10.2, 11.2],
        },
        index=idx,
    )
    loader.save_local_data(frame)
    loaded = loader.load_local_data()
    assert len(loaded) == 2
    assert str(loaded.index.tz) == "UTC"
    assert loaded.iloc[1]["close"] == 11.5


def test_local_range_query_does_not_require_full_table_load(tmp_path: Path) -> None:
    loader = AlpacaStockLoader(
        symbol="SOXL",
        timeframe="1Min",
        feed="sip",
        adjustment="split",
        data_dir=tmp_path,
        api_key_id="key",
        api_secret_key="secret",
        session=_Session([]),
    )
    idx = pd.date_range("2026-06-01 08:00", periods=10, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": range(10, 20),
            "high": range(11, 21),
            "low": range(9, 19),
            "close": [x + 0.5 for x in range(10, 20)],
            "volume": [1000.0] * 10,
            "trade_count": [10] * 10,
            "vwap": [x + 0.25 for x in range(10, 20)],
        },
        index=idx,
    )
    loader.save_local_data(frame)
    sliced = loader.load_local_data_by_date_range("2026-06-01T08:03:00Z", "2026-06-01T08:05:00Z")
    assert len(sliced) == 3
    assert sliced.index.min() == pd.Timestamp("2026-06-01T08:03:00Z")
    assert sliced.index.max() == pd.Timestamp("2026-06-01T08:05:00Z")


def test_credentials_fall_back_to_project_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        AlpacaStockLoader,
        "_load_project_env",
        staticmethod(lambda: {
            "APCA_API_KEY_ID": "dotenv-key",
            "APCA_API_SECRET_KEY": "dotenv-secret",
        }),
    )
    loader = AlpacaStockLoader(symbol="SOXL", data_dir=tmp_path, session=_Session([]))
    assert loader.api_key_id == "dotenv-key"
    assert loader.api_secret_key == "dotenv-secret"


def test_explicit_credentials_override_environment_and_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "process-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "process-secret")
    monkeypatch.setattr(
        AlpacaStockLoader,
        "_load_project_env",
        staticmethod(lambda: {
            "APCA_API_KEY_ID": "dotenv-key",
            "APCA_API_SECRET_KEY": "dotenv-secret",
        }),
    )
    loader = AlpacaStockLoader(
        symbol="SOXL",
        data_dir=tmp_path,
        api_key_id="explicit-key",
        api_secret_key="explicit-secret",
        session=_Session([]),
    )
    assert loader.api_key_id == "explicit-key"
    assert loader.api_secret_key == "explicit-secret"
