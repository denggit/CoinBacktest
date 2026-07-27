from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_feed.binance_futures_metrics_loader import (
    BinanceFuturesMetricsLoader,
    BinanceMetricsDownloadError,
)


class _Response:
    def __init__(self, content: bytes = b"", *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.text = content.decode("utf-8", errors="replace")


class _Session:
    def __init__(self, responses: dict[str, list[_Response] | _Response]) -> None:
        self.responses = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in responses.items()
        }
        self.calls: list[str] = []

    def get(self, url: str, timeout: int) -> _Response:
        self.calls.append(url)
        queue = self.responses.get(url)
        if not queue:
            return _Response(status_code=404)
        return queue.pop(0)


def _metrics_zip(day: str, *, seconds_timestamp: bool = False, rows: int = 3) -> bytes:
    start = pd.Timestamp(day, tz="UTC")
    times = pd.date_range(start, periods=rows, freq="5min")
    create_time: list[Any]
    if seconds_timestamp:
        create_time = [int(ts.timestamp()) for ts in times]
    else:
        create_time = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in times]
    frame = pd.DataFrame(
        {
            "create_time": create_time,
            "symbol": ["ETHUSDT"] * rows,
            "sum_open_interest": np.arange(rows, dtype=float) + 100.0,
            "sum_open_interest_value": (np.arange(rows, dtype=float) + 100.0) * 2000.0,
            "count_toptrader_long_short_ratio": [1.2] * rows,
            "sum_toptrader_long_short_ratio": [1.1] * rows,
            "count_long_short_ratio": [0.9] * rows,
            "sum_taker_long_short_vol_ratio": [0.8, 1.0, 1.2][:rows],
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"ETHUSDT-metrics-{day}.csv", frame.to_csv(index=False))
    return buffer.getvalue()


def _session_for_day(loader: BinanceFuturesMetricsLoader, day: str, payload: bytes) -> _Session:
    sha = hashlib.sha256(payload).hexdigest()
    return _Session(
        {
            loader.archive_url(day): _Response(payload),
            loader.checksum_url(day): _Response(f"{sha}  ETHUSDT-metrics-{day}.zip\n".encode()),
        }
    )


def test_download_parses_persists_and_uses_project_timezone(tmp_path: Path) -> None:
    seed = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    payload = _metrics_zip("2022-01-01")
    session = _session_for_day(seed, "2022-01-01", payload)
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=session)

    summary = loader.download_history("2022-01-01", "2022-01-01", workers=1)
    assert summary.downloaded_days == 1
    assert summary.rows_written == 3

    loaded = loader.load_archive_days("2022-01-01", "2022-01-01", index_mode="none")
    assert len(loaded) == 3
    assert loaded.loc[0, "timestamp"] == pd.Timestamp("2022-01-01 08:00:00")
    assert loaded.loc[0, "source_timestamp_utc"] == pd.Timestamp("2022-01-01 00:00:00")
    assert loaded.loc[0, "available_time"] == pd.Timestamp("2022-01-01 08:01:00")
    assert loaded.loc[0, "taker_volume_imbalance"] == pytest.approx((0.8 - 1.0) / (0.8 + 1.0))
    assert loader.raw_archive_path("2022-01-01").exists()


def test_numeric_seconds_create_time_is_supported(tmp_path: Path) -> None:
    seed = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    payload = _metrics_zip("2022-01-02", seconds_timestamp=True)
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=_session_for_day(seed, "2022-01-02", payload))
    result = loader.inspect_archive_day("2022-01-02")
    assert result.rows == 3
    assert result.frame is not None
    assert result.frame.loc[0, "source_timestamp_utc"] == pd.Timestamp("2022-01-02 00:00:00")


def test_resume_skips_complete_day_without_network(tmp_path: Path) -> None:
    seed = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    payload = _metrics_zip("2022-01-01")
    first = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=_session_for_day(seed, "2022-01-01", payload))
    first.download_history("2022-01-01", "2022-01-01", workers=1)

    empty_session = _Session({})
    second = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=empty_session)
    summary = second.download_history("2022-01-01", "2022-01-01", workers=1)
    assert summary.skipped_days == 1
    assert empty_session.calls == []


def test_missing_archive_is_recorded_and_resume_safe(tmp_path: Path) -> None:
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=_Session({}))
    summary = loader.download_history("2022-01-03", "2022-01-03", workers=1)
    assert summary.missing_days == 1
    coverage = loader.coverage_by_day("2022-01-03", "2022-01-03")
    assert coverage.loc[0, "status"] == "missing"
    assert "404" in coverage.loc[0, "error"]


def test_checksum_mismatch_is_error(tmp_path: Path) -> None:
    seed = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    day = "2022-01-04"
    payload = _metrics_zip(day)
    session = _Session(
        {
            seed.archive_url(day): _Response(payload),
            seed.checksum_url(day): _Response(("0" * 64 + "  file.zip\n").encode()),
        }
    )
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=session)
    summary = loader.download_history(day, day, workers=1)
    assert summary.error_days == 1
    coverage = loader.coverage_by_day(day, day)
    assert coverage.loc[0, "status"] == "error"
    assert "checksum mismatch" in coverage.loc[0, "error"]


def test_required_checksum_missing_raises_on_inspect(tmp_path: Path) -> None:
    seed = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    day = "2022-01-05"
    payload = _metrics_zip(day)
    session = _Session({seed.archive_url(day): _Response(payload)})
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=session)
    with pytest.raises(BinanceMetricsDownloadError, match="checksum missing"):
        loader.inspect_archive_day(day, require_checksum=True)


def test_relative_features_are_causal_and_gap_safe(tmp_path: Path) -> None:
    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path, session=_Session({}))
    day = pd.Timestamp("2022-01-01")
    frame = pd.DataFrame(
        {
            "symbol": ["ETHUSDT"] * 4,
            "timestamp": [
                day + pd.Timedelta(hours=8),
                day + pd.Timedelta(hours=8, minutes=5),
                day + pd.Timedelta(hours=8, minutes=10),
                day + pd.Timedelta(hours=8, minutes=25),
            ],
            "source_timestamp_utc": [
                day,
                day + pd.Timedelta(minutes=5),
                day + pd.Timedelta(minutes=10),
                day + pd.Timedelta(minutes=25),
            ],
            "period": ["5m"] * 4,
            "sum_open_interest": [100.0, 110.0, 121.0, 150.0],
            "sum_open_interest_value": [1000.0, 1100.0, 1210.0, 1500.0],
            "count_toptrader_long_short_ratio": [1.0] * 4,
            "sum_toptrader_long_short_ratio": [1.0] * 4,
            "count_long_short_ratio": [1.0] * 4,
            "sum_taker_long_short_vol_ratio": [1.0] * 4,
            "source_day_utc": ["2022-01-01"] * 4,
            "source": ["test"] * 4,
        }
    )
    from src.data_feed.binance_futures_metrics_loader import BinanceMetricsDayResult

    loader.store.save_day(
        BinanceMetricsDayResult(
            day_utc=pd.Timestamp("2022-01-01").date(),
            status="partial",
            rows=len(frame),
            frame=frame,
            source_url="test",
        )
    )

    features = loader.load_relative_features(
        "2022-01-01 08:01:00",
        "2022-01-01 08:26:00",
        windows=("5m", "15m"),
        publication_lag="1min",
        baseline_tolerance="5min",
        index_mode="none",
    )
    row_805 = features.loc[features["timestamp"] == pd.Timestamp("2022-01-01 08:05:00")].iloc[0]
    assert row_805["oi_usd_change_5m"] == pytest.approx(0.1)
    assert row_805["available_time"] == pd.Timestamp("2022-01-01 08:06:00")

    # At 08:25, the 5-minute target is 08:20. The latest baseline is 08:10,
    # which is more than the allowed 5-minute staleness and must be rejected.
    row_825 = features.loc[features["timestamp"] == pd.Timestamp("2022-01-01 08:25:00")].iloc[0]
    assert pd.isna(row_825["oi_usd_change_5m"])
    # 15-minute target is exactly 08:10, so it remains valid and causal.
    assert row_825["oi_usd_change_15m"] == pytest.approx(1500.0 / 1210.0 - 1.0)


def test_okx_style_symbol_is_normalized(tmp_path: Path) -> None:
    loader = BinanceFuturesMetricsLoader(symbol="ETH-USDT-SWAP", data_dir=tmp_path)
    assert loader.symbol == "ETHUSDT"
