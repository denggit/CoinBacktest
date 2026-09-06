from __future__ import annotations

import numpy as np
import pandas as pd

from human_replay_lab.data_service import ReplayDataService
from src.data_feed.okx_loader import OKXDataLoader


def _write_symbol(tmp_path, symbol: str, base: float) -> None:
    idx = pd.date_range("2026-06-19 18:00:00", "2026-06-20 04:01:00", freq="1min")
    x = np.arange(len(idx), dtype=float)
    frame = pd.DataFrame(
        {
            "open": base + x * 0.01,
            "high": base + x * 0.01 + 0.2,
            "low": base + x * 0.01 - 0.2,
            "close": base + x * 0.01 + 0.1,
            "volume": 1.0,
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    OKXDataLoader(symbol=symbol, timeframe="1m", db_dir=str(tmp_path)).save_local_data(frame)


def test_replay_discovers_and_switches_local_okx_symbols(tmp_path):
    _write_symbol(tmp_path, "SOXL-USDT-SWAP", 100.0)
    _write_symbol(tmp_path, "ETH-USDT-SWAP", 2500.0)

    service = ReplayDataService(tmp_path)
    assert set(service.available_symbols()[:2]) == {"SOXL-USDT-SWAP", "ETH-USDT-SWAP"}

    soxl = service.coverage("SOXL-USDT-SWAP")
    eth = service.coverage("ETH-USDT-SWAP")
    assert soxl["rows_1m"] == eth["rows_1m"] > 0

    cursor = service.cursor_for_date("ETH-USDT-SWAP", "2026-06-19")
    assert cursor == pd.Timestamp("2026-06-19 06:00:00")
    assert service.session_profile("ETH-USDT-SWAP") == "crypto_24x7_continuous_replay"
    assert service.execution_open("ETH-USDT-SWAP", cursor) > 2000
    bars = service.candles("ETH-USDT-SWAP", "15m", cursor, 50).bars
    assert bars
    assert bars[-1]["is_partial"] is True


def test_okx_loader_range_is_bounded(tmp_path):
    _write_symbol(tmp_path, "ETH-USDT-SWAP", 2500.0)
    loader = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m", db_dir=str(tmp_path))
    frame = loader.load_local_data_range("2026-06-19 19:00:00", "2026-06-19 19:10:00")
    assert len(frame) == 11
    assert frame.index.min() == pd.Timestamp("2026-06-19 19:00:00")
    assert frame.index.max() == pd.Timestamp("2026-06-19 19:10:00")
    coverage = loader.get_local_data_coverage()
    assert coverage["rows"] > len(frame)
    assert "ETH-USDT-SWAP" in OKXDataLoader.list_local_symbols(str(tmp_path), "1m")
