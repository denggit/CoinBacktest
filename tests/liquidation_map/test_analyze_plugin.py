from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.plugins import liquidation_heatmap as module


def _bars(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2026-06-01", periods=rows, freq="1min")
    close = 2000 + np.linspace(0, 15, rows)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
            "buy_notional": 1_200_000,
            "sell_notional": 800_000,
            "delta_notional": 400_000,
        },
        index=index,
    )


class _Coverage:
    def __init__(self, dataset: str, rows: int) -> None:
        self.dataset = dataset
        self.rows = rows


class _FakeLoader:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def load_open_interest(self, start, end):
        idx = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=4), pd.Timestamp(end), freq="5min")
        return pd.DataFrame({"oi_usd": 1_000_000_000 + np.arange(len(idx)) * 5_000_000}, index=idx)

    def load_funding_rates(self, start, end):
        return pd.DataFrame({"funding_rate": [0.0001]}, index=[pd.Timestamp(start)])

    def load_mark_prices(self, start, end, timeframe="1m"):
        return pd.DataFrame()

    def load_liquidations(self, start, end):
        return pd.DataFrame()

    def coverage(self):
        return [_Coverage("open_interest", 10)]


def test_plugin_returns_minimal_heatmap(monkeypatch) -> None:
    monkeypatch.setattr(module, "OKXDerivativesLoader", _FakeLoader)
    bars = _bars()
    context = PluginRunContext(
        display_df=bars,
        visible_df=bars,
        request={"symbol": "ETH-USDT-SWAP", "data_type": "trade_bar", "timeframe": "1m"},
        meta={"symbol": "ETH-USDT-SWAP"},
    )
    result = module.LiquidationHeatmapPlugin().run_with_context(context, {})
    assert result.heatmap
    assert not result.tracks
    assert not result.bands
    assert result.summary["estimated"] is True
    assert result.summary["ui"]["brief_labels"] == ["清算分布", "最近上方", "最近下方"]
    assert "brief_direction" in result.row_fields
