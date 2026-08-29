from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.rl_market_agent.config import DEFAULT_CONFIG
from src.ai_research.rl_market_agent.pipeline import run_r00


class FakeRepository:
    def __init__(self):
        self.kline_calls: list[str] = []

    def _bars(self, start, end, freq):
        idx = pd.date_range(pd.Timestamp(start).floor(freq), pd.Timestamp(end).ceil(freq), freq=freq)
        x = np.arange(len(idx), dtype=float)
        return pd.DataFrame({"open":100+x*.01,"high":101+x*.01,"low":99+x*.01,"close":100.5+x*.01,"volume":1000+x}, index=idx)

    def load_kline(self, timeframe, start, end):
        self.kline_calls.append(timeframe)
        if timeframe != "1m":
            raise AssertionError(f"R00 must not read independent HTF kline cache: {timeframe}")
        return self._bars(start, end, "1min")

    def load_trade_bars(self, timeframe, start, end):
        if timeframe == "5s":
            return pd.DataFrame()
        bars = self._bars(start, end, "1min")
        bars = bars.assign(
            notional=10000.0, delta_notional=500.0, trades_count=20.0,
            buy_notional=5250.0, sell_notional=4750.0,
            large_buy_notional=500.0, large_sell_notional=300.0,
            large_delta_notional=200.0, large_trades_count=2.0,
            max_trade_notional=1000.0, vwap=bars["close"] - 0.05,
        )
        return bars

    def load_range_bars(self, *args, **kwargs):
        return pd.DataFrame()

    def load_footprint(self, *args, **kwargs):
        return pd.DataFrame()


def _cfg(tmp_path, *, end="2023-01-03 23:59:59"):
    return replace(
        DEFAULT_CONFIG,
        warmup_start="2022-11-01 00:00:00",
        research_start="2023-01-03 00:00:00",
        sealed_holdout_start="2023-01-03 12:00:00",
        research_end=end,
        label_horizons_minutes=(15,),
        trade_windows_minutes=(5,),
        micro_context_minutes=10,
        cache_dir=str(tmp_path / "cache"),
        report_dir=str(tmp_path / "report"),
    )


def test_pipeline_smoke_builds_causal_shard(tmp_path):
    repo = FakeRepository()
    cfg = _cfg(tmp_path)
    result = run_r00(cfg, repository=repo, finalize_report=False)
    assert len(result["records"]) == 1
    assert all(row["passed"] for row in result["causal_audits"])
    assert repo.kline_calls == ["1m"]
    assert (tmp_path / "cache" / "2023-01" / "features.npy").exists()
    assert (tmp_path / "report" / "99_decision.md").exists()


def test_pipeline_uses_official_1m_as_only_kline_base(tmp_path):
    repo = FakeRepository()
    cfg = _cfg(tmp_path)
    result = run_r00(cfg, repository=repo, finalize_report=False)
    kline_cov = [x for x in result["coverage"] if x["source"].startswith("kline_")]
    assert kline_cov
    assert all(x["coverage_ratio"] >= 0.99 for x in kline_cov)
    assert all(x["note"] == "official_1m_resampled" for x in kline_cov)
    assert repo.kline_calls == ["1m"]


def test_final_decision_tail_is_reserved_for_complete_labels(tmp_path):
    cfg = _cfg(tmp_path, end="2023-01-03 23:59:59")
    result = run_r00(cfg, repository=FakeRepository(), finalize_report=False)
    record = result["records"][0]
    assert pd.Timestamp(record["end_time"]) <= cfg.decision_end
    labels = np.load(tmp_path / "cache" / "2023-01" / "labels.npy")
    # All persisted rows must have a complete 15-minute label vector in this test.
    assert np.isfinite(labels).all()
