from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.rl_market_agent.opportunity import TradeTemplate
from src.ai_research.rl_market_agent.r01_config import R01Config, WalkForwardFold
from src.ai_research.rl_market_agent.r01_pipeline import run_r01
from src.ai_research.rl_market_agent.shards import ShardStore


class _FakeRepo:
    def __init__(self, path: pd.DataFrame):
        self.path = path

    def load_trade_bars(self, timeframe, start, end):
        assert timeframe == "1m"
        return self.path.loc[(self.path.index >= pd.Timestamp(start)) & (self.path.index <= pd.Timestamp(end))].copy()


def test_r01_end_to_end_smoke_keeps_seal_closed(tmp_path):
    idx = pd.date_range("2023-01-01", "2023-01-08 23:55", freq="5min")
    rng = np.random.default_rng(7)
    x1 = np.sin(np.arange(len(idx)) / 17.0).astype(np.float32)
    x2 = rng.normal(0, 1, len(idx)).astype(np.float32)
    features = pd.DataFrame({
        "kline_5m__signal": x1,
        "trade_1m__flow": x2,
        "range_r0020__state": (x1 + x2 * 0.1).astype(np.float32),
        "availability__trade_1m": 1.0,
    }, index=idx)
    final = (0.004 * x1).astype(np.float32)
    labels = pd.DataFrame({
        "h60__final_return": final,
        "h60__long_mfe": np.maximum(final, 0) + 0.007,
        "h60__long_mae": -0.002 - np.maximum(-final, 0),
        "h60__short_mfe": np.maximum(-final, 0) + 0.007,
        "h60__short_mae": -0.002 - np.maximum(final, 0),
    }, index=idx)
    flags = pd.DataFrame({"sealed_holdout": 0, "core_valid": 1}, index=idx)
    store = ShardStore(tmp_path / "cache", project_root=tmp_path / "unrelated_root")
    store.write(
        shard_id="2023-01", features=features, labels=labels, flags=flags, sealed_holdout=False,
        extra_metadata={
            "feature_names": list(features.columns), "label_names": list(labels.columns),
            "flag_names": list(flags.columns),
        },
    )

    pidx = pd.date_range("2023-01-04", "2023-01-08 23:59", freq="1min")
    base = 100.0 + 0.01 * np.arange(len(pidx)) + 0.3 * np.sin(np.arange(len(pidx)) / 9.0)
    path = pd.DataFrame({
        "open": base,
        "high": base * 1.0015,
        "low": base * 0.9985,
        "close": base * (1 + 0.0002 * np.sin(np.arange(len(pidx)) / 7.0)),
    }, index=pidx)

    fold = WalkForwardFold(
        "SMOKE",
        "2023-01-01", "2023-01-04",
        "2023-01-04", "2023-01-06",
        "2023-01-06", "2023-01-08",
    )
    cfg = R01Config(
        sealed_holdout_start="2023-01-08",
        r00_cache_dir=str(tmp_path / "cache"), report_dir=str(tmp_path / "report"),
        threshold_quantiles=(0.70,), min_calibration_trades=1,
        folds=(fold,), trade_templates=(TradeTemplate("H60", 60, 0.006, 0.004),),
    )
    result = run_r01(cfg, repository=_FakeRepo(path), finalize_report=False)
    assert result["decision"] in {
        "PASS_R01_STRATEGY_CANDIDATE", "PROMISING_BUT_NOT_R01_PASS", "NO_TRADABLE_STRATEGY_R01"
    }
    assert (tmp_path / "report" / "01_purged_fold_manifest.csv").exists()
    env = (tmp_path / "report" / "13_environment.json").read_text(encoding="utf-8")
    assert '"sealed_holdout_opened": false' in env.lower()
