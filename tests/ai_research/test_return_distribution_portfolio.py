from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.return_distribution_portfolio.config import ReturnDistributionConfig
from src.ai_research.return_distribution_portfolio.dataset import build_causal_features, build_future_targets, feature_columns


def _raw(minutes: int = 6000) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=minutes, freq="1min")
    base = 1000.0 + np.arange(minutes, dtype=float) * 0.1
    side = np.sin(np.arange(minutes) / 10.0) * 50.0
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.05,
            "volume": 10.0,
            "notional": 10_000.0,
            "trades_count": 20.0,
            "buy_notional": 5_000.0 + side,
            "sell_notional": 5_000.0 - side,
            "delta_notional": side * 2.0,
            "large_buy_notional": 500.0 + side,
            "large_sell_notional": 500.0 - side,
            "large_delta_notional": side * 2.0,
        },
        index=idx,
    )


def test_completed_5m_bar_is_indexed_at_available_time() -> None:
    cfg = ReturnDistributionConfig(research_start="2023-01-01", research_end="2023-01-31")
    feat = build_causal_features(_raw(), cfg)
    # 00:00-00:04 bar is only visible at 00:05.
    assert feat.index[0] == pd.Timestamp("2023-01-01 00:05:00")
    assert feat.iloc[0]["open"] == 1000.0
    assert np.isclose(feat.iloc[0]["close"], 1000.45)


def test_target_return_uses_next_executable_open_and_future_only_price() -> None:
    cfg = ReturnDistributionConfig(research_start="2023-01-01", research_end="2023-01-31")
    raw = _raw()
    decisions = pd.DatetimeIndex([pd.Timestamp("2023-01-01 01:00:00")])
    t = build_future_targets(raw, decisions, cfg)
    entry = raw.loc[pd.Timestamp("2023-01-01 01:00:00"), "open"]
    exit_30 = raw.loc[pd.Timestamp("2023-01-01 01:30:00"), "open"]
    assert t.loc[decisions[0], "execution_price"] == entry
    assert np.isclose(t.loc[decisions[0], "ret_h30"], exit_30 / entry - 1.0)


def test_target_builder_does_not_leak_target_columns_into_features() -> None:
    cfg = ReturnDistributionConfig(research_start="2023-01-01", research_end="2023-01-31")
    raw = _raw()
    feat = build_causal_features(raw, cfg)
    decisions = feat.index[:100]
    frame = feat.reindex(decisions).join(build_future_targets(raw, decisions, cfg))
    cols = feature_columns(frame)
    assert "execution_price" not in cols
    assert not any(c.startswith("ret_h") for c in cols)
    assert not any(c.startswith("mfe_") for c in cols)
    assert not any(c.startswith("mae_") for c in cols)
    assert not any(c.startswith("future_rv_h") for c in cols)


def test_2022_warmup_contract_is_strict() -> None:
    cfg = ReturnDistributionConfig()
    assert pd.Timestamp(cfg.warmup_start).year == 2022
    assert pd.Timestamp(cfg.research_start) == pd.Timestamp("2023-01-01 00:00:00")


def test_future_mutation_cannot_change_current_features() -> None:
    cfg = ReturnDistributionConfig(research_start="2023-01-01", research_end="2023-01-31")
    raw = _raw()
    decision = pd.Timestamp("2023-01-03 00:00:00")
    base = build_causal_features(raw, cfg).loc[decision].copy()
    mutated = raw.copy()
    future_mask = mutated.index >= decision
    mutated.loc[future_mask, ["open", "high", "low", "close"]] *= 3.0
    mutated.loc[future_mask, ["notional", "buy_notional", "sell_notional", "delta_notional"]] *= 7.0
    after = build_causal_features(mutated, cfg).loc[decision]
    pd.testing.assert_series_equal(base, after, check_names=False)


def test_incomplete_five_minute_bar_is_not_used_as_feature() -> None:
    cfg = ReturnDistributionConfig(research_start="2023-01-01", research_end="2023-01-31")
    raw = _raw(100)
    raw = raw.drop(pd.Timestamp("2023-01-01 00:03:00"))
    feat = build_causal_features(raw, cfg)
    assert pd.Timestamp("2023-01-01 00:05:00") not in feat.index
