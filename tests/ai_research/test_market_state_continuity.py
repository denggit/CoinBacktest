from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.market_state_continuity.config import DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
from src.ai_research.market_state_continuity.state_cache import (
    build_state_frame,
    build_state_targets,
    causal_hysteresis_state,
    datetime_index_to_ns,
    ns_to_datetime,
    _causal_daily_strategic_thresholds,
)

from src.ai_research.market_state_continuity.modeling import (
    ContinuityPeriodData,
    mechanical_feature_sets,
    transition_alert_episode_audit,
)



def test_r0333_contract_keeps_2026_sealed_and_state_auxiliary() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    config.validate()
    assert pd.Timestamp(config.research_end) < pd.Timestamp(config.sealed_holdout_start)
    assert config.ordinary_kline_end.startswith("2021")
    assert config.trade_bar_start.startswith("2022")
    assert "persist" in " ".join(config.target_names())


def test_causal_hysteresis_reduces_threshold_chatter() -> None:
    values = np.array([0.0, 0.31, 0.28, 0.32, 0.27, 0.33, 0.09, -0.31, -0.20, -0.09, 0.0])
    raw = np.where(values >= 0.30, 1, np.where(values <= -0.30, -1, 0))
    stable = causal_hysteresis_state(values, enter_threshold=0.30, exit_threshold=0.10)
    raw_flips = int(np.sum(raw[1:] != raw[:-1]))
    stable_flips = int(np.sum(stable[1:] != stable[:-1]))
    assert stable_flips < raw_flips
    assert stable[1] == 1
    assert stable[5] == 1
    assert stable[6] == 0
    assert stable[7] == -1


def _feature_frame(periods: int = 500) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="15min")
    frame = pd.DataFrame(index=index)
    required = {
        "tf1d_": (90, 180, 365),
        "tf4h_": (12, 30, 90, 180, 360, 720),
        "tf1h_": (24, 72, 168),
        "tf15m_": (8, 32),
        "tf5m_": (12, 48),
        "tf1m_": (60, 240),
    }
    phase = np.linspace(-2.0, 2.0, periods)
    for prefix, windows in required.items():
        for window in windows:
            frame[f"{prefix}ret_{window}"] = 0.02 * np.tanh(phase)
            frame[f"{prefix}rv_{window}"] = 0.01
            frame[f"{prefix}trend_eff_{window}"] = 0.7
    return frame


def test_hierarchical_states_can_disagree_without_forced_single_label() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    frame = _feature_frame()
    # Keep strategic positive while forcing the entry layer negative at the end.
    for column in [name for name in frame.columns if name.startswith(("tf1d_", "tf4h_", "tf1h_")) and "_ret_" in name]:
        frame[column] = 0.20
    for column in [name for name in frame.columns if name.startswith(("tf15m_", "tf5m_", "tf1m_")) and "_ret_" in name]:
        frame[column] = -0.20
    state = build_state_frame(frame, config)
    assert state["strategic_state"].iloc[-1] == 1
    assert state["entry_state"].iloc[-1] == -1
    assert state["strategic_tactical_alignment"].notna().any()


def test_state_persistence_target_uses_future_state_only_as_label() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    index = pd.date_range("2024-01-01", periods=400, freq="15min")
    state = pd.DataFrame(index=index)
    for layer in ("strategic", "tactical", "entry"):
        state[f"{layer}_state"] = 1
        state[f"{layer}_score"] = 0.8
    state["activity_state"] = 1
    state["activity_score"] = 0.7
    # Flip tactical only after three hours.
    state.loc[index[20]:, "tactical_state"] = -1
    targets = build_state_targets(state, config)
    assert targets.loc[index[0], "tactical_persist_h3"] == 1.0
    assert targets.loc[index[8], "tactical_persist_h3"] == 0.0
    assert np.isnan(targets["strategic_persist_h72"].iloc[-1])


def test_state_frame_has_causal_age_and_flip_features() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    state = build_state_frame(_feature_frame(), config)
    required = {
        "strategic_age_bars",
        "tactical_age_bars",
        "entry_age_bars",
        "activity_age_bars",
        "strategic_flip_rate_24h",
        "tactical_entry_alignment",
        "long_pullback_setup",
        "trend_momentum_long",
    }
    assert required.issubset(state.columns)
    assert (state["strategic_age_bars"] >= 1).all()


def test_unified_loader_switches_sources_without_overlap() -> None:
    from types import MethodType

    from src.ai_research.market_state_continuity.data import UnifiedOHLCVLoader

    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    loader = object.__new__(UnifiedOHLCVLoader)
    loader.config = config
    loader.data_dir = None

    def ordinary(self, start, end):
        index = pd.date_range(start.floor("min"), end.floor("min"), freq="1min")
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "source": "ordinary_kline",
            },
            index=index,
        )

    def trade(self, start, end):
        index = pd.date_range(start.floor("min"), end.floor("min"), freq="1min")
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "source": "trade_bar",
            },
            index=index,
        )

    loader._load_ordinary = MethodType(ordinary, loader)
    loader._load_trade_bar = MethodType(trade, loader)
    frame = loader.fetch_data_by_date_range("2021-12-31 23:58:00", "2022-01-01 00:02:00")
    assert frame.index.is_unique
    assert list(frame["source"]) == [
        "ordinary_kline",
        "ordinary_kline",
        "trade_bar",
        "trade_bar",
        "trade_bar",
    ]

def test_datetime64_microseconds_are_saved_as_true_nanoseconds() -> None:
    source = pd.DatetimeIndex(np.array(["2024-01-01T00:00:00", "2024-12-31T23:45:00"], dtype="datetime64[us]"))
    encoded = datetime_index_to_ns(source)
    assert encoded[0] == 1_704_067_200_000_000_000
    decoded = ns_to_datetime(encoded)
    assert list(decoded.year) == [2024, 2024]
    assert decoded[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert decoded[-1] == pd.Timestamp("2024-12-31 23:45:00")



def test_strategic_thresholds_use_prior_days_only() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    index = pd.date_range("2020-01-01", periods=420 * 4, freq="6h")
    values = pd.Series(np.sin(np.linspace(0.0, 20.0, len(index))) * 0.1, index=index)
    original = _causal_daily_strategic_thresholds(values, config)
    perturbed = values.copy()
    cutoff = pd.Timestamp("2020-11-01")
    perturbed.loc[perturbed.index >= cutoff] = 10.0
    changed = _causal_daily_strategic_thresholds(perturbed, config)
    prior = original.index < cutoff
    pd.testing.assert_frame_equal(original.loc[prior], changed.loc[prior])


def test_persistence_rejects_flip_away_and_return() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    index = pd.date_range("2024-01-01", periods=400, freq="15min")
    state = pd.DataFrame(index=index)
    for layer in ("strategic", "tactical", "entry"):
        state[f"{layer}_state"] = 1
        state[f"{layer}_score"] = 0.8
    state["activity_state"] = 1
    state["activity_score"] = 0.7
    # Tactical state leaves +1 and returns to +1 before the 3h endpoint.
    state.loc[index[4]:index[7], "tactical_state"] = -1
    targets = build_state_targets(state, config)
    assert state.loc[index[0], "tactical_state"] == state.loc[index[12], "tactical_state"]
    assert targets.loc[index[0], "tactical_persist_h3"] == 0.0


def test_state_frame_exposes_boundary_margin_and_strategic_thresholds() -> None:
    state = build_state_frame(_feature_frame(), DEFAULT_MARKET_STATE_CONTINUITY_CONFIG)
    required = {
        "strategic_long_enter_threshold",
        "strategic_short_enter_threshold",
        "strategic_boundary_margin",
        "tactical_boundary_margin",
        "entry_boundary_margin",
        "activity_boundary_margin",
    }
    assert required.issubset(state.columns)
    assert np.isfinite(state["strategic_boundary_margin"]).any()


class _DeterministicProbabilityModel:
    def predict_proba(self, x):
        risk = np.asarray(x[:, 0], dtype=float)
        persist = np.clip(1.0 - risk, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - persist, persist])


def test_mechanical_feature_contract_is_layer_specific() -> None:
    assert mechanical_feature_sets("tactical_persist_h3")["mechanical_age_margin_state"] == (
        "tactical_age_bars",
        "tactical_boundary_margin",
        "tactical_state",
    )


def test_transition_alerts_merge_consecutive_low_persistence_scores() -> None:
    config = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
    fit_index = pd.date_range("2023-01-01", periods=100, freq="15min")
    test_index = pd.date_range("2024-01-01", periods=24, freq="15min")
    fit_risk = np.linspace(0.0, 1.0, len(fit_index), dtype=float)
    test_risk = np.zeros(len(test_index), dtype=float)
    test_risk[4:7] = 0.99
    test_risk[9] = 0.99  # within one hour of the prior warning, same episode
    fit = ContinuityPeriodData(
        timestamps_ns=fit_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        x=fit_risk[:, None],
        y=np.ones(len(fit_index), dtype=np.int8),
        feature_columns=("risk",),
        time_to_change_hours=np.full(len(fit_index), 4.0),
    )
    change = np.full(len(test_index), 4.0)
    change[4] = 2.0
    test = ContinuityPeriodData(
        timestamps_ns=test_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        x=test_risk[:, None],
        y=np.where(change <= 3.0, 0, 1).astype(np.int8),
        feature_columns=("risk",),
        time_to_change_hours=change,
    )
    metrics, episodes = transition_alert_episode_audit(
        _DeterministicProbabilityModel(),
        fit,
        test,
        fold_id="WF_TEST",
        target="tactical_persist_h3",
        config=config,
    )
    assert metrics["episodes"] == 1
    assert metrics["successful_episodes"] == 1
    assert len(episodes) == 1
    assert episodes.iloc[0]["points_in_episode"] == 4
