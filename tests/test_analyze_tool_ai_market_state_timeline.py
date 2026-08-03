from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tool.ai_market_state_artifacts import ARTIFACT_SCHEMA_VERSION
from analyze_tool.plugins import build_default_registry
from analyze_tool.plugins.ai_market_state_timeline import AIMarketStateTimelinePlugin
from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.server import _json_safe
from src.ai_research.market_state_continuity.config import DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
from src.ai_research.market_state_continuity.state_cache import CACHE_SCHEMA_VERSION


STATE_COLUMNS = (
    "strategic_score",
    "tactical_score",
    "entry_score",
    "strategic_activity_score",
    "tactical_activity_score",
    "entry_activity_score",
    "activity_score",
    "strategic_long_enter_threshold",
    "strategic_short_enter_threshold",
    "strategic_long_exit_threshold",
    "strategic_short_exit_threshold",
    "strategic_raw_state",
    "strategic_state",
    "strategic_boundary_margin",
    "strategic_age_bars",
    "strategic_flip_rate_6h",
    "strategic_flip_rate_24h",
    "tactical_raw_state",
    "tactical_state",
    "tactical_boundary_margin",
    "tactical_age_bars",
    "tactical_flip_rate_6h",
    "tactical_flip_rate_24h",
    "entry_raw_state",
    "entry_state",
    "entry_boundary_margin",
    "entry_age_bars",
    "entry_flip_rate_6h",
    "entry_flip_rate_24h",
    "activity_raw_state",
    "activity_state",
    "activity_boundary_margin",
    "activity_age_bars",
    "activity_flip_rate_6h",
    "activity_flip_rate_24h",
    "strategic_tactical_alignment",
    "tactical_entry_alignment",
    "all_direction_alignment",
    "long_pullback_setup",
    "short_pullback_setup",
    "trend_momentum_long",
    "trend_momentum_short",
)


def _write_fake_state_cache(root: Path) -> None:
    target = root / "state_2024"
    target.mkdir(parents=True)
    index = pd.DatetimeIndex(["2024-01-01 00:15:00", "2024-01-01 00:30:00"])
    matrix = np.zeros((2, len(STATE_COLUMNS)), dtype=np.float32)
    positions = {column: i for i, column in enumerate(STATE_COLUMNS)}
    values = [
        {
            "strategic_score": 0.4,
            "tactical_score": -0.5,
            "entry_score": 0.1,
            "activity_score": 0.6,
            "strategic_state": 1,
            "tactical_state": -1,
            "entry_state": 0,
            "activity_state": 1,
            "strategic_boundary_margin": 0.2,
            "tactical_boundary_margin": 0.3,
            "entry_boundary_margin": 0.2,
            "activity_boundary_margin": 0.4,
            "strategic_age_bars": 192,
            "tactical_age_bars": 8,
            "entry_age_bars": 2,
            "activity_age_bars": 5,
            "strategic_tactical_alignment": -0.2,
            "tactical_entry_alignment": -0.05,
            "all_direction_alignment": -1 / 3,
            "long_pullback_setup": 0.3,
            "short_pullback_setup": 0.0,
            "trend_momentum_long": 0.0,
            "trend_momentum_short": 0.0,
        },
        {
            "strategic_score": 0.45,
            "tactical_score": 0.35,
            "entry_score": 0.4,
            "activity_score": -0.4,
            "strategic_state": 1,
            "tactical_state": 1,
            "entry_state": 1,
            "activity_state": -1,
            "strategic_boundary_margin": 0.25,
            "tactical_boundary_margin": 0.25,
            "entry_boundary_margin": 0.3,
            "activity_boundary_margin": 0.2,
            "strategic_age_bars": 193,
            "tactical_age_bars": 1,
            "entry_age_bars": 1,
            "activity_age_bars": 1,
            "strategic_tactical_alignment": 0.1575,
            "tactical_entry_alignment": 0.14,
            "all_direction_alignment": 1.0,
            "long_pullback_setup": 0.0,
            "short_pullback_setup": 0.0,
            "trend_momentum_long": 0.063,
            "trend_momentum_short": 0.0,
        },
    ]
    for row, mapping in enumerate(values):
        for column, value in mapping.items():
            matrix[row, positions[column]] = value
    np.save(target / "decision_times_ns.npy", index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    np.save(target / "features.npy", matrix)
    np.save(target / "states.npy", matrix)
    np.save(target / "targets.npy", np.zeros((2, 1), dtype=np.float32))
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "year": 2024,
                "timestamp_unit": "ns",
                "feature_columns": list(STATE_COLUMNS),
                "state_columns": list(STATE_COLUMNS),
                "target_columns": ["activity_persist_h3"],
            }
        ),
        encoding="utf-8",
    )


def _write_fake_prediction(root: Path) -> None:
    target = root / "activity_persist_oos_2024"
    target.mkdir(parents=True)
    index = pd.DatetimeIndex(["2024-01-01 00:15:00", "2024-01-01 00:30:00"])
    np.save(target / "decision_times_ns.npy", index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    np.save(target / "prediction.npy", np.asarray([0.8, 0.2], dtype=np.float32))
    np.save(target / "actual_persist.npy", np.asarray([1, 0], dtype=np.int8))
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "year": 2024,
                "fold_id": "WF_2024",
                "target": "activity_persist_h3",
                "architecture": "universal_ohlcv_lightgbm",
                "timestamp_unit": "ns",
            }
        ),
        encoding="utf-8",
    )


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:14:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:29:00",
            "2024-01-01 00:30:00",
        ]
    )
    return pd.DataFrame(
        {
            "open": [100, 100, 101, 101],
            "high": [101, 102, 102, 103],
            "low": [99, 99, 100, 100],
            "close": [100, 101, 101, 102],
            "volume": [1, 1, 1, 1],
        },
        index=index,
    )


def test_registry_contains_ai_market_state_timeline() -> None:
    by_id = {item["id"]: item for item in build_default_registry().list_plugins()}
    assert "ai_market_state_timeline_r03_3_3_1" in by_id


def test_ai_state_timeline_aligns_backward_without_future_leak(tmp_path: Path) -> None:
    cache_dir = tmp_path / "state"
    artifact_dir = tmp_path / "artifact"
    _write_fake_state_cache(cache_dir)
    _write_fake_prediction(artifact_dir)
    config = replace(DEFAULT_MARKET_STATE_CONTINUITY_CONFIG, cache_dir=str(cache_dir))
    df = _frame()
    result = AIMarketStateTimelinePlugin(config=config, artifact_dir=artifact_dir).run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            request={"symbol": "ETH-USDT-SWAP", "data_type": "normal", "timeframe": "1m"},
            meta={"symbol": "ETH-USDT-SWAP"},
        ),
        {"view_mode": "research", "show_transition_markers": "all"},
    )

    assert len(result.bands) == 5
    assert all(len(band.codes) == len(df) for band in result.bands)
    strategic = result.bands[0].codes
    activity = result.bands[3].codes
    outcome = result.bands[4].codes
    assert strategic == [None, 1, 1, 1]
    assert activity == [None, 1, 1, -1]
    assert outcome == [None, 1, 1, 0]
    probability = result.row_fields["activity_persist_h3_probability"]
    assert probability[0] is None
    assert probability[1] == 0.8
    assert probability[2] == 0.8
    assert probability[3] == 0.2
    assert result.summary["prediction_rows"] == 3
    assert result.summary["not_trade_signal"] is True
    assert result.summary["contains_future_outcome_audit"] is True
    assert result.summary["ui"]["brief_labels"] == ["战略", "战术", "入场/活跃"]
    # One tactical, entry and activity transition at 00:30; strategic remains unchanged.
    assert len(result.markers) == 3
    raw = json.dumps(_json_safe(result.as_dict()), ensure_ascii=False, allow_nan=False)
    assert "NaN" not in raw and "Infinity" not in raw


def test_ai_state_timeline_degrades_to_states_when_prediction_artifact_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "state"
    _write_fake_state_cache(cache_dir)
    config = replace(DEFAULT_MARKET_STATE_CONTINUITY_CONFIG, cache_dir=str(cache_dir))
    df = _frame()
    result = AIMarketStateTimelinePlugin(
        config=config,
        artifact_dir=tmp_path / "missing",
    ).run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            request={"symbol": "ETH-USDT-SWAP", "data_type": "normal", "timeframe": "1m"},
            meta={},
        ),
        {"view_mode": "overview", "show_transition_markers": "none"},
    )
    assert result.summary["state_rows"] == 3
    assert result.summary["prediction_rows"] == 0
    assert len(result.tracks) == 3
    assert all(value is None for value in result.row_fields["activity_persist_h3_probability"])


def test_ai_state_overview_uses_compact_row_fields(tmp_path: Path) -> None:
    cache_dir = tmp_path / "state"
    artifact_dir = tmp_path / "artifact"
    _write_fake_state_cache(cache_dir)
    _write_fake_prediction(artifact_dir)
    config = replace(DEFAULT_MARKET_STATE_CONTINUITY_CONFIG, cache_dir=str(cache_dir))
    df = _frame()
    result = AIMarketStateTimelinePlugin(config=config, artifact_dir=artifact_dir).run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            request={"symbol": "ETH-USDT-SWAP", "data_type": "normal", "timeframe": "1m"},
            meta={"symbol": "ETH-USDT-SWAP"},
        ),
        {"view_mode": "overview", "show_transition_markers": "none"},
    )
    assert result.summary["payload_mode"] == "compact-overview"
    assert "brief_reason_1" not in result.row_fields
    assert "strategic_age_days" in result.row_fields
    assert "activity_transition_risk_h3" in result.row_fields
    assert "strategic_score" not in result.row_fields


def test_ai_state_timeline_accepts_microsecond_display_index_against_nanosecond_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "state"
    artifact_dir = tmp_path / "artifact"
    _write_fake_state_cache(cache_dir)
    _write_fake_prediction(artifact_dir)
    config = replace(DEFAULT_MARKET_STATE_CONTINUITY_CONFIG, cache_dir=str(cache_dir))
    df = _frame()
    df.index = pd.DatetimeIndex(df.index.to_numpy(dtype="datetime64[us]"))
    assert str(df.index.dtype) == "datetime64[us]"
    result = AIMarketStateTimelinePlugin(config=config, artifact_dir=artifact_dir).run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            request={"symbol": "ETH-USDT-SWAP", "data_type": "trade_bar", "timeframe": "15m"},
            meta={"symbol": "ETH-USDT-SWAP"},
        ),
        {"view_mode": "overview", "show_transition_markers": "none"},
    )
    assert result.bands[0].codes == [None, 1, 1, 1]
    assert result.row_fields["activity_persist_h3_probability"] == [None, 0.8, 0.8, 0.2]
