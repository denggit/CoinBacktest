from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.plugins import build_default_registry
from analyze_tool.plugins.liquidity_wall_discovery import LiquidityWallDiscoveryPlugin


def test_registry_contains_research_wall_overlay_plugin() -> None:
    ids = {item["id"] for item in build_default_registry().list_plugins()}
    assert "liquidity_wall_discovery_v1" in ids


def test_plugin_reads_bounded_research_segments_and_hides_ghosts(tmp_path: Path) -> None:
    start_ms = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)
    pd.DataFrame(
        [
            {
                "wall_id": 1,
                "side": "bid",
                "start_ms": start_ms,
                "end_ms": start_ms + 60_000,
                "price_low": 1800.0,
                "price_high": 1806.0,
                "morphology": "BAND",
                "is_ghost": 0,
                "shape_score": 60.0,
                "retention": 0.9,
                "age_seconds": 120,
            },
            {
                "wall_id": 2,
                "side": "ask",
                "start_ms": start_ms,
                "end_ms": start_ms + 60_000,
                "price_low": 1820.0,
                "price_high": 1822.0,
                "morphology": "POINT",
                "is_ghost": 1,
                "shape_score": 80.0,
                "retention": 0.7,
                "age_seconds": 60,
            },
        ]
    ).to_csv(tmp_path / "13_wall_overlay_segments.csv", index=False)
    df = pd.DataFrame(
        {"open": [1805], "high": [1807], "low": [1799], "close": [1804]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 08:00:00")]),
    )
    context = PluginRunContext(
        display_df=df,
        visible_df=df,
        request={"start": "2026-01-01 07:59:00", "end": "2026-01-01 08:02:00"},
    )
    result = LiquidityWallDiscoveryPlugin(tmp_path).run_with_context(
        context,
        {"minimum_shape_score": 35, "side": "all", "show_ghost": "no", "maximum_regions": 100},
    )
    assert len(result.price_regions) == 1
    assert result.price_regions[0].price_low == 1800.0
    assert result.summary["research_only"] is True
