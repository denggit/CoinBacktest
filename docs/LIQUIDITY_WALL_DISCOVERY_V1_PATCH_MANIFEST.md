# Liquidity Wall Discovery V1 Patch Manifest

## Added

- `src/research_common/liquidity_wall_discovery.py`
- `research/liquidity/liquidity_wall_discovery_v1/__init__.py`
- `research/liquidity/liquidity_wall_discovery_v1/01_liquidity_wall_discovery_research.py`
- `research/liquidity/liquidity_wall_discovery_v1/README.md`
- `analyze_tool/plugins/liquidity_wall_discovery.py`
- `tests/liquidity_map/test_liquidity_wall_discovery_v1.py`
- `tests/liquidity_map/test_liquidity_wall_discovery_plugin.py`
- `docs/LIQUIDITY_WALL_DISCOVERY_V1_RUNBOOK.md`
- `docs/LIQUIDITY_WALL_DISCOVERY_V1_PATCH_MANIFEST.md`

## Modified

- `analyze_tool/plugins/__init__.py`: registers the research-only wall overlay.

## Explicitly unchanged

- Existing `src/liquidity_map/wall_detector.py` behavior.
- Existing Liquidity Touch Rebound V1 and Absorption V2 results.
- Backtest strategies, TP/SL logic, fees, leverage and portfolio logic.
- AetherEdge.
- Git history; no commit was executed.
