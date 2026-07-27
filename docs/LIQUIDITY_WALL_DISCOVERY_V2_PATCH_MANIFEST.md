# Liquidity Wall Discovery V2 Patch Manifest

## Added

- `src/data_feed/okx_liquidity_primitives.py`
- `tools/prebuild_okx_liquidity_primitives.py`
- `research/liquidity/liquidity_wall_discovery_v2/`
- `tests/liquidity_map/test_liquidity_primitives_v1.py`
- `docs/LIQUIDITY_PRIMITIVES_V1_RUNBOOK.md`

## Modified

- `src/research_common/liquidity_wall_discovery.py`
- `src/research_common/progress.py`

## Design decision

The prebuilt artifact is a low-semantic primitive cache, not a final wall cache.
It retains raw per-price depth/flow and multiple relative-depth summaries. Wall
candidate definitions remain mutable in the research layer.

The shared progress reporter also avoids duplicate 100% lines and clears stale terminal suffixes.

## Verification

- Primitive and legacy candidate paths agree on synthetic snapshots.
- Atomic day save/reload and checkpoint cleanup are tested.
- Primitive snapshots can drive the existing lifecycle engine without a Pandas
  DataFrame per snapshot.
- Liquidity test suite passes.
