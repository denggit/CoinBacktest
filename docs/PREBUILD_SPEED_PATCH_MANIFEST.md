# Prebuild Speed Safe Patch Manifest

## Modified

- `src/data_feed/okx_tick_loader.py`
- `src/data_feed/okx_books_loader.py`
- `src/data_feed/okx_range_bar_loader.py`
- `src/data_feed/okx_trade_bar_loader.py`
- `src/data_feed/okx_liquidity_map_loader.py`
- `src/data_feed/okx_liquidity_primitives.py`
- `src/liquidity_map/replay.py`
- `src/liquidity_map/builder.py`
- `src/liquidity_map/store.py`
- `tools/prebuild_okx_offline_liquidity_map.py`
- `tools/prebuild_okx_liquidity_primitives.py`
- `tools/prebuild_okx_liquidity_period_end_cache.py`
- `tools/prebuild_okx_trade_bars.py`

## Tests

- `tests/liquidity_map/test_liquidity_primitives_v1.py`
- `tests/test_prebuild_speed_paths.py`

## Documentation

- `docs/PREBUILD_SPEED_AND_RECOVERY.md`

No strategy, signal, label, entry, exit, fee, leverage or portfolio code is changed.
