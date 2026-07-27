# Liquidity Touch Rebound Feature Research V1

This patch removes the failed wall-annotation UI and starts a causal liquidity/rebound research line.

## Strategy hypothesis

- A broad, visually dense and position-stable lower bid-liquidity band is touched.
- Price reacts upward rather than fully consuming the band.
- Profit is taken before the nearest upper ask-liquidity band.
- Stop is placed below the lower wall boundary with a configurable buffer.

## Important separation

The script does not treat the current Analyze Tool wall boxes as truth. It builds broad candidate bands directly from exact 15m final snapshots and tests continuous features against future reaction labels and fully costed trade outcomes.

## Performance

Data is streamed one UTC day at a time through `OKXLiquidityMapLoader.iter_period_end_snapshot_days`. The persistent period-end cache is reused, and only the configured rolling history is kept in memory.

## Annotation removal

The patch restores the Analyze Tool to the pre-annotation server/static files. Run:

```text
python tools\remove_wall_annotation_feature.py
```

This deletes orphan annotation code/tests/docs but preserves any previously saved annotation data.

## Additional causal flow features

The event table also streams the existing 1-second strategy feature artifacts one day at a time. It reconstructs the first second where best ask reaches the frozen lower-band limit when possible, then measures:

- pre-touch bid addition/removal and cancellation risk;
- pre-touch withdrawal/ghost proxy;
- touch-minute estimated bid consumption and replenishment;
- replenishment-to-consumption ratio and depth-imbalance change;
- an absorption proxy combining consumed/replenished quantity with wall penetration.

Touch-minute flow is never used by passive same-touch limit families. It is actionable only after the minute completes, through next-open reclaim variants.
