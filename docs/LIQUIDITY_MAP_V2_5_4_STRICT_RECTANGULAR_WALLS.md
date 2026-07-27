# Liquidity Map V2.5.4 — Deep-Blue Strict Rectangular Walls

## Scope

This patch changes only the liquidity-wall detector and wall overlay. It keeps:

- one final order-book snapshot per completed chart bar;
- the latest snapshot for the current live bar;
- the causal 24-hour robust depth scale;
- period-end V2 cache reuse;
- compact JSON and gzip response handling from V2.5.3.

No raw Liquidity Map rebuild is required. Existing `period_end_v2` cache files remain valid.

## Fixed wall shape

A detected wall is emitted as one fixed rectangle:

- one start time: causal confirmation time;
- one end time: wall invalidation/end time;
- one stable price band extracted from recurring price bins;
- no stepped outline, no irregular polygon, no dashed bridge.

The stable band keeps price bins that recur through at least 60% of the confirmed lifecycle by default. This extracts the persistent blocking core rather than the union of every noisy edge.

## Stricter density requirements

A main wall must now satisfy all of the following default conditions:

- 15%+ row coverage over time: at least 70%;
- 30%+ row coverage over time: at least 55%;
- 50%+ row coverage over time: at least 20%;
- historical average depth ratio: at least 16%;
- current depth ratio: at least 12%;
- price-span coverage: at least 70%;
- full rectangle 15%+ matrix occupancy: at least 65%;
- full rectangle 30%+ matrix occupancy: at least 30%;
- current column 15%+ occupancy inside the rectangle: at least 55%;
- at most one missing/light price bin inside a cluster;
- two consecutive qualifying chart bars before confirmation;
- minimum strength score: 35.

These checks target the user-visible problem where a yellow box enclosed mostly white cells because only a few intermittent rows were deep.

## Wall must remain ahead of price

A resting wall is only valid when it is outside the market:

- bid wall: strictly below the reconstructed market midpoint;
- ask wall: strictly above the reconstructed market midpoint;
- default minimum clearance: one price bin.

If the midpoint enters/crosses an active wall, its lifecycle ends immediately instead of being bridged as a temporary fade.

The Analyze Tool additionally clips the displayed rectangle at the first chart bar whose traded OHLC range touches the wall:

- bid wall ends when candle low reaches its upper boundary;
- ask wall ends when candle high reaches its lower boundary.

This guarantees that the displayed rectangle does not wrap around the price path. Touch/reaction analysis can be implemented as a separate lifecycle stage later.

## Overlay appearance

- border: deep blue `#123A73`;
- shape: single rectangle;
- very light blue fill for visibility;
- no chart labels;
- no forming/fading dashed geometry.

## Performance

This patch preserves the V2.5.3 performance path:

- daily period-end cache;
- one shared period-end matrix for heatmap and walls;
- reused causal depth ratio;
- compact columnar heatmap JSON;
- gzip for large responses;
- fewer wall objects and simpler rectangle rendering.

## Verification

- `PYTHONPATH=. pytest tests/liquidity_map -q` → 74 passed
- `node --check analyze_tool/static/app.js` → passed
- `PYTHONPATH=. python analyze_tool/selftest.py` → passed

The repository-wide suite still stops during collection on two pre-existing unrelated panic-module import errors.
