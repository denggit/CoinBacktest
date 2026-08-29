# Human Trader Replay Lab V1.5 — Single Chart + Forming HTF

## UX change

The four-pane layout was removed. There is one large main chart with fast timeframe switching:

- 30m / 15m = setup workflow
- 2m / 1m = execution workflow
- 5m / 1H / 4H / 1D remain available from the timeframe selector

All annotations continue to share one Episode event stream, so a liquidity line created on 30m is visible after switching to 15m/2m/1m. The event keeps its original source timeframe and anchor candle.

## Forming candle semantics

A replay cursor such as `07:31 ET` means the 07:30 1m bar has closed and the 07:31 1m open is known.

For a 30m candle that started at 07:30:

- at 07:30, the 30m candle appears immediately using the observable 07:30 1m open only;
- at 07:31, its high/low/volume incorporate the closed 07:30 1m bar and its latest price may use the observable 07:31 1m open;
- each minute repeats this causal update;
- at 08:00 the 07:30 30m candle becomes closed, while a new 08:00 forming candle can appear from the 08:00 open.

No current 1m high/low/close/volume is used before that 1m candle closes.

## Causality distinction

`available_time` remains the hard rule for **confirmed/closed** HTF features and future strategy research. The Replay UI is additionally allowed to display a clearly-marked forming candle because a real trader sees the evolving candle live. The two states must never be conflated:

- `is_closed=true`: confirmed historical candle, safe for closed-bar feature logic.
- `is_partial=true`: visual/behavioral replay state only; it may change as replay advances.

If the user places an annotation on a forming candle, the event stores `anchor_is_partial` and `anchor_observed_through` so future behavioral models know exactly what information was visible at decision time.

## Performance

Playback now renders only one canvas. Each step requests only the active main-chart timeframe and receives small bar upserts. The local 1m source and resampled day frames remain in memory, so replay avoids repeated full-table reads and avoids repainting four charts every minute.
