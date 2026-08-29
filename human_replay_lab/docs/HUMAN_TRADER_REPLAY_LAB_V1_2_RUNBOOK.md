# Human Trader Replay Lab V1.2 — SOXL Manual Setup Capture

## Frozen workflow

- Symbol: SOXL only
- Source: existing `src.data_feed.alpaca_stock_loader.AlpacaStockLoader`
- Feed / adjustment: SIP / split
- Replay timezone: `America/New_York`
- Episode start: 07:30 ET
- Market open reference: 09:30 ET
- Episode end: 16:00 ET
- Only real trading weekdays are sampled; weekends and local-data holidays are rejected.
- Default panes: 30m / 15m setup, 2m / 1m execution.

## Start

From CoinBacktest root:

```bash
python human_replay_lab/server.py --host 127.0.0.1 --port 8775
```

Open `http://127.0.0.1:8775`.

If the Alpaca database is under a non-default data directory:

```bash
python human_replay_lab/server.py --data-dir <path-to-data> --host 127.0.0.1 --port 8775
```

## Shared marking + magnet

All horizontal marks are shared across the four panes. The source timeframe, source pane and anchor candle time are kept in the event payload.

The magnet toggle is ON by default. Clicking a candle snaps the selected price to the nearest O/H/L/C value of that exact candle. The event also stores:

- `magnet_enabled`
- `snap_field` (`O/H/L/C`)
- `raw_clicked_price`
- `anchor_time`
- `anchor_timeframe`
- `source_pane`

Turning the magnet OFF restores free-price selection.

## Causal rules

Charts contain closed bars only. A bar starting at `t` becomes visible at `t + timeframe`.

The market-order simulator may use the current 1m **open** at the decision cursor, because that open is known at that minute boundary. It may not use that current minute's final high/low/close.

## Playback performance

V1 previously refreshed all four full chart windows after every 1-minute step. That repeatedly hit SQLite date-range queries, resampled the same history, serialized thousands of bars, and redrew all data.

V1.2 changes playback to:

1. prefetch the replay day's required 1m history once;
2. keep causal resampled frames in memory;
3. on each step return only bars whose `available_time` became visible since the previous cursor;
4. append those bars in the browser instead of refetching all four 700-bar windows;
5. event/trade clicks update the local event stream without forcing a market-data refresh.

Future bars can exist only in the server cache; they are inaccessible to the browser until their `available_time <= cursor` gate passes.
