# Human Trader Replay Lab V1.11

Multi-symbol OKX manual-trader replay capture tool for CoinBacktest.

## V1.11 chronological / from-date replay

The default start mode is now **从日期顺序 Replay** instead of random sampling.

For `ETH-USDT-SWAP` and `XAU-USDT-SWAP`:

1. choose a Beijing start date/time (for example `2026-01-01 00:00`);
2. the first Episode starts at the first locally available 1m bar at or after that timestamp;
3. after the Episode is closed (including automatic TP/SL finalization), press **继续下一 Episode**;
4. the next Episode starts at the first locally available 1m bar strictly after the previous Episode's final cursor;
5. no random jump is introduced between Episodes.

This is deliberately chronological blind replay. The service may scan timestamps to skip genuine local-data gaps, but it does not expose future OHLC values while choosing the next available row.

For SOXL/session-style symbols, sequential replay preserves the pre-open workflow: the first available weekday on/after the selected date starts at `07:30 America/New_York`; subsequent Episodes advance to the next locally available weekday at 07:30 ET.

To restart a sequence from a different point, change the sequential start date/time. That clears the front-end continuation pointer and the next Episode starts from the newly selected point.

### Extend local ETH 1m data through 2026-08-24

From the CoinBacktest repository root:

```bash
python tools/prebuild_okx_ohlcv.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date "2026-07-01 00:00:00" --end-date "2026-08-24 23:59:59"
```

The command uses the existing `src.data_feed.OKXDataLoader`; it merges/deduplicates into the local OHLCV cache rather than adding a separate data path.


## Defaults

- Symbol: select any local OKX `*_1m` table, including `SOXL-USDT-SWAP`, `ETH-USDT-SWAP`, and `XAU-USDT-SWAP`
- Data: local OKX 1m through `src.data_feed.okx_loader.OKXDataLoader`
- Episode profile: `SOXL-USDT-SWAP` uses weekdays + `07:30 America/New_York`; `ETH-USDT-SWAP` / `XAU-USDT-SWAP` use 24/7 replay including weekends.
- UI timestamps display Beijing time (`Asia/Shanghai`) everywhere; SOXL's New York session conversion handles DST automatically.
- One main chart, quick-switch: `30m / 15m / 2m / 1m` (plus 5m/1H/4H/1D)
- Chart history: **all local OKX bars** in the lookback; chart context is not filtered by session/weekend
- Magnet: on by default, snaps click price to candle O/H/L/C
- Entry: LIMIT by default; MARKET remains available


## Trade lifecycle + outcome recorder

V1.7 records the complete trade lifecycle instead of stopping at an entry marker.

For a filled trade the active event stream now carries:

- `ORDER_FILLED`
- `TRADE_OPEN`
- attached `SL` / `TP`
- `TAKE_PROFIT_HIT` or `STOP_LOSS_HIT`
- `TRADE_EXIT_AMBIGUOUS` when one 1m OHLC bar touches both SL and TP
- `TRADE_CLOSED` with exit reason, gross/net return, R multiple, MFE, MAE, and holding minutes

A 1m bar that touches both SL and TP has unknown intrabar ordering. The replay never awards the optimistic TP in that case; it records the ambiguity and uses a conservative stop-price resolution. A resting limit fill also excludes its own trigger bar from automatic SL/TP outcome detection because the OHLC path before/after the intrabar fill is unknowable.

Round-trip net return uses the project default `0.11%` full buy/sell cost assumption.

Older V1.6.x Episodes remain usable. If an old `LONG` / `SHORT` entry exists without V1.7 lifecycle rows, loading the Episode reconstructs it as a legacy active trade and catches up any SL/TP outcome that had already occurred by the current cursor.

## Autosave vs End Episode

Every user action is persisted to SQLite immediately when it happens. **You do not need to press End Episode to save the work.**

`结束 Episode` now only:

1. catches up any SL/TP outcome through the current cursor;
2. appends `EPISODE_SUMMARY`;
3. marks the Episode `closed` so it is treated as a finished training sample.

If the browser/server is closed before pressing End Episode, the existing labels/orders/fills are still in SQLite; the Episode simply remains `active` and does not yet have a final summary.

## Annotation correction

Mistaken chart annotations can be corrected directly from the Decision Timeline.

- `LIQUIDITY` (BSL / SSL / Other), `TARGET`, and shared `MARKER` rows show a `删除` button.
- Deleting immediately removes the annotation from the active chart and current decision state.
- The original event is soft-deleted (`is_active=0`) rather than physically removed.
- JSON export keeps the original under `discarded_events` and records an active `ANNOTATION_DELETE` correction event referencing the original event id.
- Trade / fill / order lifecycle events cannot be deleted through this annotation action.

## Manual order ticket

V1.6 no longer forces the chart-selected price to be the order price.

- `Entry`: type any LIMIT price manually.
- `SL`: optional manual attached stop price.
- `TP`: optional manual attached take-profit price.
- A chart selection can be copied into Entry / SL / TP with separate buttons.
- For LONG: `SL < Entry < TP` when those bracket prices are supplied.
- For SHORT: `TP < Entry < SL` when those bracket prices are supplied.
- Pending LIMIT orders show Entry + pending SL/TP on the chart.
- Once the limit fills causally, attached SL/TP become normal timeline protection events.

## Replay rewind

Playback supports `-1m / -5m / -15m`.

Rewind is branch-aware rather than destructive:

1. move the decision cursor backward, never before the Episode's 07:30 start;
2. events strictly after the new cursor are marked inactive and removed from the current training trajectory;
3. abandoned events are retained in JSON export under `discarded_events`;
4. a pre-existing resting LIMIT order becomes active again if its later fill/cancel was archived by the rewind.

This avoids mixing hindsight decisions from the abandoned future path into the new replay branch.

## Forming higher-timeframe candles

The main chart behaves like a live chart while replay advances in 1m steps:

- fully closed HTF candles remain causal: `bar_available_time = bar_start + timeframe`;
- the current 2m/5m/15m/30m/1H/4H/1D candle is also shown as `is_partial=true`;
- its OHLCV is built only from already-closed 1m children;
- at the exact current minute, only that 1m candle's **open** may be used because the open is known at the cursor; its future high/low/close/volume are never used;
- the same partial candle is upserted/replaced every replay minute and becomes a normal closed candle once its scheduled close is reached.

## Run

```bash
python human_replay_lab/server.py --host 127.0.0.1 --port 8775
```

Then open `http://127.0.0.1:8775`.

## V1.9 ETH 24/7 profile

`ETH-USDT-SWAP` no longer inherits the SOXL session clock. ETH Episodes can start at any Beijing datetime (or a random 30-minute-aligned point), include weekends, cross calendar days, and continue until local data ends or the trade resolves. When an ETH trade with SL/TP reaches either bracket, the Episode is automatically finalized at that exact causal replay minute. SOXL keeps the weekday 07:30-16:00 ET profile.


## V1.10 XAU 24/7 profile

`XAU-USDT-SWAP` uses the same continuous blind-replay workflow as ETH: Beijing-time random/specific starts, weekends allowed, no equity-session cutoff, and automatic Episode finalization when a filled trade reaches TP or SL. XAU only appears in the Symbol selector after its local `XAU_USDT_SWAP_1m` table exists in `data/crypto_history.db`.

Populate local XAU 1m OHLCV through the existing data-feed loader (Windows/Unix compatible from repo root):

```bash
python tools/prebuild_okx_ohlcv.py --symbol XAU-USDT-SWAP --timeframe 1m --start-date 2026-01-15 --end-date 2026-08-24
```

The start date follows OKX's XAUTUSDT -> XAUUSDT rename date. Older history used the retired `XAUT-USDT-SWAP` instrument id and is intentionally not merged automatically in this Replay Lab patch.

## V1.12 six persistent editable timeframe slots

The single main chart now has six configurable timeframe shortcut slots instead of four fixed buttons.

Default slots:

```text
30m / 15m / 5m / 2m / 1m / 4H
```

Each slot can be changed independently to any Replay Lab supported timeframe (`1m`, `2m`, `5m`, `15m`, `30m`, `1H`, `4H`, `1D`). The slot configuration and active slot are browser-local UI preferences and persist across:

- switching to another shortcut and returning;
- `Fit` (Fit only resets zoom/pan);
- new Episodes;
- Symbol changes;
- browser refresh/reopen in the same browser profile.

Changing slot 1 from `30m` to `4H`, for example, permanently makes slot 1 a `4H` shortcut until the user changes or resets it. `重置6周期` restores the defaults above.
