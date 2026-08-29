# Human Trader Replay Lab V1.3 — OKX SOXL source

## Why this revision exists

The Alpaca SOXL history used in V1.2 has gaps that are undesirable for manual replay. V1.3 switches the replay source back to the local OKX `SOXL-USDT-SWAP` 1m table while keeping the user workflow unchanged.

## Frozen workflow

- Symbol: `SOXL-USDT-SWAP`.
- Source: `src.data_feed.okx_loader.OKXDataLoader`, local data only.
- Decision timezone: `America/New_York`.
- Each episode starts at 07:30 ET.
- Market-open reference is 09:30 ET.
- Episode ends at 16:00 ET.
- Saturday and Sunday episodes are forbidden.
- Default charts: 30m / 15m / 2m / 1m.
- Magnet defaults ON and snaps to candle O/H/L/C.
- Shared annotations preserve anchor timestamp and source timeframe.

## Performance change

`OKXDataLoader.load_local_data()` is called once per server process for the short SOXL history. The resulting 1m frame is converted to New York wall time and held in memory. Each replay day slices that frame once, then caches resampled frames. Playback sends only newly available bars.

This avoids the V0 pattern of repeated database reads + repeated full resampling + repeated full-chart transfers.

## Causal rule

For every timeframe:

```text
available_time = bar_start_time + timeframe
```

A bar is sent to the browser only when:

```text
available_time <= decision_cursor
```

The server may cache future raw bars for speed, but those bars remain behind the availability gate.

## Tests

Dedicated suite:

```bash
PYTHONPATH=. pytest tests/human_replay_lab -q
```

V1.3 dedicated result during patch build: `11 passed`.
