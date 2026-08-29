# Human Trader Replay Lab V1.2

Interactive SOXL manual-trading replay capture for CoinBacktest.

- 07:30 ET start on real trading weekdays only
- 30m / 15m setup panes + 2m / 1m execution panes
- shared Liquidity / Delivery Target / horizontal marks / SL / TP
- optional O/H/L/C magnet, enabled by default
- causal closed-bar visibility
- incremental cached playback
- complete Episode event history in SQLite

Run from CoinBacktest root:

```bash
python human_replay_lab/server.py --host 127.0.0.1 --port 8775
```

See `docs/HUMAN_TRADER_REPLAY_LAB_V1_2_RUNBOOK.md`.
