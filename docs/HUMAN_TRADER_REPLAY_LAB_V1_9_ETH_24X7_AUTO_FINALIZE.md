# Human Trader Replay Lab V1.9 — ETH 24/7 Episode Profile

## Goal

Separate equity-style SOXL replay from 24/7 ETH replay.

## ETH profile

- Symbol: `ETH-USDT-SWAP`
- Replay eligibility: 24/7, including weekends.
- Random mode: random 30-minute-aligned blind start anywhere inside the selected Beijing-date range with local OKX 1m data.
- Specific mode: user selects an exact Beijing datetime.
- No 07:30 ET anchor and no 16:00 ET end.
- Replay may cross hours/days until local data ends or the user manually ends it.
- Once a trade has Entry/SL/TP and is filled, the first causal 1m SL/TP hit closes the trade and automatically finalizes the Episode.
- `+30m` and `+60m` are available, but the backend still advances order lifecycle minute-by-minute and stops exactly at the first bracket exit. It never reveals post-exit bars merely because a larger step was requested.

## SOXL profile

Unchanged:

- weekday Episode only;
- start at 07:30 America/New_York;
- decision window ends at 16:00 America/New_York;
- all available OKX off-hours/weekend chart context is still visible.

## Causality

HTF partial candles remain built only from already closed 1m children plus the known live minute open. Trade fill/exit ordering is evaluated on causally closed 1m bars.
