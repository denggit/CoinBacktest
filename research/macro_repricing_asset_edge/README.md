# Macro Repricing Asset Edge Research

This directory is an independent, iterative research workspace. It does not
connect the macro monitor to a strategy and does not modify existing backtest
logic.

## Research question

Test whether changes in the local macro monitor contain a measurable forward
return or excursion edge for:

- SOXX (primary listed-equity exposure)
- SOXL (listed 3x ETF; kept separate from the OKX perpetual)
- QQQ
- ETH-USDT-SWAP
- XAU-USDT-SWAP (with gold futures used only as a clearly labelled long-history proxy)

The study deliberately separates two evidence tracks:

1. **True high-frequency monitor data**: CME FedWatch distribution, expected
   rate, US2Y, US10Y, curve and DXY from `macro_monitor.sqlite`. This is the
   closest match to the live alert, but the local history currently starts on
   2026-08-28 and is not yet large enough to establish a stable edge.
2. **Long-history proxy data**: FRED DGS2/DGS10, Yahoo Finance DXY and 30-Day
   Fed Funds Futures (`ZQ=F`) plus asset bars. This can test broader rate/DXY
   repricing regimes, but it is not historical CME FedWatch probability data.
3. **Scheduled-release five-minute data**: free Yahoo 60-day bars for explicit
   monthly Fed Funds Futures, Treasury futures, DXY, the 10Y yield quote and
   listed ETFs; short-window exact US2Y/US10Y yields from CNBC/Tradeweb; and
   direct OKX ETH/XAU perpetual bars. The study observes 5/10/15 minutes after
   a release, then stresses an additional 0/5/10 minutes of execution delay.

No proxy result may be labelled as a FedWatch result.

## Anti-look-ahead contract

- Macro timestamps are normalized to UTC.
- Beijing time is presentation-only.
- Intraday entry uses the first asset bar strictly after the observation.
- Window changes use an as-of observation at or before the requested lookback.
- Repeated polling rows are collapsed before event selection.
- Dense threshold crossings are thinned with a configurable event cooldown.
- SOXL is evaluated directly; SOXX results are never mechanically multiplied
  by three.
- `ZT=F` is always named as a futures-price diagnostic, never a 2Y yield.
- `ZQV26.CBT` is always named as a post-FOMC implied-rate proxy, never
  historical FedWatch probability.

## Layout

```text
config.py           thresholds, horizons and paths
data_sources.py     local SQLite adapters and free public downloads
features.py         macro panels, window changes and signal classification
event_study.py      causal alignment, excursions and statistical summaries
report.py           standalone HTML research report
run_research.py     reproducible command-line pipeline
tests/              deterministic unit tests
data/               ignored local research cache
outputs/            ignored generated evidence and report
notebooks/          reader-facing analysis notebook
```

## Run

From the repository root on Windows or Linux:

```bash
python -m research.macro_repricing_asset_edge.run_research --inventory-only
python -m research.macro_repricing_asset_edge.run_research --download
python -m research.macro_repricing_asset_edge.run_research
pytest research/macro_repricing_asset_edge/tests -q
```

The download path uses Alpaca SIP minute bars around true monitor events,
Yahoo daily histories for the longer proxy track, and public OKX data for
ETH/XAU. It also downloads the free 60-day Yahoo five-minute panel and CNBC's
short exact-yield window. Secrets are never written to research outputs.

## Current interpretation

- True FedWatch evidence remains one-day timing forensics.
- The daily proxy candidate remains `US2Y-led dovish repricing -> 5-session
  QQQ/SOXX strength`, but cost-stressed BH-FDR is still slightly above 5%.
- In the scheduled five-minute sample, six dovish proxy events were followed
  by a roughly -1.02% gross / -1.07% after-5-bp mean SOXX response over the
  next 60 minutes. SOXL was about -2.93% after 5 bp. The one FOMC case rose,
  while several data-release cases fell. With only six independent events,
  this is a counterexample to a simple `dovish = bullish` rule, not a confirmed
  short edge.
