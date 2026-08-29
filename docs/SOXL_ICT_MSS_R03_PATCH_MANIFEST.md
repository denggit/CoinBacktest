# SOXL ICT/MSS R03 — Spot/Perpetual Audit + Long-History Runner

## Purpose

Use the locally prebuilt Alpaca `SOXL` SIP split-adjusted 1m history as a long-history research proxy only after a structural overlap audit against the available OKX `SOXL-USDT-SWAP` 1m history.

## Key changes

1. R02 state-machine research now supports `--data-source okx|alpaca` without changing MSS/FVG/entry/SL/TP semantics.
2. Both sources are clipped to New York `04:00-16:30` before liquidity or structure construction; OKX's extra 24h session cannot influence the strategy.
3. Added `03_spot_perp_overlap_audit.py` with fixed proxy-validation gates for minute returns, rebased price paths, external sweeps and R02 setup agreement.
4. Added causal densification for omitted no-trade US-stock minutes. It is forward-only inside the same day, creates zero-volume synthetic bars, and never fills before the first or after the last observed print.
5. Added an Alpaca-specific data-quality gate for sparse extended-hours data.
6. Optimized repeated New York day slicing using DatetimeIndex binary-search bounds instead of full-table scans, which is critical for the 1.4M-row history.
7. Alpaca local range queries now stay inside the data-feed adapter and create a timestamp SQLite index lazily for repeated overlap reads.
8. Includes the prior `.env` credential fallback fix.

## Run 1 — overlap audit

```text
python research\ict\soxl_premarket_mss_fvg\03_spot_perp_overlap_audit.py --start-date 2026-05-20 --end-date 2026-06-30
```

Report:

`data/reports/research/ict/soxl/mss/r03_spot_perp_overlap_audit/`

If the console verdict is `FAIL`, stop and review the audit before treating Alpaca as an OKX-perpetual proxy.

## Run 2 — long-history R02 proxy study

After `PASS` (or consciously accepted `CAUTION` for hypothesis screening only):

```text
python research\ict\soxl_premarket_mss_fvg\02_sweep_episode_state_machine_research.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2019-01-02 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r02_state_machine_alpaca_2019_2026
```

## Validation completed

- R02 self-test: PASS
- R03 self-test: PASS
- targeted regression suite: 21 passed
- no research-to-research imports in new/modified research entrypoints
