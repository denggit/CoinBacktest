# ETH External Strategy Factory / Tournament V1

This research line intentionally does **not** start from abstract edge discovery, ML prediction metrics, or parameter mining.
It starts from externally specified trading systems and asks one practical question:

> Does the frozen strategy survive ETH-USDT-SWAP causal replay, real costs, stress, and portfolio combination well enough to justify live migration?

## One-command run

Windows:

```bat
python research\eth_strategy_factory\00_strategy_tournament.py
```

Unix:

```bash
python research/eth_strategy_factory/00_strategy_tournament.py
```

## Frozen evaluation protocol

- warmup: 2022-01-01
- evaluation: 2023-01-01 through 2025-12-31
- 2026: hard sealed; this V1 script refuses to open it
- round-trip cost: 0.11%
- cost stress: 2x / 3x
- execution stress: +1m / +2m beyond the conservative next-observable-open rule
- same-minute stop+target: stop first
- high-timeframe data: use only after `available_time`
- no parameter feedback from tournament results

## Strategy families

S01 Donchian ensemble trend; S02 vol-scaled MA trend; S03 Bollinger/RSI regime strategies; S04 Turtle System 2; S05 footprint absorption reversal; S06 CVD exhaustion fade; S07 flow-confirmed breakout; S08 quarter-hour opening order imbalance.

See `SOURCE_CATALOG.md` for what is source-faithful versus our explicitly frozen engineering conversion.

## Selection

Survivor gate is declared before results. A base survivor requires positive return, PF > 1, MDD <= 20%, at least 2 positive years, and minimum activity. A robust survivor also needs positive return under 2x costs.

Survivors are ranked lexicographically by the user's live-trading priorities:

1. maximum flat/no-trade days (smaller)
2. maximum consecutive losing days (smaller)
3. MDD (smaller)
4. CAGR (larger)
5. total return (larger)

Portfolio weights are not optimized. V1 uses equal daily-return sleeves among top survivors, preferring family diversity.


## R02 — continuous position management

V1 full-data results favored simple trend/breakout systems over microstructure event systems. The active next stage is therefore **continuous risk-managed net exposure**, not isolated entry/TP/SL trades.

Run on Windows:

```bat
python research\eth_strategy_factory\02_continuous_risk_managed_portfolio.py
```

R02 keeps 2026 sealed and compares three frozen portfolio-construction variants. See `R02_CONTINUOUS_PORTFOLIO_SPEC.md`.

## R03 — Source-Locked Trend Replication

R02 showed that adding our own portfolio heuristics too early can dilute stronger simple source strategies. R03 therefore resets the process: replicate complete public source systems first, make required ETH adaptations explicit, and postpone portfolio construction.

Run on Windows:

```bat
python research\eth_strategy_factory\03_source_locked_trend_replication.py
```

R03 tests Zarattini/Pagani/Barbon Donchian long-only and long-short, Moskowitz/Ooi/Pedersen 12M TSMOM, and Original Turtle System-2 core. See `R03_SOURCE_LOCKED_SPEC.md` for fidelity boundaries and disclosed adaptations. 2026 remains hard sealed.

## R04 — Turtle Path Atlas

R03 found the source-locked Turtle System 2 profitable but too high-drawdown. R04 does **not** tune its entry or exit. It reconstructs every Turtle episode minute-by-minute and studies how risk evolves after entry: MFE/MAE in N, speed to Unit 2/3/4, pyramid-then-fail paths, and profit giveback before the 20D exit.

Windows:

```bat
python research\eth_strategy_factory\04_turtle_path_atlas.py
```

2023-2024 are path discovery, 2025 is validation, and 2026 remains hard sealed. Path labels are retrospective diagnostics only; they cannot be used as live features without a later causal rule and independent validation.
