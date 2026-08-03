# R03.4.2.4 Patch Manifest

## Goal

Audit the frozen q70 opening pool across both WF_2024 and WF_2025 before any further non-time exit research.

## Key changes

- Decouples q70/q90 opening-pool validation from all holding-path classifiers.
- Guarantees that a missing recoverable-drawdown model cannot suppress the WF_2024 baseline.
- Keeps the R03.4.1 LightGBM opening model, calibration windows and next-minute execution frozen.
- Compares q70 and q90 under 1x/2x/3x cost and 1/3/5-minute entry delay.
- Separately audits the incremental q70-to-q90 score band instead of crediting q70 for q90-grade events.
- Reports annual, quarterly and monthly expectancy, PF, win rate, compounded return, MDD, Top-10 concentration and losing streaks.
- Requires q70 to beat q90 total compounded profit in both years before declaring a stable expansion.
- Treats the six-hour close only as a frozen opening-edge benchmark, not as the final live exit.
- Keeps 2026 sealed and does not load the abandoned market-state model.
