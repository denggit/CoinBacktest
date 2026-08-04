# R03.4.2.15 Patch Manifest

## Goal

Freeze the best C2 sleeve and perform the last pre-holdout account, tail-risk, exchange-lot and live-state audit before opening 2026.

## Frozen inputs

- q70 immediate next-open entry.
- Equal one-R for every q70 score tier.
- Real 2% exchange-side hard stop.
- 1.5% completed-close soft-failure exit.
- Deterministic `failed_reclaim` non-time exit.
- No occupied-signal add, score-based sizing or fixed-time take profit.

## New outputs

- Continuous 2024-2025 OOS account without annual reset.
- Cost/delay stress matrix.
- Monthly and quarterly stability.
- Losing streak, inactivity, holding and drawdown-duration diagnostics.
- Top-10 removal and concentration.
- Net-R fee/slippage reserve.
- OKX whole-contract sizing by initial equity.
- Monthly audit / shadow retrain / quarterly release governance.
- Restart-safe AetherEdge state contract.

## Holdout contract

2026 is not loaded. Passing this stage only permits one-time sealed validation; it does not permit another parameter round.
