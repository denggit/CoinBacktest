# ETH AI Trading — Open Items and Roadmap

Updated through: **R03.4.2.18 archive closeout**

## Archived branch

```text
ETH Q70 Reclaim MF Long V1
Lifecycle: ARCHIVED_AFTER_SEALED_HOLDOUT_FAILURE
Live approved: NO
Capital allocation: 0
Archive: research/eth_ai_trading/archived_models/eth_q70_reclaim_mf_long_v1
```

Binding facts:

- 2024–2025 development account was strong.
- January–June 2026 untouched holdout failed.
- July recovery did not reverse the consumed seal.
- R03.4.2.17 diagnosed broad score drift and found no simple causal 1D/4H gate repair.
- No further q70/stop/exit/state-gate optimization is allowed on opened 2026 data.

## Next independent model

Working title: **ETH Trend Pullback Continuation Long/Short V1**.

This is a new branch, not C2 V2 and not a breakout-chasing system. Initial research contract:

1. Use 1D/4H to estimate trend direction, persistence and remaining runway.
2. Use 1H/30m to identify an orderly pullback, compression or absorption phase.
3. Use 15m/5m reclaim and re-acceleration for entry eligibility.
4. Execute on the next observable 1m path.
5. Start with one position and no add-ons.
6. Anchor the hard stop to local structure plus a causal volatility buffer.
7. Predeclare a maximum stop distance and skip trades that require a wider stop.
8. Size notional from actual hard-stop distance and account risk; leverage is margin efficiency only.
9. Build Long and Short as separate outputs and evaluate them independently before portfolio combination.

## Research goal

Capture a meaningful portion of multi-day 3%–15% moves without buying the breakout high or shorting the breakdown low. The desired improvement comes from a better entry location and remaining-trend estimate, not from using a dangerously wide stop or pretending high leverage increases Edge.

## Forbidden

- reusing the V1 q70 threshold or score as the new model's core;
- repairing V1 on 2026 and calling it validated;
- breakout entry with an unconstrained far structural stop;
- choosing leverage first and reverse-engineering risk afterward;
- adding to losing positions;
- large parameter-grid optimization.
