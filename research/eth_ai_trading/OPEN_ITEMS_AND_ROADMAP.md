# ETH AI Trading — Open Items and Roadmap

Updated through: **R03.4.2.10 code delivery; R03.4.2.9 empirical result complete**

## Active

### R03.4.2.10 — Soft-structure partial de-risking and q70 risk migration

The Pivot hard-stop route is closed after R03.4.2.9. The active stage keeps:

- q70 ML entry unchanged;
- 3% disaster hard protection;
- deterministic `failed_reclaim` as the normal non-time exit;
- one full-R primary whenever the account cycle starts flat;
- maximum two virtual Tranches;
- 2x/3x cost, 1/3/5-minute delay and sealed 2026.

Pre-registered policies:

- P0: one full-R position, overlapping q70 events skipped.
- R1/R2: real 25%/50% partial close on the first proven soft structure break while non-losing.
- M1/M2: migrate at most 0.35R/0.50R to a later q70 by reducing the healthy, non-losing old tranche first.
- H1: combine a 25% soft-break reduction with later 0.35R migration.

A risk cycle uses the one-R dollar budget fixed when its first primary opens. Floating profit cannot increase the budget. Old and new Tranches keep independent Failed-Reclaim exits.

## Immediate decision after R03.4.2.10

### Pass branch

Retain only a same-rule policy that:

- preserves at least 95% of P0 return in both 2024 and 2025;
- has combined cross-year return at least equal to P0;
- for migration policies restores at least 70% q70 coverage and about 25+ Tranches/month;
- remains positive under 3x costs and 3/5-minute delay;
- stays positive without the top ten winners;
- keeps MDD within 1.10x P0 and cycle risk within 1R;
- has zero BROKEN-state migration and near-zero losing-position migration.

Then proceed to entry/MAE research, score-tier risk allocation and final non-time exit re-audit.

### Fail branch

If all real partial-close and migration policies fail, stop the current account-capacity branch. Do not add a third Tranche, permit averaging down, lower the return-retention gate or choose rules by year. Retain P0 and move to entry execution plus complementary Sleeves.

## Final strategy work still incomplete

- Empirically validated account-capacity policy, or formal retention of P0 single-position capacity.
- Entry execution that reduces MAE without deleting most q70 signals.
- Frozen q70/q80/q90 risk budgets.
- Final non-time profit protection and exit chain.
- Complete account-level fee/slippage/delay backtest.
- Complementary Long/Short and higher-frequency Sleeves.
- 2026 sealed holdout.
- AetherEdge plugin, shadow-live and monitoring.

## Kill rules

- No historical-year-specific policy selection.
- No lower MDD accepted in exchange for excessive return loss.
- No PF improvement accepted if q70 coverage collapses.
- No third Tranche, static dilution, Pivot-stop grid or Martingale recovery.
- No fixed holding duration as final exit.
- Stop a branch when added complexity does not improve executable account profit.
