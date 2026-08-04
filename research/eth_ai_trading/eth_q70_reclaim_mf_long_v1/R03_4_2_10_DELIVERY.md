# R03.4.2.10 Delivery — Soft-Structure Partial De-Risking and q70 Risk Migration

## Current project position

- R03.4.2.7: `FAIL_NO_ROBUST_NON_TIME_STRUCTURAL_EXIT`; `failed_reclaim` retained as the best deterministic working exit.
- R03.4.2.8A: strict healthy/recovered add subset failed and must not collapse trade frequency.
- R03.4.2.8B: static dual slots restored frequency but diluted every primary and reduced annual return too much.
- R03.4.2.9: `FAIL_NO_ROBUST_STRUCTURE_PROTECTION`; direct 15m Pivot hard stops were swept too often and are abandoned.
- R03.4.2.10: code delivered; local complete-data run pending.

## Frozen strategy components

- q70 ML opening pool unchanged.
- 3% disaster hard protection unchanged.
- deterministic `failed_reclaim` remains each virtual tranche's normal non-time exit.
- fixed six hours remains diagnostic only.
- 2026 remains sealed.

## Research policies

- `P0_single_1R`: one full-R position, skip occupied signals.
- `R1_soft_break_reduce025`: first proven, non-losing soft BROKEN transition closes 25%.
- `R2_soft_break_reduce050`: same with 50%.
- `M1_signal_migrate035`: later q70 may receive up to 0.35R after old exposure is physically reduced.
- `M2_signal_migrate050`: same with 0.50R.
- `H1_reduce025_then_migrate035`: reuse real capacity created by the 25% soft-break reduction, with same-open extra reduction only when required.

## Risk semantics

A new flat-to-position cycle fixes a one-R dollar budget at primary entry. Floating profit does not enlarge it. At a later q70 event, the simulator may use only capacity freed by actual old-unit closes; if that is insufficient, it closes old units at the same open before the new tranche is sized. Simultaneous cycle risk remains at or below one R.

## Causal boundaries

- 15m structure bars must be fully closed.
- Right-confirmed pivots only.
- Partial close becomes executable at the next 1m open.
- New q70 executes at its frozen delayed open.
- Entries are evaluated before an equal-time frozen exit, preserving P0 parity.
- Losing, BROKEN or pending-Failed-Reclaim roots cannot migrate risk.
- Maximum two simultaneous virtual tranches.

## Run

```bat
python research\eth_ai_trading\03_4_2_10_risk_migration.py
```

Report directory:

```text
data\reports\research\eth_ai_trading\03_4_2_10_risk_migration
```

Primary review files:

- `99_decision.md`
- `05_event_decisions.csv`
- `06_account_actions.csv`
- `08_account_trades.csv`
- `10_account_policy_summary.csv`
- `11_policy_gate.csv`
- `gpt_review_pack.zip`

## Pass criteria

A policy must use one identical rule in both years, retain at least 95% of P0 return in each year, equal or exceed combined P0 return, remain positive under 2x/3x costs and 1/3/5-minute delay, stay positive without the top ten winners, and keep MDD within 1.10x P0. Migration policies must additionally reach at least 70% q70 coverage and about 25 tranches/month with no BROKEN-state migration and near-zero losing-position migration.

## What remains after this stage

- Run the complete 2024–2025 audit locally.
- If a policy passes, continue to entry/MAE optimization and score-tier risk allocation.
- If all fail, retain P0 and stop this capacity branch rather than adding complexity.
- Re-audit the final non-time exit chain, then open 2026 and design the AetherEdge plugin.

## Delivery tests

- `PYTHONPATH=. pytest -q tests/ai_research/test_long_tail_risk_migration.py` → 9 passed.
- Related R03.4.2.7/8A/8B/9/10 group → 41 passed.
- `PYTHONPATH=. pytest -q tests/ai_research tests/data_feed` → 180 passed.
- Full `pytest -q` remains blocked at collection by five pre-existing missing liquidity/analyze-tool modules.
- Import-boundary audit remains blocked by existing Swing-Low/liquidity research imports; this stage adds no research-script import.
