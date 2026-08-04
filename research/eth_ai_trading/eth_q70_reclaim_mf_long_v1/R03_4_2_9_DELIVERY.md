# R03.4.2.9 Delivery — Structural Protection and Dynamic Risk Release

## Status

- Code and tests delivered.
- Empirical 2024–2025 result requires the local Trade Bar dataset.
- Current frozen opening pool: q70.
- Current working exit: deterministic `failed_reclaim` plus 3% disaster protection.
- 2026 remains sealed.

## Why this stage exists

R03.4.2.8B showed that static 0.5/0.5 or 0.65/0.35 slots restore frequency and reduce MDD, but permanently dilute every primary and reduce annual return by roughly 21%–23% versus P0. The second Tranche itself was not the main failure. The risk allocation method was.

R03.4.2.9 therefore keeps every primary at 1R and tests whether a real, enforceable structural stop can release risk for a later q70 signal.

## Pre-registered protection policies

1. `S0_disaster_only`: 3% outer hard stop only.
2. `S1_latest_confirmed`: latest causally confirmed structural floor minus buffer.
3. `S2_lagged_confirmed`: the prior confirmed structural floor minus buffer.

All stops:

- use only completed 15m structure bars;
- activate at the next 1m open;
- can rise but never widen;
- fill at the open on a gap through the stop;
- leave `failed_reclaim` unchanged unless the hard stop executes first.

## Pre-registered dynamic policies

1. `D0_single_1R`: one full-R primary, no second Tranche.
2. `D1_release_cap035`: second risk capped at 0.35R, requiring at least 0.20R enforceable release.
3. `D2_release_cap050`: second risk capped at 0.50R, requiring at least 0.20R release.
4. `D3_release_cap050_non_losing`: same 0.50R cap but the active Tranche must not be losing.

The second Tranche may use only risk already removed by the active Tranche's live hard stop. There is no static reservation and no third simultaneous Tranche.

## Sequential gate

Protection-only simulation runs first. A stop must retain at least 90% of P0 return in both years, keep MDD within 1.10x P0, avoid excessive hard-stop exits, and remain positive under 2x/3x costs and 1/3/5-minute delay.

Only passed protection policies enter dynamic-release testing. A dynamic candidate must:

- retain at least 95% of P0 return in both years;
- not reduce combined 2024+2025 return;
- restore at least 70% q70 coverage and about 25+ Tranches/month;
- remain positive after removing the top ten winners;
- keep live stop-defined remaining risk at or below 1R;
- avoid broken-state additions and keep losing-position additions below the frozen limit.

## Run command

```bat
python research\eth_ai_trading\03_4_2_9_dynamic_risk_release.py
```

## Report directory

```text
data\reports\research\eth_ai_trading\03_4_2_9_dynamic_risk_release
```

Priority files:

- `99_decision.md`
- `03_protection_summary.csv`
- `04_protection_trades.csv`
- `05_stop_updates.csv`
- `08_account_policy_summary.csv`
- `11_policy_gate.csv`
- `gpt_review_pack.zip`

## Non-negotiable boundaries

- Do not tune q70, `failed_reclaim`, the 3% disaster floor or structural pivot parameters.
- Do not choose different rules for 2024 and 2025.
- Do not add a third Tranche.
- Do not lower return-retention gates merely because MDD improves.
- Do not use fixed six hours as the final exit.
- Do not open 2026.
