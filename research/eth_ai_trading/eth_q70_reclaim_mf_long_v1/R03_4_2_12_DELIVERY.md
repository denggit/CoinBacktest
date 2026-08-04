# R03.4.2.12 Delivery

## Status

```text
R03.4.2.11 empirical result:
FAIL_NO_ROBUST_STAGED_EXECUTION

R03.4.2.12:
code complete
causal audit complete
unit/regression validation complete
local full-data run pending
```

## Research question

Can the F1 completed-close soft-failure idea increase initial nominal exposure while restoring the actual worst price loss to one account-R?

## Important distinction

F1 in R03.4.2.11 used approximately 0.67x initial notional because it sized from 1.5%, but its exchange-side disaster tail remained 3%. Its maximum hard tail was therefore 2R. R03.4.2.12 treats F1 as an attribution reference only.

A qualifying R03.4.2.12 policy must size from its real executable hard-stop distance:

```text
units = one-R account dollars / real hard-stop price distance
```

## Real-tail candidates

| Policy | Real hard stop | Soft failure | Expected initial notional | Tail |
|---|---:|---:|---:|---:|
| C2 | 2.0% | completed close below -1.5% | ~0.50x | 1R |
| C15 hard | 1.5% | none | ~0.67x | 1R |
| C15 soft | 1.5% | completed close below -1.0% | ~0.67x | 1R |
| V1 | causal ATR, 1.5%-3.0% | 75% of frozen hard distance | ~0.33x-0.67x | 1R |

## Required reports

```text
00_run_manifest.json
01_preflight.json
02_source_p0_baseline.csv
03_source_f1_reference.csv
04_selected_p0_cycles.csv
05_f1_exit_attribution.csv
06_f1_attribution_summary.csv
07_account_cycles.csv
08_account_legs.csv
09_account_actions.csv
10_daily_equity.csv
11_policy_summary.csv
12_policy_gate.csv
13_causal_audit.csv
14_runtime_rejections.csv
15_failures.csv
99_decision.md
gpt_review_pack.zip
```

## No empirical claim in this patch

The delivery environment does not contain the user's full Trade Bar database. The patch is `READY / DATA RUN PENDING`; it does not claim a trading pass or failure.

## Validation

- R03.4.2.12专项 tests: 9 passed.
- R03.4.2.7 through R03.4.2.12 related regression: 59 passed.
- `tests/ai_research` plus `tests/data_feed`: 198 passed.
- Source-report chain: loaded 1,438 selected P0 events, 110,331 structure-timeline rows, and the frozen P0/F1 account outputs.
- Entrypoint smoke: reached `BLOCKED_DATA` cleanly because this environment has no local Trade Bar rows.
- Full repository collection remains blocked by five pre-existing missing liquidity/analyze-tool modules.
- Import-boundary audit still reports 155 pre-existing violations outside this stage; R03.4.2.12 adds zero.
- No git commit was executed.
