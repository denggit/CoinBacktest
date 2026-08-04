# R03.4.2.11 Patch Manifest

## Goal

Audit whether split entry, confirmed soft-failure sizing, Turtle-style adds and independent-stop pyramiding can improve the frozen q70 P0 account result without reducing the base winner.

## New code

- `src/ai_research/long_tail_staged_execution/*`
- `research/eth_ai_trading/03_4_2_11_staged_entry_pyramiding.py`
- `tests/ai_research/test_long_tail_staged_execution.py`

## New documentation

- `docs/ETH_AI_TRADING_R03_4_2_11_STAGED_ENTRY_PYRAMIDING_RUNBOOK.md`
- `research/eth_ai_trading/R03_4_2_11_DELIVERY.md`
- this manifest

## Updated cumulative handoff

- `README.md`
- `RESEARCH_HANDOFF.md`
- `COMPLETED_WORK.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `STAGE_DELIVERY.md`

## Explicitly unchanged

- q70 model and thresholds;
- public data loaders;
- 3% base disaster floor;
- `failed_reclaim` base exit;
- 2026 seal;
- prior empirical reports.

## Suggested commit message

```text
research: add R03.4.2.11 staged entry and asymmetric pyramiding audit
```

No automatic git commit is allowed.
