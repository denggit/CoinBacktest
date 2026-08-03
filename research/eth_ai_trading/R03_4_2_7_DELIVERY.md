# R03.4.2.7 Delivery — Causal Non-Time Structural Exit Audit

## Purpose

Test whether one unified causal structure state machine can replace fixed-time holding for the frozen q70 long opening pool across both 2024 and 2025.

## New code

- `research/eth_ai_trading/03_4_2_7_non_time_structural_exit.py`
- `src/ai_research/long_tail_structural_exit/__init__.py`
- `src/ai_research/long_tail_structural_exit/config.py`
- `src/ai_research/long_tail_structural_exit/structure.py`
- `src/ai_research/long_tail_structural_exit/simulator.py`
- `src/ai_research/long_tail_structural_exit/analysis.py`
- `src/ai_research/long_tail_structural_exit/pipeline.py`
- `src/ai_research/long_tail_structural_exit/reports.py`
- `tests/ai_research/test_long_tail_structural_exit.py`

## Run

```text
python research\eth_ai_trading\03_4_2_7_non_time_structural_exit.py
```

Optional cache rebuild:

```text
python research\eth_ai_trading\03_4_2_7_non_time_structural_exit.py --force-rebuild-outcomes
```

## Report directory

```text
data\reports\research\eth_ai_trading\03_4_2_7_non_time_structural_exit
```

Key outputs:

- `04_policy_summary.csv`
- `05_quarter_summary.csv`
- `06_score_tier_summary.csv`
- `07_exit_reason_summary.csv`
- `08_censoring_audit.csv`
- `10_vs_fixed6h_comparison.csv`
- `11_stable_candidates.csv`
- `14_trade_details.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Research contract

- q70 is retained as the main pool.
- q70-q80/q80-q90/q90+ remain visible.
- Candidate policies contain no scheduled or maximum holding-time exit.
- The exact same policies and parameters run in 2024 and 2025.
- Opening-score persistence/upgrade is not a holding or adding signal.
- OOS-end/data-gap positions are explicitly censored.
- Fixed 6h is a benchmark only.
- 2026 remains sealed.

## Included cumulative handoff files

- `RESEARCH_HANDOFF.md`
- `COMPLETED_WORK.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `R03_4_2_7_DELIVERY.md`

These files must be read first by the next conversation/window.

## Validation executed before delivery

- New structural-exit tests: 9 passed.
- All `tests/ai_research` plus `tests/data_feed`: 148 passed.
- Python compilation: passed.
- CLI `--help`: passed.
- Patch-overlay verification: performed on a clean R03.4.2.6 repository copy.
