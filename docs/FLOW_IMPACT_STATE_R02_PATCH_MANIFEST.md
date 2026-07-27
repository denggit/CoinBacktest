# Flow–Impact State R02 Patch Manifest

## Purpose

Add a strict conditional-edge discovery round after R01 without mining small-sample cells or repeatedly opening the final holdout.

## Files

```text
research/mhf/flow_impact_state/00_research_log.md
research/mhf/flow_impact_state/01_pressure_event_atlas.py
research/mhf/flow_impact_state/02_conditional_edge_discovery.py
src/research_common/flow_impact.py
src/research_common/flow_impact_io.py
src/research_common/flow_impact_outcomes.py
src/research_common/conditional_edge.py
tests/research_common/test_conditional_edge.py
docs/FLOW_IMPACT_STATE_R02_RUNBOOK.md
docs/FLOW_IMPACT_STATE_R02_WEB_RESEARCH.md
docs/FLOW_IMPACT_STATE_R02_PATCH_MANIFEST.md
```

## Main changes

- Freeze R02 event threshold at `pressure_z >= 2.0` from R01 frequency calibration.
- Add relative volume, average/max trade size, large-trade share and flow-concentration features.
- Extract reusable next-open forward-path outcomes from the R01 entrypoint.
- Fit all thresholds on 2023–2024 discovery only.
- Open 2025 validation and 2025-10 onward holdout only for discovery-frozen specifications.
- Correct discovery multiple testing with Benjamini–Hochberg q-values.
- Limit interaction search to two features and only among frozen single-feature conditions.
- Enforce >=1,000 total events and 40–90 events/month for final qualification.
- Add a hard stop rule: no more 1m environment-filter versions if R02 has no qualified condition.
- Replace deprecated `pd.Timestamp.utcnow()` with `pd.Timestamp.now("UTC")`.

## Run

```bat
python research\mhf\flow_impact_state\02_conditional_edge_discovery.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --min-pressure-z 2.0
```

## Validation performed

```text
R01 self-test: PASS
R02 self-test: PASS
tests/research_common: 10 passed
compileall: PASS
new-code research->research imports: 0
new-code direct SQLite reads: 0
Timestamp.utcnow occurrences in Flow-Impact files: 0
```

Full repository pytest still cannot collect because the supplied repository baseline is missing:

```text
research/liquidity/panic_selloff_rejection_recovery_long
research/liquidity/liquidity_touch_rebound_v1
```

These pre-existing missing modules produce nine collection errors and are outside this patch.

No git commit was executed.
