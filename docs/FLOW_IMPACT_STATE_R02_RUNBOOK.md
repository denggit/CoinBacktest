# OKX Flow–Impact State Round 02 Runbook

## Purpose

Search the R01 pressure-event universe for broad causal conditions that remain profitable after costs without shrinking into a small-sample cell.

## Default Windows command

```bat
python research\mhf\flow_impact_state\02_conditional_edge_discovery.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --min-pressure-z 2.0
```

The script is cache-only and reads `data\okx_trade_bars.db` through `OKXTradeBarLoader`. It does not download, rebuild, or read Books.

## Fast validation

```bat
python research\mhf\flow_impact_state\02_conditional_edge_discovery.py --self-test
```

## Frozen split

```text
Discovery:  2023-01-01 -> 2024-12-31
Validation: 2025-01-01 -> 2025-09-30
Holdout:    2025-10-01 -> 2026-06-30
```

Quantile thresholds and candidate conditions are fitted only on Discovery. Validation and Holdout are opened only after a condition is frozen.

## Features

R02 evaluates relative volume/activity, average and maximum trade-size expansion, large-trade participation, directional flow consistency, pressure persistence, normalized price response, impact efficiency and close-location rejection/acceptance.

## Hard qualification

A final condition must satisfy all of the following:

```text
full events >= 1,000
discovery/validation/holdout events >= 500/200/200
positive net mean in all three splits
full net PF >= 1.20
positive months >= 65%
active dates >= 65%
40-90 events/month
at least 3 positive years
no concentration in the five largest winners
```

## Output

```text
data/reports/research/mhf/flow_impact_state/02_conditional_edge_discovery
```

Review in this order:

1. `03_time_split_definition.csv`
2. `04_base_universe_summary.csv`
3. `06_univariate_tail_scan.csv`
4. `07_feature_monotonicity.csv`
5. `08_frozen_single_candidates.csv`
6. `10_frozen_pair_candidates.csv`
7. `11_final_candidate_ranking.csv`
8. `12_qualified_candidates.csv`
9. `13_yearly_candidate_stability.csv`
10. `14_monthly_candidate_stability.csv`
11. `16_signal_audit.csv`
12. `20_research_brief.md`

## Stop rule

If `12_qualified_candidates.csv` is empty, do not add more 1m filters. Move to raw-trade/5s impact-efficiency decay. Books remain a later incremental comparison, not a prerequisite for the long-history mechanism.
