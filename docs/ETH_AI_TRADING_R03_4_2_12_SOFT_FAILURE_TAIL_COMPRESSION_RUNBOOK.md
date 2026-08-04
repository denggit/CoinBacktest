# ETH AI Trading R03.4.2.12 Runbook

## Purpose

R03.4.2.12 separates the apparent F1 improvement into:

1. account return caused by sizing from 1.5% while retaining a 3% / 2R tail;
2. genuine incremental value from completed-close soft-failure exits;
3. performance of real one-R executable hard stops.

The stage does not change q70, `failed_reclaim`, the 2024/2025 folds, or the sealed 2026 holdout.

## Run

Windows one-line command:

```text
python research\eth_ai_trading\03_4_2_12_soft_failure_tail_compression.py
```

Optional explicit Trade Bar directory:

```text
python research\eth_ai_trading\03_4_2_12_soft_failure_tail_compression.py --data-dir data\okx_trade_bars
```

## Policies

- `P0_single_1R`: frozen 3% real hard-tail baseline.
- `F1_reference_1p5size_3ptail`: R03.4.2.11 reference; 1.5% sizing and soft failure, but 3% / 2R disaster tail. It cannot pass the one-R gate.
- `C2_real_2p_soft1p5`: 2% executable hard stop, 1.5% completed-close soft failure, one-R sizing.
- `C15_real_1p5_hard`: 1.5% executable hard stop without soft confirmation.
- `C15_real_1p5_soft1p0`: 1.5% executable hard stop plus 1.0% completed-close soft failure.
- `V1_causal_volatility_1R`: entry-frozen `2 × prior 60m ATR%`, clamped to 1.5%-3%; soft threshold is 75% of the frozen hard distance.

## Output

```text
data\reports\research\eth_ai_trading\03_4_2_12_soft_failure_tail_compression
```

Read first:

```text
99_decision.md
06_f1_attribution_summary.csv
11_policy_summary.csv
12_policy_gate.csv
07_account_cycles.csv
08_account_legs.csv
gpt_review_pack.zip
```

## Interpretation

F1 return must be normalized by its two-R hard tail before judging its exit logic. A real-tail candidate passes only if it:

- keeps executable price tail near one account-R;
- retains at least 95% of P0 return in each year;
- exceeds P0 combined return by at least 10%;
- stays profitable in every 1/3/5-minute and 2x/3x-cost cell;
- keeps absolute MDD at or below 12% and no more than 1.4x P0;
- remains positive after removing the top ten winners;
- turns no more than 5% of P0 winning cycles into losses;
- raises average initial nominal exposure to at least 0.45x equity;
- does not open 2026 or use a fixed-time final exit.
