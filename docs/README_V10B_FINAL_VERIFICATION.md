# V10B Final Verification v2

This package hardens the V10B all-engine swing structural stop candidate before any AetherEdge migration.

## What changed in v2

- `eth_lf_portfolio_v10b_all_swing_structural_stop_backtest.py`
  - Uses strict full-window `rolling(21, min_periods=21)` for structural high/low.
  - Computes structural columns on warmup-inclusive features before slicing to `start_date`.
  - Preserves V10A delayed range-exit priority: due delayed exits execute at the bar open before reading that bar's high/low/close.
  - Computes and commits structural stop updates only after all current-bar exits are confirmed false.
  - Restores V10A add-on guard: no add-on while a delayed range exit is pending.
  - Separates `STRUCTURAL_STOP` exits from `PROTECTED_TRAILING_STOP` exits.

- `research/v10b_final_verification.py`
  - Adds stricter static audit gates for full 21-bar windows, warmup-inclusive structure columns, delayed range exit priority, and structural update-after-exit order.
  - Adds explicit trade-key null/duplicate/row-conservation checks.

- `research/v10b_structural_neighbourhood.py`
  - Runs the all-swing family around the promoted candidate: default `n=13/21/34` and `buffer=0/0.1/0.25`.

## Recommended Windows commands

Run formal V10A/V10B v2 verification:

```powershell
$env:PYTHONPATH="."
python research/v10b_final_verification.py --rerun-formal --stress --end-date 2026-06-15
```

Run neighbourhood check:

```powershell
$env:PYTHONPATH="."
python research/v10b_structural_neighbourhood.py --end-date 2026-06-15
```

If you want to keep the neighbourhood trades too:

```powershell
$env:PYTHONPATH="."
python research/v10b_structural_neighbourhood.py --end-date 2026-06-15 --write-trades
```

## Recommended Linux/macOS commands

```bash
PYTHONPATH=. python research/v10b_final_verification.py --rerun-formal --stress --end-date 2026-06-15
PYTHONPATH=. python research/v10b_structural_neighbourhood.py --end-date 2026-06-15
```

## Outputs to review

Final verification output:

```text
data/reports/research/v10b_final_verification/
  01_summary_compare.csv
  02_engine_compare.csv
  03_yearly_compare.csv
  04_top_trade_dependency.csv
  05_trade_diff_summary.csv
  06_trade_diff_detail.csv
  07_structural_stop_audit.csv
  08_no_lookahead_static_audit.csv
  09_fee_slippage_stress.csv
  10_decision_matrix.csv
```

Neighbourhood output:

```text
data/reports/research/v10b_structural_neighbourhood/
  01_neighbourhood_summary.csv
```

## Promotion rule

Do not migrate V10B into AetherEdge until the v2 rerun confirms:

- no-lookahead static audit has zero `FAIL` rows;
- V10B still beats V10A on return, drawdown, PF, and stress;
- trade diff key quality passes;
- `n=21/buffer=0` is not an isolated parameter point;
- parity/dry-run work is ready for AetherEdge.
