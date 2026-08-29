# RDPOS-03 Path-Aware Positioning Counterfactual

Goal: test whether mature, persistent trend expansion should receive more exposure than the frozen RDPOS-01 location penalty allows.

This is **not** a new entry/exit strategy and does not tune trend horizons.

Central causal path state:

- medium/slow same direction and both |trend| >= 0.25;
- state has persisted at least 12h;
- mean aligned extension >= 0.50;
- strong-agreement share over the past 72h >= 0.60.

The script also evaluates a pre-specified neighborhood around these values. No best row is selected as a new strategy.

Run on Windows:

`python research\eth_dynamic_positioning\03_path_aware_positioning_counterfactual.py`

Input:

`data\reports\research\eth_dynamic_positioning\01_trend_location_vol_positioning\`

Output:

`data\reports\research\eth_dynamic_positioning\03_path_aware_positioning_counterfactual\`

Interpretation warning: the path hypothesis was discovered using the same 2023-2026 history, so this counterfactual is not an independent holdout. It can justify continued research, not live deployment.
