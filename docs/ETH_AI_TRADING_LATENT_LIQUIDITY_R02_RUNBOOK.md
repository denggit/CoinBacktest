# ETH Latent Liquidity R02 Runbook

Run from CoinBacktest repository root:

```text
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_latent_pool_location_depth_model.py
```

R02 uses existing local data only.  It does not rebuild or download 1s/1m Trade Bars by default.

Expected long stages:

```text
[latent-liquidity-r02] stream R01.1 Episode labels ...
[latent-liquidity-r02] spatial chunks ...
[stage] fit distance baseline vs liquidity-path-no-Swing vs full-with-15m+-Swing
[stage] write compact R02 report
```

Every 14-day spatial chunk is checkpointed.  If the process is interrupted, rerunning the same command reuses completed chunks.

Do not delete `data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/r02_latent_pool_location_depth_model` until the full report has been reviewed.

## Causal-label boundary

The primary label horizon is 12 hours. The runner therefore stops creating decision rows 12 hours before the requested research end, while still loading those final hours as future-label support. Do not reinterpret the omitted tail as missing data.

## Swing supplement

R02 requires the R01.1 15m+ lifecycle cache so the full model can be compared with the no-Swing path model. All causally active unswept 15m/30m/1H/4H/1D levels are retained regardless of age until first sweep. Swing is never an admission gate.

## Full-lattice audit sample

Training controls are row-sampled with inverse-probability weights for memory safety. Separately, 5% of decision-time x direction groups retain all 25 price cells. Calibration q90, score deciles and Top-zone quality use only those complete lattices, so location selection is not biased by missing candidate cells.
