# ETH LF Portfolio V5 Validation Report

## Baseline presets

| Preset | Trades | Total Return | Max DD | PF | Win Rate | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stable | 80 | 772.90% | 15.84% | 4.20 | 25.00% | 4.88% | 107.74% | 91.96% | 108.70% |
| high | 80 | 1291.48% | 20.21% | 4.17 | 25.00% | 5.45% | 142.26% | 123.46% | 143.76% |
| turbo | 80 | 1994.06% | 24.40% | 4.14 | 25.00% | 5.65% | 176.99% | 156.39% | 179.10% |
| ultra | 80 | 3252.51% | 29.74% | 4.08 | 25.00% | 5.39% | 222.95% | 202.14% | 226.00% |

## Walk-forward / subperiod checks

Turbo preset with same rules, period capital reset, warmup retained.

| Period | Trades | Return | Max DD | PF | Win Rate |
|---|---:|---:|---:|---:|---:|
| 2023H2 | 18 | 5.65% | 15.86% | 1.14 | 22.22% |
| 2024 | 24 | 181.19% | 13.37% | 3.11 | 25.00% |
| 2025 | 23 | 156.39% | 13.35% | 2.95 | 34.78% |
| 2026H1 | 15 | 179.10% | 12.99% | 6.06 | 13.33% |
| OOS 2025-2026 | 38 | 615.61% | 24.40% | 4.60 | 26.32% |

Observation: the weakest subperiod is 2023H2. The strategy remains profitable, but PF is only 1.14, so this is the main robustness concern.

## Parameter perturbation

Tested mature-long ADX threshold values: 14, 16, 18, 20, 22.  
Tested mature-long risk multipliers: 0.4, 0.5, 0.6, 0.7.

Across 20 combinations under turbo preset:

```text
Total return range: 1816.22% - 2061.37%
Max DD range:       23.75% - 26.62%
PF range:           3.92 - 4.22
```

Observation: the result is not a single-point parameter miracle. The rule is reasonably robust in a coarse range.

## Bear standalone contribution

| Bear Scale | Trades | Return | Max DD | PF | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 70 | 1560.20% | 22.04% | 4.11 | 3.35% | 180.55% | 201.73% | 89.76% |
| 0.5 | 81 | 1050.09% | 20.23% | 2.87 | 7.53% | 187.57% | 167.41% | 39.08% |
| 0.8 | 80 | 1822.54% | 23.30% | 4.05 | 6.55% | 182.41% | 159.31% | 146.39% |
| 1.0 | 80 | 1994.06% | 24.40% | 4.14 | 5.65% | 176.99% | 156.39% | 179.10% |
| 1.2 | 80 | 2116.97% | 23.91% | 4.25 | 4.77% | 174.76% | 158.62% | 197.78% |

Observation: Bear-only improves 2026 and total return, but too-low Bear scale damages the payoff structure. 1.0 is balanced; 1.2 is interesting but more aggressive.

## Fee, slippage, latency stress

### Fee stress

| Fee Per Side | Return | Max DD | PF | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.055% | 1994.06% | 24.40% | 4.14 | 5.65% | 176.99% | 156.39% | 179.10% |
| 0.075% | 1897.25% | 24.93% | 4.06 | 3.77% | 173.68% | 153.38% | 177.55% |
| 0.100% | 1782.37% | 25.57% | 3.95 | 1.47% | 169.59% | 149.67% | 175.61% |
| 0.150% | 1571.71% | 26.85% | 3.75 | -2.99% | 161.58% | 142.39% | 171.77% |

### Slippage stress

| Slippage | Return | Max DD | PF | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.020% | 1994.06% | 24.40% | 4.14 | 5.65% | 176.99% | 156.39% | 179.10% |
| 0.050% | 1893.60% | 24.93% | 4.06 | 3.86% | 173.23% | 153.13% | 177.54% |
| 0.100% | 1093.89% | 26.47% | 3.49 | 5.83% | 87.78% | 118.43% | 175.04% |
| 0.200% | 161.26% | 46.35% | 2.00 | -30.03% | -15.75% | 109.66% | 111.36% |

### Signal delay stress

| Delay | Return | Max DD | PF | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bar | 1994.06% | 24.40% | 4.14 | 5.65% | 176.99% | 156.39% | 179.10% |
| 1 bar | 523.80% | 35.66% | 3.22 | -9.74% | 10.99% | 155.08% | 144.11% |
| 2 bars | 627.27% | 25.79% | 3.85 | 1.40% | -5.02% | 130.51% | 227.60% |

Observation: fee and mild slippage are acceptable. Extreme slippage and delayed execution damage the strategy heavily, especially in 2023/2024. This strategy needs timely execution at the next 4H open.

## Recommendation

- Main research version: `turbo`.
- More robust / less stressful version: `high`.
- High-risk research only: `ultra`.
- 2023 is the weakest period. The strategy still wins, but not strongly.
- Do not add V8/V4B fallback by default. It adds noise and worsens the structure.
