# R03.4.2.5 Patch Manifest

## Goal

Test whether a score-tier-aware, extremely selective persistent-failure overlay can cut only the most certain losing q70 trades without sacrificing the q70 cross-year positive expectancy established in R03.4.2.4.

## Key changes

- Preserves all three opening-score tiers: q70-q80, q80-q90 and q90+.
- Keeps the frozen R03.4.1 opening model and prior-quarter q70 calibration unchanged.
- Trains the holding-risk classifier on rolling, strictly causal OOF q50 events across the full pretest history.
- Uses T+60 only to arm a warning; it can never exit a position.
- Allows a T+180 exit only after prior warning, extreme persistent-failure probability and multiple independent path-structure failures.
- Requires stricter failure evidence for higher opening-score tiers.
- Compares global, tiered and ultra-conservative overlays against the identical fixed-six-hour q70 diagnostic pool.
- Separately audits a pre-registered 3% disaster safety floor using next-minute-open execution after the breach.
- Keeps score-tier policy metrics in every cost and delay stress report.
- Records whether a position's score upgrades to a higher tier by T+180 or T+360 for later pyramiding research, without executing hindsight add-ons.
- Requires both WF_2024 and WF_2025 to retain positive expectancy, PF, Top-10 robustness and quarterly stability.
- Treats six hours only as a benchmark; it does not claim a final time-based live exit.
- Keeps 2026 sealed and does not load the abandoned market-state model.
