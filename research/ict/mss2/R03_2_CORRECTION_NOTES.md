# R03.2 Correction Notes

R03.2 corrects the two issues found in the real R03 report:

1. **Microstructure grain:** extract Trade Bar / Footprint for the exact union of first-`>=3` and first-`>=4` concrete trade checkpoints. Do not assume the first `>=4` trade row is the same row as first `>=3`.
2. **Execution grain:** execution overlays now start from the exact frozen **5m episode-reclaim core opportunity**, not merely from a liquidity-stack threshold stage.

The corrected execution comparison keeps the same R02 signal, structural stop, opposing 4H target, and absolute censor horizon. It compares original reclaim market, post-reclaim FVG market, post-reclaim FVG proximal limit, and 50/50 reclaim-market + FVG-limit execution on 1m/2m/5m FVG timeframes.

Every opportunity remains in the denominator even when no FVG appears. The original reclaim-market path is recomputed from naked 1m K and must tie exactly to R02 before overlays are accepted.
