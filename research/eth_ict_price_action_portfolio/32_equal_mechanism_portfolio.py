#!/usr/bin/env python
"""Frozen 50/50 mechanism diversification: daily PA core plus EMA trend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _multispeed_ema_bridge as ema
from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v19" / "results"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    ema_features = ema.build_daily_ema_features(minute)
    ema1 = ema.positions_from_features(ema_features, minute.index, 1) * 0.5
    ema2 = ema.positions_from_features(ema_features, minute.index, 2) * 0.5
    # core_state already includes a causal +1m execution shift and a 0.40 cap.
    core1 = pd.DataFrame({"daily_pa_core": base.core_state(minute) * 0.5}, index=minute.index)
    core2 = core1.shift(1).fillna(0.0)
    pos1 = pd.concat([core1, ema1], axis=1)
    pos2 = pd.concat([core2, ema2], axis=1)
    variants = {
        "equal_pa_core_ema_1m": (pos1, base.ONE_WAY_COST),
        "equal_pa_core_ema_2m": (pos2, base.ONE_WAY_COST),
        "equal_pa_core_ema_1m_double_cost": (pos1, 2.0 * base.ONE_WAY_COST),
    }
    replays = {name: base.simulate_minute(minute, pos, cost=cost) for name, (pos, cost) in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_equal_mechanism_screen.csv", index=False)
    yearly = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = int(year)
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "02_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP perpetual K-lines only",
                "portfolio": "50% causal daily PA core + 50% fixed multi-speed EMA mechanism",
                "weight_search": "none; exact equal mechanism allocation",
                "execution": "1m baseline; 2m fixed latency stress",
                "one_way_cost": base.ONE_WAY_COST,
                "double_cost_stress": 2.0 * base.ONE_WAY_COST,
            }, indent=2
        ), encoding="utf-8"
    )
    print(screen.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
