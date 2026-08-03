#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vector-light event replay for fixed-risk staged execution."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd
from .config import PostSweepStagedExecutionConfig, SchemeSpec


def build_fill_table(triggers: pd.DataFrame, schemes: tuple[SchemeSpec, ...]) -> pd.DataFrame:
    lookup = triggers.set_index(["zone_event_id", "trigger_name"], drop=False)
    events = triggers.loc[triggers["trigger_name"] == "INITIAL", "zone_event_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for event_id in events:
        previous_activation = -1
        for scheme in schemes:
            previous_activation = -1
            for sequence, stage in enumerate(scheme.stages, start=1):
                key = (event_id, stage.trigger)
                if key not in lookup.index:
                    continue
                source = lookup.loc[key]
                if isinstance(source, pd.DataFrame):
                    source = source.iloc[0]
                signal_elapsed = int(source["elapsed_bars"])
                if signal_elapsed > stage.max_signal_elapsed:
                    continue
                activation_elapsed = signal_elapsed + 1
                if activation_elapsed < previous_activation:
                    activation_elapsed = previous_activation
                previous_activation = activation_elapsed
                rows.append({
                    "zone_event_id": event_id,
                    "period": source["period"],
                    "scheme": scheme.name,
                    "stage_sequence": sequence,
                    "stage_name": stage.name,
                    "trigger_name": stage.trigger,
                    "weight": float(stage.weight),
                    "signal_checkpoint_id": source["checkpoint_id"],
                    "signal_elapsed": signal_elapsed,
                    "entry_activation_elapsed": activation_elapsed,
                    "signal_time": source["checkpoint_available_time"],
                    "entry_time": source["entry_reference_time"],
                    "entry_price": float(source["entry_reference_price"]),
                    "future_mfe_15m": source.get("future_mfe_15m", np.nan),
                    "future_mae_15m": source.get("future_mae_15m", np.nan),
                    "future_mfe_60m": source.get("future_mfe_60m", np.nan),
                    "future_mae_60m": source.get("future_mae_60m", np.nan),
                    "future_mfe_180m": source.get("future_mfe_180m", np.nan),
                    "future_mae_180m": source.get("future_mae_180m", np.nan),
                })
    return pd.DataFrame(rows)


def _prepare_fill_map(fills: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    if fills.empty:
        return out
    for (event_id, scheme), group in fills.groupby(["zone_event_id", "scheme"], sort=False):
        g = group.sort_values(["entry_activation_elapsed", "stage_sequence"], kind="mergesort")
        out[(str(event_id), str(scheme))] = {
            "elapsed": pd.to_numeric(g["entry_activation_elapsed"], errors="coerce").to_numpy(dtype=float),
            "weights": pd.to_numeric(g["weight"], errors="coerce").to_numpy(dtype=float),
            "prices": pd.to_numeric(g["entry_price"], errors="coerce").to_numpy(dtype=float),
            "mfe180": pd.to_numeric(g["future_mfe_180m"], errors="coerce").to_numpy(dtype=float),
            "mae180": pd.to_numeric(g["future_mae_180m"], errors="coerce").to_numpy(dtype=float),
        }
    return out


def _scheme_event_replay_arrays(
    event_id: str,
    period: object,
    elapsed: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fill_data: dict[str, np.ndarray] | None,
    scheme: SchemeSpec,
    cfg: PostSweepStagedExecutionConfig,
) -> dict[str, object]:
    if fill_data is None:
        fill_elapsed = np.array([], dtype=float)
        weights = np.array([], dtype=float)
        prices = np.array([], dtype=float)
        mfe180 = np.array([], dtype=float)
        mae180 = np.array([], dtype=float)
    else:
        fill_elapsed = fill_data["elapsed"]
        weights = fill_data["weights"]
        prices = fill_data["prices"]
        mfe180 = fill_data["mfe180"]
        mae180 = fill_data["mae180"]
    result: dict[str, object] = {"zone_event_id": event_id, "period": period, "scheme": scheme.name, "filled_stages": int(weights.size)}
    result["expected_stages"] = len(scheme.stages)
    result["full_deployment_ever"] = bool(abs(weights.sum() - 1.0) < 1e-9)
    result["final_deployed_weight"] = float(weights.sum())
    if weights.size:
        result["weighted_average_entry_price"] = float(weights.sum() / np.sum(weights / prices))
        result["last_entry_elapsed"] = float(fill_elapsed.max())
        result["weighted_tranche_mfe_180m"] = float(np.nansum(weights * mfe180))
        result["weighted_tranche_mae_180m"] = float(np.nansum(weights * mae180))
    else:
        result["weighted_average_entry_price"] = np.nan
        result["last_entry_elapsed"] = np.nan
        result["weighted_tranche_mfe_180m"] = 0.0
        result["weighted_tranche_mae_180m"] = 0.0
    entry_cost_per_weight = cfg.fee_rate_per_side + cfg.slippage_rate_per_side
    stress_entry_cost_per_weight = cfg.fee_rate_per_side + cfg.stressed_slippage_rate_per_side
    for horizon in cfg.horizons:
        n = int(np.searchsorted(elapsed, horizon, side="right"))
        if n <= 0:
            for name in ("deployed_weight", "gross_close_return", "net_close_return", "stress_net_close_return", "sparse_mfe", "sparse_mae"):
                result[f"{name}_{horizon}m"] = np.nan
            continue
        h_elapsed = elapsed[:n]
        h_high = high[:n]
        h_low = low[:n]
        h_close = close[:n]
        # Number of fills active at each checkpoint. Fills are tiny (<=3), so
        # broadcasting is faster than repeated pandas joins and remains causal.
        if weights.size:
            active = h_elapsed[:, None] >= fill_elapsed[None, :]
            A = (active * (weights / prices)[None, :]).sum(axis=1)
            W = (active * weights[None, :]).sum(axis=1)
        else:
            A = np.zeros(n, dtype=float)
            W = np.zeros(n, dtype=float)
        C = W * entry_cost_per_weight
        CS = W * stress_entry_cost_per_weight
        pnl_high = h_high * A - W - C
        pnl_low = h_low * A - W - C
        pnl_close = h_close * A - W - C
        deployed = float(W[-1])
        exit_cost = deployed * (cfg.fee_rate_per_side + cfg.slippage_rate_per_side)
        stress_exit_cost = deployed * (cfg.fee_rate_per_side + cfg.stressed_slippage_rate_per_side)
        result[f"deployed_weight_{horizon}m"] = deployed
        result[f"gross_close_return_{horizon}m"] = float(h_close[-1] * A[-1] - W[-1])
        result[f"net_close_return_{horizon}m"] = float(pnl_close[-1] - exit_cost)
        result[f"stress_net_close_return_{horizon}m"] = float(h_close[-1] * A[-1] - W[-1] - CS[-1] - stress_exit_cost)
        result[f"sparse_mfe_{horizon}m"] = float(np.nanmax(pnl_high)) if pnl_high.size else np.nan
        result[f"sparse_mae_{horizon}m"] = float(np.nanmin(pnl_low)) if pnl_low.size else np.nan
        result[f"net_return_per_deployed_{horizon}m"] = float((pnl_close[-1] - exit_cost) / deployed) if deployed > 0 else np.nan
    return result


def simulate_schemes(path: pd.DataFrame, fills: pd.DataFrame, schemes: tuple[SchemeSpec, ...], cfg: PostSweepStagedExecutionConfig, progress: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fill_map = _prepare_fill_map(fills)
    grouped = path.groupby("zone_event_id", sort=False)
    total = int(path["zone_event_id"].nunique())
    for i, (event_id_raw, group) in enumerate(grouped, start=1):
        event_id = str(event_id_raw)
        period = group.iloc[0]["period"]
        elapsed = pd.to_numeric(group["elapsed_bars"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(group["checkpoint_high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(group["checkpoint_low"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(group["checkpoint_close"], errors="coerce").to_numpy(dtype=float)
        for scheme in schemes:
            rows.append(_scheme_event_replay_arrays(event_id, period, elapsed, high, low, close, fill_map.get((event_id, scheme.name)), scheme, cfg))
        if progress and (i == total or i % 1000 == 0):
            print(f"[replay] events={i:,}/{total:,}", flush=True)
    return pd.DataFrame(rows)
