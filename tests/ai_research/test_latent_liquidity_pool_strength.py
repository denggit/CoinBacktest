from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_pool_strength.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_pool_strength.labels import attach_strength_labels, attach_train_frozen_strength_thresholds
from src.ai_research.latent_liquidity_pool_strength.modeling import feature_columns, fit_models, metric_table, predict
from src.ai_research.latent_liquidity_pool_strength.reports import causal_audit, top1_zone_summary


def _zones() -> pd.DataFrame:
    t = pd.Timestamp("2025-01-01 12:00:00")
    rows = []
    for side in ("DOWN", "UP"):
        for d in (10.0, 30.0):
            rows.append({
                "zone_id": f"{side}-{d}", "decision_time": t, "feature_available_time": t,
                "period": "VALIDATION_2025Q1_Q3", "zone_side": side, "current_price": 100.0,
                "zone_distance_bp": d, "zone_price": 100 * (1-d/1e4) if side == "DOWN" else 100 * (1+d/1e4),
                "side_is_down": int(side == "DOWN"), "touch_720m": True,
                "primary_touch_label_complete": True, "model_sample_keep": True,
                "full_lattice_audit_group": True, "sample_weight": 1.0,
                "macro_pressure_without_progress_60m": 0.5, "micro_path_efficiency_60s": 0.4,
                "swing_count_25bp": 2,
            })
    return pd.DataFrame(rows)


def test_strength_labels_aggregate_multiple_episodes_in_same_zone() -> None:
    zones = _zones()
    t = zones["decision_time"].iloc[0]
    episodes = pd.DataFrame({
        "event_time": [t + pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=40)],
        "event_side": ["DOWN", "DOWN"], "event_reference_price": [99.90, 99.905],
        "release_density_proxy": [2.0, 3.0], "release_episode_size": [4, 5], "release_score": [3.0, 4.0],
        "favorable_reversal": [True, False], "outcome_type": ["EXTEND_STABILIZE_REVERSAL", "ACCEPT_CONTINUATION"],
        "future_extension_bp": [20.0, 30.0], "future_reversal_after_extreme_bp": [50.0, 10.0],
    })
    out = attach_strength_labels(zones, episodes, horizon_minutes=720, zone_offsets_bp=(10.0, 30.0), zone_half_width_bp=10.0)
    row = out.loc[(out["zone_side"].eq("DOWN")) & out["zone_distance_bp"].eq(10.0)].iloc[0]
    assert row["release_episode_count"] == 2
    assert np.isclose(row["release_density_sum"], 5.0)
    assert row["favorable_episode_count"] == 1
    assert row["continuation_episode_count"] == 1


def test_strength_labels_exclude_exact_horizon_right_endpoint() -> None:
    zones = _zones()
    t = zones["decision_time"].iloc[0]
    episodes = pd.DataFrame({
        "event_time": [t + pd.Timedelta(minutes=720)], "event_side": ["DOWN"],
        "event_reference_price": [99.90], "release_density_proxy": [4.0], "release_episode_size": [5],
        "release_score": [3.0], "favorable_reversal": [True], "outcome_type": ["EXTEND_STABILIZE_REVERSAL"],
        "future_extension_bp": [20.0], "future_reversal_after_extreme_bp": [50.0],
    })
    out = attach_strength_labels(zones, episodes, horizon_minutes=720, zone_offsets_bp=(10.0, 30.0), zone_half_width_bp=10.0)
    row = out.loc[(out["zone_side"].eq("DOWN")) & out["zone_distance_bp"].eq(10.0)].iloc[0]
    assert row["release_episode_count"] == 0


def test_train_strength_threshold_is_frozen_by_side() -> None:
    frame = pd.DataFrame({
        "period": ["TRAIN_2023_2024"] * 10 + ["HOLDOUT_2025Q4_2026H1"] * 2,
        "zone_side": ["DOWN"] * 12, "touch_720m": [True] * 12,
        "release_density_log": list(np.arange(10, dtype=float)) + [100.0, 101.0],
    })
    out, thresholds = attach_train_frozen_strength_thresholds(frame, train_period="TRAIN_2023_2024", quantile=0.8)
    threshold = thresholds.loc[thresholds["zone_side"].eq("DOWN"), "release_density_log_threshold"].iloc[0]
    assert threshold < 10
    assert out.iloc[-1]["high_strength_label"]


def _model_frame(n=1200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    parts = []
    for period in DEFAULT_CONFIG.periods:
        path = rng.normal(size=n)
        dist = rng.choice([10., 30., 50., 100., 200., 300.], size=n)
        side = rng.choice(["DOWN", "UP"], size=n)
        density = np.maximum(0, 2.0 + 1.4 * path + rng.normal(0, 1, size=n))
        high = density > 2.7
        release = density > 0.5
        fav = release & (rng.random(n) < 1/(1+np.exp(-path)))
        cont = release & ~fav & (rng.random(n) < 0.5)
        decision = pd.date_range("2023-01-01", periods=n, freq="15min")
        parts.append(pd.DataFrame({
            "zone_id": [f"{period}-{i}" for i in range(n)], "decision_time": decision,
            "feature_available_time": decision, "period": period, "zone_side": side,
            "zone_price": 100.0, "current_price": 100.0, "zone_distance_bp": dist,
            "side_is_down": (side == "DOWN").astype(np.int8), "touch_720m": True,
            "primary_touch_label_complete": True, "model_sample_keep": True,
            "full_lattice_audit_group": True, "sample_weight": 1.0,
            "macro_pressure_without_progress_60m": path, "micro_path_efficiency_60s": rng.uniform(size=n),
            "swing_count_25bp": rng.integers(0, 4, size=n),
            "release_episode_count": release.astype(float), "release_density_sum": density,
            "release_density_max": density, "release_episode_size_sum": release.astype(float)*3,
            "release_score_sum": density, "favorable_episode_count": fav.astype(float),
            "continuation_episode_count": cont.astype(float), "favorable_density_sum": density*fav,
            "continuation_density_sum": density*cont, "sweep_depth_weighted_bp": np.where(release, 25 - 3*path + rng.normal(0,2,n), 0),
            "reversal_room_weighted_bp": np.where(release, 30 + 4*path + rng.normal(0,2,n), 0),
            "first_release_minutes": np.where(release, 60.0, np.nan), "release_within_horizon": release,
            "release_density_log": np.log1p(density), "release_count_log": np.log1p(release.astype(float)),
            "release_size_log": np.log1p(release.astype(float)*3), "release_peak_log": np.log1p(density),
            "high_strength_label": high,
        }))
    return pd.concat(parts, ignore_index=True)


def test_primary_feature_schema_can_exclude_swing_and_future_labels() -> None:
    frame = _model_frame(100)
    path = feature_columns(frame, include_swing=False)
    assert "macro_pressure_without_progress_60m" in path
    assert "swing_count_25bp" not in path
    assert "high_strength_label" not in path
    assert "release_density_sum" not in path


def test_fit_predict_primary_pool_strength_excludes_touch_and_swing() -> None:
    config = replace(DEFAULT_CONFIG, model_n_estimators=30, minimum_train_rows=100, minimum_class_rows=10, model_train_cap_rows=10000)
    frame = _model_frame(700)
    models = fit_models(frame, config)
    pred = predict(frame, models)
    assert np.allclose(pred["pool_strength_score"], pred["p_strength_path"])
    assert pred["pool_strength_score"].between(0,1).all()
    metrics = metric_table(pred, config)
    assert set(metrics["task"]) >= {"HIGH_STRENGTH_PATH_NO_SWING", "HIGH_STRENGTH_DISTANCE_BASELINE", "DENSITY_PATH_NO_SWING", "FAVORABLE_PATH_NO_SWING"}


def test_top1_strength_summary_keeps_touch_separate() -> None:
    frame = _model_frame(100)
    frame["pool_strength_score"] = np.linspace(0,1,len(frame))
    result = top1_zone_summary(frame, DEFAULT_CONFIG)
    assert {"top1_touch_rate", "top1_high_strength_rate_if_touched", "high_strength_lift"} <= set(result.columns)


def test_causal_audit_requires_primary_score_to_exclude_swing_and_touch() -> None:
    frame = _model_frame(100)
    # make 25-zone complete lattices for the audit check
    rows=[]
    for i in range(4):
        t=pd.Timestamp("2025-01-01")+pd.Timedelta(minutes=15*i)
        for side in ("DOWN","UP"):
            for j in range(25):
                r=frame.iloc[0].copy(); r["decision_time"]=t; r["feature_available_time"]=t; r["zone_side"]=side; r["zone_id"]=f"{i}-{side}-{j}"; rows.append(r)
    audit_frame=pd.DataFrame(rows)
    source=pd.DataFrame([{"check":"source","status":"PASS"}])
    features=("zone_distance_bp","macro_pressure_without_progress_60m")
    audit=causal_audit(audit_frame,audit_frame,features,source,DEFAULT_CONFIG)
    assert not audit["status"].eq("FAIL").any()


def _reference_strength_labels(zones: pd.DataFrame, episodes: pd.DataFrame, *, horizon_minutes: int, offsets: tuple[float, ...], half: float) -> pd.DataFrame:
    """Tiny pandas reference mirroring the pre-hotfix semantics for equivalence tests."""
    out = zones.copy()
    numeric = (
        "release_episode_count", "release_density_sum", "release_density_max",
        "release_episode_size_sum", "release_score_sum", "favorable_episode_count",
        "continuation_episode_count", "favorable_density_sum", "continuation_density_sum",
        "sweep_depth_weighted_bp", "reversal_room_weighted_bp",
    )
    for name in numeric:
        out[name] = 0.0
    out["first_release_minutes"] = np.nan
    out["release_within_horizon"] = False
    eps = episodes.copy()
    eps["event_time"] = pd.to_datetime(eps["event_time"], errors="coerce")
    eps = eps.loc[eps["event_time"].notna() & pd.to_numeric(eps["event_reference_price"], errors="coerce").gt(0)].copy()
    eps_by_side = {s:g.sort_values("event_time", kind="mergesort").reset_index(drop=True) for s,g in eps.groupby("event_side", sort=False)}
    times_by_side = {s:g["event_time"].to_numpy(dtype="datetime64[ns]") for s,g in eps_by_side.items()}
    off = np.asarray(offsets, dtype=float)
    horizon = pd.Timedelta(minutes=horizon_minutes)
    for decision_time, idxs in out.groupby("decision_time", sort=False).groups.items():
        idxs = list(idxs)
        t = pd.Timestamp(decision_time)
        current = float(out.at[idxs[0], "current_price"])
        for side in ("DOWN", "UP"):
            group = eps_by_side.get(side)
            if group is None or group.empty:
                continue
            times = times_by_side[side]
            left = int(np.searchsorted(times, np.datetime64(t, "ns"), side="right"))
            right = int(np.searchsorted(times, np.datetime64(t + horizon, "ns"), side="left"))
            if right <= left:
                continue
            future = group.iloc[left:right].copy()
            ref = pd.to_numeric(future["event_reference_price"], errors="coerce").to_numpy(dtype=float)
            distance = np.where(side == "DOWN", (current-ref)/current*1e4, (ref-current)/current*1e4)
            valid = np.isfinite(distance) & (distance > 0) & (distance <= off[-1] + half)
            future = future.iloc[np.flatnonzero(valid)].reset_index(drop=True)
            distance = distance[valid]
            if not len(distance):
                continue
            nearest = np.abs(distance[:,None] - off[None,:]).argmin(axis=1)
            valid_zone = np.abs(distance - off[nearest]) <= half
            future = future.iloc[np.flatnonzero(valid_zone)].reset_index(drop=True)
            nearest = nearest[valid_zone]
            row_by_zone = {int(np.argmin(np.abs(off-float(out.at[idx,"zone_distance_bp"])))):idx for idx in idxs if str(out.at[idx,"zone_side"]) == side}
            for zone_pos in np.unique(nearest):
                row = row_by_zone.get(int(zone_pos))
                if row is None:
                    continue
                c = future.iloc[np.flatnonzero(nearest == zone_pos)].copy()
                density = np.where(pd.to_numeric(c["release_density_proxy"],errors="coerce").to_numpy(float) > 0, pd.to_numeric(c["release_density_proxy"],errors="coerce").to_numpy(float), 0.0)
                size = np.where(pd.to_numeric(c["release_episode_size"],errors="coerce").to_numpy(float) > 0, pd.to_numeric(c["release_episode_size"],errors="coerce").to_numpy(float), 0.0)
                score = np.where(pd.to_numeric(c["release_score"],errors="coerce").to_numpy(float) > 0, pd.to_numeric(c["release_score"],errors="coerce").to_numpy(float), 0.0)
                fav = c["favorable_reversal"].astype(bool).to_numpy()
                cont = c["outcome_type"].astype(str).eq("ACCEPT_CONTINUATION").to_numpy()
                sweep = pd.to_numeric(c["future_extension_bp"],errors="coerce").to_numpy(float)
                room = pd.to_numeric(c["future_reversal_after_extreme_bp"],errors="coerce").to_numpy(float)
                w = np.where(density > 0, density, 1.0)
                out.at[row,"release_within_horizon"] = True
                out.at[row,"release_episode_count"] = float(len(c))
                out.at[row,"release_density_sum"] = float(density.sum())
                out.at[row,"release_density_max"] = float(density.max(initial=0.0))
                out.at[row,"release_episode_size_sum"] = float(size.sum())
                out.at[row,"release_score_sum"] = float(score.sum())
                out.at[row,"favorable_episode_count"] = float(fav.sum())
                out.at[row,"continuation_episode_count"] = float(cont.sum())
                out.at[row,"favorable_density_sum"] = float(density[fav].sum())
                out.at[row,"continuation_density_sum"] = float(density[cont].sum())
                vs=np.isfinite(sweep); vr=np.isfinite(room)
                if vs.any(): out.at[row,"sweep_depth_weighted_bp"] = float(np.average(sweep[vs], weights=w[vs]))
                if vr.any(): out.at[row,"reversal_room_weighted_bp"] = float(np.average(room[vr], weights=w[vr]))
                out.at[row,"first_release_minutes"] = float((pd.Timestamp(c["event_time"].min())-t).total_seconds()/60)
    return out


def test_vectorized_strength_labels_match_reference_across_chunks() -> None:
    rng = np.random.default_rng(123)
    offsets = (10.0, 30.0, 50.0, 70.0)
    decision_times = pd.date_range("2025-01-01", periods=9, freq="15min")
    rows = []
    for t in decision_times:
        price = 100.0 + rng.normal(0, 0.2)
        for side in ("DOWN", "UP"):
            for d in offsets:
                if rng.random() < 0.2:  # mimic sampled R02 spatial lattices
                    continue
                rows.append({"decision_time":t,"current_price":price,"zone_side":side,"zone_distance_bp":d})
    zones = pd.DataFrame(rows)
    ep_rows=[]
    for _ in range(100):
        t = decision_times[0] + pd.Timedelta(minutes=float(rng.uniform(-10, 240)))
        side = "DOWN" if rng.random() < 0.5 else "UP"
        anchor = float(100.0 + rng.normal(0, 0.2))
        dist = float(rng.uniform(0.1, 79.9))
        ref = anchor*(1-dist/1e4) if side == "DOWN" else anchor*(1+dist/1e4)
        ep_rows.append({
            "event_time":t,"event_side":side,"event_reference_price":ref,
            "release_density_proxy":float(max(0,rng.normal(2,1))),"release_episode_size":int(rng.integers(1,8)),
            "release_score":float(max(0,rng.normal(3,1))),"favorable_reversal":bool(rng.random()<0.35),
            "outcome_type":"ACCEPT_CONTINUATION" if rng.random()<0.25 else "MIXED",
            "future_extension_bp":float(rng.normal(25,8)),"future_reversal_after_extreme_bp":float(rng.normal(45,15)),
        })
    # Explicit exclusive-right-edge Episode.
    ep_rows.append({
        "event_time":decision_times[0]+pd.Timedelta(minutes=180),"event_side":"DOWN","event_reference_price":99.9,
        "release_density_proxy":9.0,"release_episode_size":9,"release_score":9.0,"favorable_reversal":True,
        "outcome_type":"EXTEND_STABILIZE_REVERSAL","future_extension_bp":20.0,"future_reversal_after_extreme_bp":50.0,
    })
    episodes=pd.DataFrame(ep_rows)
    ref=_reference_strength_labels(zones,episodes,horizon_minutes=180,offsets=offsets,half=10.0)
    fast=attach_strength_labels(zones,episodes,horizon_minutes=180,zone_offsets_bp=offsets,zone_half_width_bp=10.0,decision_chunk_size=3)
    cols=[
        "release_episode_count","release_density_sum","release_density_max","release_episode_size_sum","release_score_sum",
        "favorable_episode_count","continuation_episode_count","favorable_density_sum","continuation_density_sum",
        "sweep_depth_weighted_bp","reversal_room_weighted_bp","first_release_minutes",
    ]
    for col in cols:
        assert np.allclose(pd.to_numeric(fast[col],errors="coerce"),pd.to_numeric(ref[col],errors="coerce"),equal_nan=True,rtol=1e-9,atol=1e-7), col
    assert fast["release_within_horizon"].tolist() == ref["release_within_horizon"].tolist()
