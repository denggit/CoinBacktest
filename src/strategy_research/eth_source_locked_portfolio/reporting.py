from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


def yearly(strategy_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    r = daily["equity"].pct_change().fillna(0.0)
    x = r.groupby(daily.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
    return pd.DataFrame({"strategy_id": strategy_id, "year": x.index.astype(int), "return_pct": x.to_numpy(float) * 100.0})


def monthly(strategy_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    r = daily["equity"].pct_change().fillna(0.0)
    x = r.groupby(daily.index.to_period("M")).apply(lambda s: (1.0 + s).prod() - 1.0)
    return pd.DataFrame({"strategy_id": strategy_id, "month": x.index.astype(str), "return_pct": x.to_numpy(float) * 100.0})


def top_days(strategy_id: str, daily: pd.DataFrame) -> list[dict[str, float | int | str]]:
    r = daily["equity"].pct_change().fillna(0.0)
    rows = []
    for n in (1, 5, 10):
        y = r.copy()
        y.loc[y.nlargest(min(n, len(y))).index] = 0.0
        rows.append({"strategy_id": strategy_id, "remove_top_positive_days": n, "return_pct_after_removal": float(((1.0 + y).prod() - 1.0) * 100.0)})
    return rows


def write_review_pack(root: Path) -> Path:
    pack = root / "gpt_review_pack.zip"
    names = [
        "00_source_catalog.csv", "01_summary.csv", "02_yearly.csv", "03_monthly.csv",
        "04_cost_delay_stress.csv", "05_top_day_dependency.csv", "06_causal_audit.csv",
        "07_rule_interpretations.csv", "08_selection.csv", "99_decision.md",
    ]
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            p = root / name
            if p.exists():
                zf.write(p, arcname=name)
    return pack
