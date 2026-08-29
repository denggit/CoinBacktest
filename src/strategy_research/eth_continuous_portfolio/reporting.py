from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import ContinuousBacktestResult


def yearly_table(spec_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    ret = daily["equity"].pct_change().fillna(0.0)
    out = ret.groupby(daily.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    return pd.DataFrame({"spec_id": spec_id, "year": out.index.astype(int), "return_pct": out.to_numpy(float) * 100.0})


def monthly_table(spec_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    ret = daily["equity"].pct_change().fillna(0.0)
    out = ret.groupby(daily.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    return pd.DataFrame({"spec_id": spec_id, "month": out.index.astype(str), "return_pct": out.to_numpy(float) * 100.0})


def top_day_dependency(spec_id: str, daily: pd.DataFrame) -> list[dict[str, float | int | str]]:
    ret = daily["equity"].pct_change().fillna(0.0)
    rows: list[dict[str, float | int | str]] = []
    for n in (1, 5, 10):
        x = ret.copy()
        top = x.nlargest(min(n, len(x))).index
        x.loc[top] = 0.0
        total = float((1.0 + x).prod() - 1.0)
        rows.append({"spec_id": spec_id, "remove_top_positive_days": n, "return_pct_after_removal": total * 100.0})
    return rows


def write_review_pack(root: Path) -> Path:
    pack = root / "gpt_review_pack.zip"
    names = [
        "00_portfolio_specs.csv",
        "01_summary.csv",
        "02_yearly.csv",
        "03_monthly.csv",
        "04_cost_delay_stress.csv",
        "05_top_day_dependency.csv",
        "06_causal_audit.csv",
        "07_sleeve_snapshot.csv",
        "08_selection.csv",
        "99_decision.md",
    ]
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            p = root / name
            if p.exists():
                zf.write(p, arcname=name)
        for p in (root / "specs").rglob("summary.csv") if (root / "specs").exists() else []:
            zf.write(p, arcname=p.relative_to(root).as_posix())
    return pack
