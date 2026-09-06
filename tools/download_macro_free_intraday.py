#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Free macro intraday downloader for CoinBacktest
================================================

Data sources (NO API KEY):
1) Dukascopy via `npx dukascopy-node`
   - DXY 1m proxy: `dollaridxusd`
   - Long-duration U.S. Treasury bond 1m proxy: `ustbondtrusd`
2) Yahoo Finance via yfinance
   - 2Y Treasury futures: `ZT=F`
   - 10Y Treasury futures: `ZN=F`
   - Intraday history is limited; the script automatically tries recent 1m chunks.
3) FRED CSV
   - DGS2: 2-Year Treasury yield (daily)
   - DGS10: 10-Year Treasury yield (daily)

IMPORTANT
---------
- `ustbondtrusd` is a LONG-BOND proxy. It is NOT the 2Y or 10Y yield itself.
- ZT/ZN are Treasury FUTURES PRICES, so price direction is inverse to yield direction:
      ZT down ~ 2Y yield up  -> hawkish
      ZN down ~ 10Y yield up -> valuation pressure
- DXY up is usually tighter financial conditions for tech/crypto.
- Yahoo 1m history is limited. Older 2026 events may have no ZT/ZN minute bars.
- Dukascopy DXY / USTBond history is the main free full-period intraday source.

Requirements
------------
Python:
    pip install pandas yfinance pyarrow

Node.js:
    Node.js must be installed so that `npx` works.
    The script uses:
        npx -y dukascopy-node ...

Run
---
Default: built-in 2026 event study
    python download_macro_free_intraday.py

Custom output directory:
    python download_macro_free_intraday.py --out-dir data/macro_free

Only download / rebuild:
    python download_macro_free_intraday.py --pre-min 30 --post-min 90

Outputs
-------
data/macro_free/
    raw/
        dxy_1m.csv
        ustbond_1m.csv
        zt_1m.csv
        zn_1m.csv
        fred_dgs2_dgs10_daily.csv

    macro_event_bars_free_2026.csv
    macro_event_bars_free_2026.parquet
    macro_event_summary_free_2026.csv

The event summary contains T+1m / 5m / 15m / 30m / 60m moves.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


BJT = "Asia/Shanghai"
NY = "America/New_York"
UTC = "UTC"

HORIZONS_MIN = [1, 5, 15, 30, 60]

# Beijing event anchors used in our current 2026 study.
EVENTS_2026_BJT = [
    ("2026-03-19 02:00", "FOMC"),
    ("2026-04-03 20:30", "NFP+Unemployment"),
    ("2026-04-10 20:30", "CPI"),
    ("2026-04-30 02:00", "FOMC"),
    ("2026-04-30 20:30", "Core PCE"),
    ("2026-05-01 22:00", "ISM Manufacturing"),
    ("2026-05-08 20:30", "NFP+Unemployment"),
    ("2026-05-12 20:30", "CPI"),
    ("2026-05-13 20:30", "PPI"),
    ("2026-05-28 20:30", "Core PCE"),
    ("2026-06-05 20:30", "NFP+Unemployment"),
    ("2026-06-10 20:30", "CPI"),
    ("2026-06-11 20:30", "PPI"),
    ("2026-06-17 20:30", "Retail Sales"),
    ("2026-06-18 02:00", "FOMC"),
    ("2026-06-25 20:30", "Core PCE"),
    ("2026-07-01 22:00", "ISM Manufacturing"),
    ("2026-07-02 20:30", "NFP+Unemployment"),
    ("2026-07-14 20:30", "CPI"),
    ("2026-07-15 20:30", "PPI"),
    ("2026-07-16 20:30", "Retail Sales"),
    ("2026-07-30 02:00", "FOMC"),
    ("2026-07-30 20:30", "Core PCE"),
    ("2026-08-03 22:00", "ISM Manufacturing"),
    ("2026-08-07 20:30", "NFP+Unemployment"),
    ("2026-08-12 20:30", "CPI"),
    ("2026-08-13 20:30", "PPI"),
    ("2026-08-14 20:30", "Retail Sales"),
    ("2026-08-26 20:30", "Core PCE"),
]

DUKASCOPY = {
    "DXY": "dollaridxusd",
    "USTBOND": "ustbondtrusd",
}

YAHOO = {
    "ZT": "ZT=F",
    "ZN": "ZN=F",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/macro_free")
    p.add_argument("--pre-min", type=int, default=30)
    p.add_argument("--post-min", type=int, default=90)
    p.add_argument(
        "--yahoo-lookback-days",
        type=int,
        default=59,
        help="How far back to TRY Yahoo intraday retrieval. Older bars may be unavailable.",
    )
    p.add_argument(
        "--skip-yahoo",
        action="store_true",
        help="Skip ZT/ZN recent Yahoo download.",
    )
    p.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip daily FRED yield background.",
    )
    p.add_argument(
        "--skip-dukascopy",
        action="store_true",
        help="Skip DXY/USTBond Dukascopy download and use cached CSVs if present.",
    )
    return p.parse_args()


def load_events() -> pd.DataFrame:
    e = pd.DataFrame(EVENTS_2026_BJT, columns=["event_time_bjt", "event"])
    e["event_time_bjt"] = pd.to_datetime(e["event_time_bjt"]).dt.tz_localize(BJT)
    e["event_time_utc"] = e["event_time_bjt"].dt.tz_convert(UTC)
    e["event_time_ny"] = e["event_time_bjt"].dt.tz_convert(NY)
    e["event_id"] = (
        e["event_time_bjt"].dt.strftime("%Y%m%d_%H%M")
        + "_"
        + e["event"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_")
    )
    return e


def _find_timestamp_col(df: pd.DataFrame) -> str:
    candidates = [
        "timestamp", "time", "date", "datetime", "gmt time",
        "ts", "ts_event", "index",
    ]
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    # Dukascopy-node CSV normally puts time first.
    return df.columns[0]


def _normalize_ohlc(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _find_timestamp_col(df)
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    # Sometimes a textual ISO field can parse without explicit utc.
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(UTC)
        else:
            ts = ts.dt.tz_convert(UTC)

    colmap = {str(c).strip().lower(): c for c in df.columns}

    def get_col(name: str) -> pd.Series:
        if name in colmap:
            return pd.to_numeric(df[colmap[name]], errors="coerce")
        return pd.Series(index=df.index, dtype="float64")

    out = pd.DataFrame({
        "ts_utc": ts,
        "open": get_col("open"),
        "high": get_col("high"),
        "low": get_col("low"),
        "close": get_col("close"),
        "volume": get_col("volume"),
    })

    # Fallback if OHLC names differ but file is position-based:
    if out["close"].isna().all() and len(df.columns) >= 5:
        numeric_candidates = list(df.columns[1:5])
        out["open"] = pd.to_numeric(df[numeric_candidates[0]], errors="coerce")
        out["high"] = pd.to_numeric(df[numeric_candidates[1]], errors="coerce")
        out["low"] = pd.to_numeric(df[numeric_candidates[2]], errors="coerce")
        out["close"] = pd.to_numeric(df[numeric_candidates[3]], errors="coerce")

    out = out.dropna(subset=["ts_utc", "close"]).copy()
    out["ts_bjt"] = out["ts_utc"].dt.tz_convert(BJT)
    out["instrument"] = instrument
    return out.sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)


def run_dukascopy(
    instrument_label: str,
    instrument_id: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    if shutil.which("npx") is None:
        raise RuntimeError(
            "npx not found. Install Node.js first, then reopen your terminal."
        )

    work = raw_dir / "_dukascopy_tmp"
    work.mkdir(parents=True, exist_ok=True)

    # Use a deterministic file name; CSV timestamps are requested in UTC ISO form.
    file_stem = f"{instrument_label.lower()}_dukascopy"
    cmd = [
        "npx", "-y", "dukascopy-node",
        "-i", instrument_id,
        "-from", start_date,
        "-to", end_date,
        "-t", "m1",
        "-p", "bid",
        "-v",
        "-f", "csv",
        "-dir", str(work),
        "-fn", file_stem,
        "-df", "iso",
        "-tz", "UTC",
        "-r", "2",
        "-rp", "1000",
    ]

    print(f"\n[Dukascopy] {instrument_label}: {start_date} -> {end_date}")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    candidates = sorted(work.glob(f"{file_stem}*.csv"))
    if not candidates:
        # Fallback: find any newly-created CSV containing instrument id.
        candidates = sorted(work.glob("*.csv"))

    if not candidates:
        raise RuntimeError(f"No Dukascopy CSV generated for {instrument_label}")

    # If CLI emitted several batched files, concatenate all of them.
    pieces = []
    for f in candidates:
        try:
            x = pd.read_csv(f)
            if not x.empty:
                pieces.append(x)
        except Exception as e:
            print(f"[warn] Could not read {f}: {e}")

    if not pieces:
        raise RuntimeError(f"Dukascopy files for {instrument_label} were empty")

    df = pd.concat(pieces, ignore_index=True)
    out = _normalize_ohlc(df, instrument_label)

    save_path = raw_dir / f"{instrument_label.lower()}_1m.csv"
    out.to_csv(save_path, index=False)
    print(f"[saved] {save_path} ({len(out):,} rows)")
    return out


def _download_yahoo_one(
    ticker: str,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    import yfinance as yf

    pieces = []
    cursor = start

    # 1m requests are deliberately chunked to 7 days.
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=7), end)
        try:
            x = yf.download(
                ticker,
                start=cursor.strftime("%Y-%m-%d"),
                end=(chunk_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1m",
                auto_adjust=False,
                progress=False,
                prepost=True,
                threads=False,
            )
        except Exception as e:
            print(f"[warn] Yahoo {ticker} {cursor.date()} failed: {e}")
            cursor = chunk_end
            continue

        if not x.empty:
            if isinstance(x.columns, pd.MultiIndex):
                # yfinance may return (Price, Ticker) columns.
                x.columns = [c[0] for c in x.columns]

            x = x.reset_index()
            ts_col = x.columns[0]
            ts = pd.to_datetime(x[ts_col], errors="coerce")

            if getattr(ts.dt, "tz", None) is None:
                # Yahoo US futures timestamps normally come localized; fallback to NY.
                ts = ts.dt.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")
            ts = ts.dt.tz_convert(UTC)

            part = pd.DataFrame({
                "ts_utc": ts,
                "open": pd.to_numeric(x.get("Open"), errors="coerce"),
                "high": pd.to_numeric(x.get("High"), errors="coerce"),
                "low": pd.to_numeric(x.get("Low"), errors="coerce"),
                "close": pd.to_numeric(x.get("Close"), errors="coerce"),
                "volume": pd.to_numeric(x.get("Volume"), errors="coerce"),
            }).dropna(subset=["ts_utc", "close"])

            pieces.append(part)

        cursor = chunk_end

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values("ts_utc").drop_duplicates("ts_utc")
    out["ts_bjt"] = out["ts_utc"].dt.tz_convert(BJT)
    out["instrument"] = label
    return out.reset_index(drop=True)


def download_yahoo_recent(
    events: pd.DataFrame,
    raw_dir: Path,
    lookback_days: int,
) -> dict[str, pd.DataFrame]:
    now_utc = pd.Timestamp.now(tz=UTC)
    requested_start = events["event_time_utc"].min() - pd.Timedelta(days=1)
    start = max(requested_start, now_utc - pd.Timedelta(days=lookback_days))
    end = now_utc + pd.Timedelta(days=1)

    result = {}
    for label, ticker in YAHOO.items():
        print(f"\n[Yahoo] {label} {ticker}: trying {start.date()} -> {end.date()}")
        df = _download_yahoo_one(ticker, label, start, end)
        result[label] = df

        save_path = raw_dir / f"{label.lower()}_1m.csv"
        df.to_csv(save_path, index=False)
        print(f"[saved] {save_path} ({len(df):,} rows)")
    return result


def download_fred(raw_dir: Path) -> pd.DataFrame:
    # No API key needed for fredgraph CSV.
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=DGS2,DGS10"
    )
    print("\n[FRED] downloading DGS2 + DGS10 daily")
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date"})
    for c in ("DGS2", "DGS10"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace(".", pd.NA), errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    if "DGS2" in df.columns:
        df["DGS2_change_bp"] = df["DGS2"].diff() * 100.0
    if "DGS10" in df.columns:
        df["DGS10_change_bp"] = df["DGS10"].diff() * 100.0

    save_path = raw_dir / "fred_dgs2_dgs10_daily.csv"
    df.to_csv(save_path, index=False)
    print(f"[saved] {save_path} ({len(df):,} rows)")
    return df


def nearest_close(
    g: pd.DataFrame,
    target: pd.Timestamp,
    tolerance_min: int = 3,
) -> Optional[float]:
    if g.empty:
        return None
    diffs = (g["ts_utc"] - target).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(minutes=tolerance_min):
        return None
    return float(g.loc[idx, "close"])


def baseline_close(g: pd.DataFrame, t0: pd.Timestamp) -> Optional[float]:
    if g.empty:
        return None
    pre = g[g["ts_utc"] <= t0]
    if pre.empty:
        return None
    # Avoid using a stale quote from hours before the event.
    if t0 - pre.iloc[-1]["ts_utc"] > pd.Timedelta(minutes=5):
        return None
    return float(pre.iloc[-1]["close"])


def add_event_fields(
    event_row: pd.Series,
    bars: pd.DataFrame,
    instrument: str,
    out: dict,
) -> None:
    t0 = event_row["event_time_utc"]
    g = bars[
        (bars["instrument"] == instrument)
        & (bars["ts_utc"] >= t0 - pd.Timedelta(minutes=45))
        & (bars["ts_utc"] <= t0 + pd.Timedelta(minutes=120))
    ].copy()

    pre = baseline_close(g, t0)
    out[f"{instrument}_T0_close"] = pre

    for h in HORIZONS_MIN:
        px = nearest_close(g, t0 + pd.Timedelta(minutes=h))
        out[f"{instrument}_{h}m_close"] = px

        if pre is None or px is None or pre == 0:
            out[f"{instrument}_{h}m_px_move_bp"] = None
            out[f"{instrument}_{h}m_macro_signed_bp"] = None
            continue

        px_move_bp = (px / pre - 1.0) * 10000.0
        out[f"{instrument}_{h}m_px_move_bp"] = px_move_bp

        # Unified sign:
        # positive = tighter / more hawkish / greater tech headwind
        if instrument in ("USTBOND", "ZT", "ZN"):
            out[f"{instrument}_{h}m_macro_signed_bp"] = -px_move_bp
        else:  # DXY
            out[f"{instrument}_{h}m_macro_signed_bp"] = px_move_bp


def build_event_bars(
    events: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    pre_min: int,
    post_min: int,
) -> pd.DataFrame:
    parts = []

    for _, ev in events.iterrows():
        t0 = ev["event_time_utc"]
        for instrument, bars in data.items():
            if bars is None or bars.empty:
                continue

            x = bars[
                (bars["ts_utc"] >= t0 - pd.Timedelta(minutes=pre_min))
                & (bars["ts_utc"] <= t0 + pd.Timedelta(minutes=post_min))
            ].copy()

            if x.empty:
                continue

            x["event_id"] = ev["event_id"]
            x["event"] = ev["event"]
            x["event_time_bjt"] = ev["event_time_bjt"]
            x["event_time_utc"] = ev["event_time_utc"]
            x["minutes_from_event"] = (
                (x["ts_utc"] - t0).dt.total_seconds() / 60.0
            )
            parts.append(x)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True).sort_values(
        ["event_time_utc", "instrument", "ts_utc"]
    )


def latest_fred_for_event(
    fred: pd.DataFrame,
    event_time_ny: pd.Timestamp,
) -> dict:
    if fred is None or fred.empty:
        return {}

    event_date = pd.Timestamp(event_time_ny.date())
    available = fred[fred["date"] <= event_date].dropna(
        subset=[c for c in ("DGS2", "DGS10") if c in fred.columns],
        how="all",
    )
    if available.empty:
        return {}

    r = available.iloc[-1]
    out = {
        "FRED_obs_date": r["date"].date().isoformat(),
        "DGS2_daily_pct": r.get("DGS2"),
        "DGS10_daily_pct": r.get("DGS10"),
        "DGS2_daily_change_bp": r.get("DGS2_change_bp"),
        "DGS10_daily_change_bp": r.get("DGS10_change_bp"),
    }
    if pd.notna(out.get("DGS2_daily_pct")) and pd.notna(out.get("DGS10_daily_pct")):
        out["DGS2_DGS10_curve_bp"] = (
            float(out["DGS10_daily_pct"]) - float(out["DGS2_daily_pct"])
        ) * 100.0
    return out


def build_event_summary(
    events: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    fred: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, ev in events.iterrows():
        row = {
            "event_id": ev["event_id"],
            "event": ev["event"],
            "event_time_bjt": ev["event_time_bjt"],
            "event_time_utc": ev["event_time_utc"],
            "event_time_ny": ev["event_time_ny"],
        }

        for instrument in ("DXY", "USTBOND", "ZT", "ZN"):
            bars = data.get(instrument, pd.DataFrame())
            if bars is not None and not bars.empty:
                add_event_fields(ev, bars, instrument, row)

        row.update(latest_fred_for_event(fred, ev["event_time_ny"]))
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Helpful confirmation flags using the 5m response.
    # macro_signed > 0 = tighter / bearish-for-tech direction.
    def classify(row):
        vals = []
        for key in (
            "DXY_5m_macro_signed_bp",
            "USTBOND_5m_macro_signed_bp",
            "ZT_5m_macro_signed_bp",
            "ZN_5m_macro_signed_bp",
        ):
            v = row.get(key)
            if pd.notna(v):
                vals.append(float(v))

        if not vals:
            return "NO_INTRADAY_DATA"

        pos = sum(v > 0 for v in vals)
        neg = sum(v < 0 for v in vals)

        if pos >= 2 and pos > neg:
            return "TIGHTER_5M"
        if neg >= 2 and neg > pos:
            return "EASIER_5M"
        return "MIXED_5M"

    summary["cross_asset_5m_state"] = summary.apply(classify, axis=1)
    return summary


def load_cached(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_csv(path)
    if x.empty:
        return pd.DataFrame()

    x["ts_utc"] = pd.to_datetime(x["ts_utc"], utc=True)
    if "ts_bjt" in x.columns:
        x["ts_bjt"] = pd.to_datetime(x["ts_bjt"], utc=True).dt.tz_convert(BJT)
    else:
        x["ts_bjt"] = x["ts_utc"].dt.tz_convert(BJT)
    x["instrument"] = label
    return x


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    events = load_events()

    # Do not ask Dukascopy for future dates.
    now_bjt = pd.Timestamp.now(tz=BJT)
    historical_events = events[
        events["event_time_bjt"] <= now_bjt + pd.Timedelta(days=1)
    ].copy()

    if historical_events.empty:
        raise RuntimeError("No historical event anchors available.")

    start_date = (
        historical_events["event_time_utc"].min() - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    end_date = (
        min(
            historical_events["event_time_utc"].max() + pd.Timedelta(days=2),
            pd.Timestamp.now(tz=UTC) + pd.Timedelta(days=1),
        )
    ).strftime("%Y-%m-%d")

    data: dict[str, pd.DataFrame] = {}

    # ---------- Dukascopy ----------
    for label, instrument_id in DUKASCOPY.items():
        cache = raw_dir / f"{label.lower()}_1m.csv"
        if args.skip_dukascopy:
            print(f"[cache] {label}: {cache}")
            data[label] = load_cached(cache, label)
            continue

        try:
            data[label] = run_dukascopy(
                label,
                instrument_id,
                start_date,
                end_date,
                raw_dir,
            )
        except Exception as e:
            print(f"[ERROR] Dukascopy {label}: {e}")
            print("[fallback] trying cached file")
            data[label] = load_cached(cache, label)

    # ---------- Yahoo recent ZT / ZN ----------
    if args.skip_yahoo:
        for label in YAHOO:
            data[label] = load_cached(raw_dir / f"{label.lower()}_1m.csv", label)
    else:
        try:
            data.update(
                download_yahoo_recent(
                    historical_events,
                    raw_dir,
                    args.yahoo_lookback_days,
                )
            )
        except Exception as e:
            print(f"[ERROR] Yahoo download: {e}")
            for label in YAHOO:
                data[label] = load_cached(
                    raw_dir / f"{label.lower()}_1m.csv",
                    label,
                )

    # ---------- FRED daily background ----------
    fred = pd.DataFrame()
    if args.skip_fred:
        fred_path = raw_dir / "fred_dgs2_dgs10_daily.csv"
        if fred_path.exists():
            fred = pd.read_csv(fred_path)
            fred["date"] = pd.to_datetime(fred["date"])
    else:
        try:
            fred = download_fred(raw_dir)
        except Exception as e:
            print(f"[ERROR] FRED download: {e}")
            fred_path = raw_dir / "fred_dgs2_dgs10_daily.csv"
            if fred_path.exists():
                fred = pd.read_csv(fred_path)
                fred["date"] = pd.to_datetime(fred["date"])

    # ---------- Event outputs ----------
    bars = build_event_bars(
        historical_events,
        data,
        pre_min=args.pre_min,
        post_min=args.post_min,
    )

    summary = build_event_summary(
        historical_events,
        data,
        fred,
    )

    bars_csv = out_dir / "macro_event_bars_free_2026.csv"
    bars_pq = out_dir / "macro_event_bars_free_2026.parquet"
    summary_csv = out_dir / "macro_event_summary_free_2026.csv"

    bars.to_csv(bars_csv, index=False)
    if not bars.empty:
        try:
            bars.to_parquet(bars_pq, index=False)
        except Exception as e:
            print(f"[warn] parquet write skipped: {e}")
    summary.to_csv(summary_csv, index=False)

    print("\n================ DONE ================")
    print(f"Event bars:    {bars_csv}")
    if bars_pq.exists():
        print(f"Event parquet: {bars_pq}")
    print(f"Event summary: {summary_csv}")
    print("\nSend me macro_event_summary_free_2026.csv first.")
    print("If needed, also send macro_event_bars_free_2026.csv/parquet.")


if __name__ == "__main__":
    main()
