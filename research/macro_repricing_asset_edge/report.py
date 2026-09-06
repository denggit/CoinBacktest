from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_sources import Coverage


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, columns: list[tuple[str, str, int | None]], *, limit: int = 80) -> str:
    if frame.empty:
        return '<p class="empty">No observations available.</p>'
    work = frame.head(limit)
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    rows: list[str] = []
    for _, row in work.iterrows():
        cells = []
        for column, _, digits in columns:
            value = row.get(column)
            cells.append(f"<td>{_fmt(value, digits or 0)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    suffix = f'<p class="footnote">Showing {len(work)} of {len(frame)} rows.</p>' if len(frame) > len(work) else ""
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{suffix}'


def _mean_return_chart(summary: pd.DataFrame) -> str:
    if summary.empty or "mean_return_pct" not in summary:
        return '<p class="empty">Not enough aligned observations for a chart.</p>'
    work = summary.sort_values(["asset", "regime", summary.columns[summary.columns.str.startswith("horizon_")][0]])
    work = work.head(36)
    values = pd.to_numeric(work["mean_return_pct"], errors="coerce").fillna(0.0)
    maximum = max(float(values.abs().max()), 0.01)
    width = 920
    row_height = 25
    top = 34
    middle = 520
    height = top + len(work) * row_height + 28
    horizon_column = "horizon_minutes" if "horizon_minutes" in work else "horizon_sessions"
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Mean forward return by asset, regime and horizon">',
        f'<line x1="{middle}" y1="20" x2="{middle}" y2="{height - 12}" class="zero"/>',
    ]
    for i, (_, row) in enumerate(work.iterrows()):
        y = top + i * row_height
        value = float(row["mean_return_pct"])
        length = min(abs(value) / maximum * 340.0, 340.0)
        x = middle if value >= 0 else middle - length
        tone = "positive" if value >= 0 else "negative"
        label = f"{row['asset']} · {row['regime']} · {int(row[horizon_column])}{'m' if horizon_column == 'horizon_minutes' else 'd'}"
        parts.append(f'<text x="8" y="{y + 4}" class="axis-label">{html.escape(label)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y - 9}" width="{length:.1f}" height="14" rx="3" class="bar {tone}"/>')
        value_x = x + length + 6 if value >= 0 else x - 6
        anchor = "start" if value >= 0 else "end"
        parts.append(f'<text x="{value_x:.1f}" y="{y + 3}" text-anchor="{anchor}" class="value">{value:+.3f}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_html_report(
    output_path: str | Path,
    *,
    coverages: Iterable[Coverage],
    intraday_events: pd.DataFrame,
    intraday_summary: pd.DataFrame,
    daily_proxy_events: pd.DataFrame,
    daily_proxy_summary: pd.DataFrame,
    daily_proxy_net_summary: pd.DataFrame,
    daily_proxy_fold_summary: pd.DataFrame,
    daily_proxy_ablation_summary: pd.DataFrame,
    static_background_events: pd.DataFrame,
    static_background_summary: pd.DataFrame,
    scheduled_signals: pd.DataFrame,
    scheduled_summary: pd.DataFrame,
    scheduled_event_type_summary: pd.DataFrame,
    scheduled_latency_cost_summary: pd.DataFrame,
    scheduled_leave_one_event_out: pd.DataFrame,
    scheduled_threshold_sensitivity: pd.DataFrame,
    scheduled_data_quality: pd.DataFrame,
    generated_at_utc: pd.Timestamp,
) -> Path:
    coverage_rows = []
    for item in coverages:
        coverage_rows.append(
            {
                "source": item.source,
                "dataset": item.dataset,
                "rows": item.rows,
                "start": item.start_utc,
                "end": item.end_utc,
                "notes": item.notes,
            }
        )
    coverage_frame = pd.DataFrame(coverage_rows)
    hf_unique = int(intraday_events["timestamp_utc"].nunique()) if not intraday_events.empty else 0
    daily_unique = int(daily_proxy_events.index.nunique()) if not daily_proxy_events.empty else 0
    daily_overlap = 0
    if not daily_proxy_summary.empty:
        daily_overlap = int(daily_proxy_summary.groupby("regime")["events"].max().sum())
    high_frequency_grade = "CASE STUDY ONLY" if hf_unique < 20 else "PRELIMINARY"
    daily_grade = "INSUFFICIENT" if daily_overlap < 50 else "PROXY SAMPLE ADEQUATE"
    scheduled_non_stable = (
        scheduled_signals.loc[scheduled_signals["regime"].ne("stable")]
        if not scheduled_signals.empty
        else pd.DataFrame()
    )
    scheduled_events = (
        int(scheduled_non_stable["event_id"].nunique())
        if not scheduled_non_stable.empty
        else 0
    )
    scheduled_grade = "CASE STUDY ONLY" if scheduled_events < 20 else "PRELIMINARY"
    best_rows = pd.DataFrame()
    if not daily_proxy_summary.empty:
        best_rows = daily_proxy_summary.sort_values(
            ["bh_adjusted_p_value", "events"], ascending=[True, False], na_position="last"
        ).head(12)
    net_rows = pd.DataFrame()
    if not daily_proxy_net_summary.empty:
        net_rows = daily_proxy_net_summary.sort_values(
            ["bh_adjusted_p_value", "events"], ascending=[True, False], na_position="last"
        ).head(12)
    candidate_text = "No cost-stressed candidate is currently defined."
    if not daily_proxy_net_summary.empty:
        candidate = daily_proxy_net_summary.loc[
            daily_proxy_net_summary["asset"].isin(["QQQ", "SOXX"])
            & daily_proxy_net_summary["regime"].eq("dovish")
            & daily_proxy_net_summary["horizon_sessions"].eq(5)
        ].sort_values("asset")
        if len(candidate) == 2:
            parts = [
                f"{row.asset} net mean {row.mean_return_pct:+.3f}% (BH p={row.bh_adjusted_p_value:.3f})"
                for row in candidate.itertuples()
            ]
            candidate_text = "; ".join(parts) + "."
    fold_consistency = pd.DataFrame()
    if not daily_proxy_fold_summary.empty:
        fold_consistency = (
            daily_proxy_fold_summary.groupby(["asset", "regime", "horizon_sessions"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                positive_folds=("mean_return_pct", lambda x: int((x > 0).sum())),
                min_fold_mean_pct=("mean_return_pct", "min"),
                max_fold_mean_pct=("mean_return_pct", "max"),
                total_events=("events", "sum"),
            )
        )
        fold_consistency["same_sign_all_folds"] = (
            (fold_consistency["positive_folds"].eq(0) | fold_consistency["positive_folds"].eq(fold_consistency["folds"]))
            & fold_consistency["folds"].ge(3)
        )
        fold_consistency = fold_consistency.sort_values(
            ["same_sign_all_folds", "total_events"], ascending=[False, False]
        ).head(20)
    ablation_rows = pd.DataFrame()
    if not daily_proxy_ablation_summary.empty:
        ablation_rows = daily_proxy_ablation_summary.sort_values(
            ["signal_subset", "bh_adjusted_p_value", "events"],
            ascending=[True, True, False],
            na_position="last",
        ).groupby("signal_subset", as_index=False, group_keys=False).head(4)
    static_rows = pd.DataFrame()
    if not static_background_summary.empty:
        static_rows = static_background_summary.sort_values(
            ["bh_adjusted_p_value", "events"], ascending=[True, False], na_position="last"
        ).head(15)

    scheduled_best = pd.DataFrame()
    if not scheduled_summary.empty:
        scheduled_best = scheduled_summary.loc[
            scheduled_summary["signal_delay_minutes"].eq(5)
            & scheduled_summary["execution_delay_minutes"].eq(0)
        ].sort_values(
            ["bh_adjusted_p_value", "events"], ascending=[True, False], na_position="last"
        ).head(16)
    scheduled_cost_rows = pd.DataFrame()
    if not scheduled_latency_cost_summary.empty:
        scheduled_cost_rows = scheduled_latency_cost_summary.loc[
            scheduled_latency_cost_summary["signal_delay_minutes"].eq(5)
            & scheduled_latency_cost_summary["horizon_minutes"].eq(60)
        ].sort_values(
            ["asset", "regime", "execution_delay_minutes", "cost_stress_bp"]
        ).head(80)
    scheduled_event_type_rows = pd.DataFrame()
    if not scheduled_event_type_summary.empty:
        scheduled_event_type_rows = scheduled_event_type_summary.loc[
            scheduled_event_type_summary["signal_delay_minutes"].eq(5)
            & scheduled_event_type_summary["execution_delay_minutes"].eq(0)
            & scheduled_event_type_summary["horizon_minutes"].eq(60)
        ].sort_values(["event_type", "asset", "regime"]).head(80)
    scheduled_loo_rows = pd.DataFrame()
    if not scheduled_leave_one_event_out.empty:
        scheduled_loo_rows = scheduled_leave_one_event_out.loc[
            scheduled_leave_one_event_out["signal_delay_minutes"].eq(5)
            & scheduled_leave_one_event_out["execution_delay_minutes"].eq(0)
            & scheduled_leave_one_event_out["horizon_minutes"].eq(60)
        ].sort_values(["same_sign_all_loo", "events"], ascending=[False, False]).head(30)
    threshold_rows = pd.DataFrame()
    if not scheduled_threshold_sensitivity.empty:
        threshold_rows = scheduled_threshold_sensitivity.loc[
            scheduled_threshold_sensitivity["signal_delay_minutes"].eq(5)
            & scheduled_threshold_sensitivity["execution_delay_minutes"].eq(0)
            & scheduled_threshold_sensitivity["horizon_minutes"].eq(60)
        ].sort_values(["asset", "regime", "threshold_multiplier"]).head(80)

    scheduled_display = scheduled_non_stable.copy()
    if not scheduled_display.empty:
        scheduled_display["event_time_bjt_display"] = pd.to_datetime(
            scheduled_display["event_time_utc"], utc=True
        ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M")

    event_display = intraday_events.copy()
    if not event_display.empty:
        event_display["timestamp_bjt"] = pd.to_datetime(event_display["timestamp_utc"], utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M:%S")

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Macro Repricing Signals: Cross-Asset Edge Audit</title>
<style>
:root{{--bg:#f4f6f8;--paper:#fff;--ink:#16202a;--muted:#667384;--line:#dfe5eb;--navy:#102a43;--cyan:#0e7490;--green:#087f5b;--red:#c92a2a;--amber:#b7791f;--soft:#eef3f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e151c;--paper:#151f29;--ink:#e7edf2;--muted:#9cabb9;--line:#2a3947;--navy:#c9e2f5;--cyan:#5fd1e6;--green:#63d6ae;--red:#ff8787;--amber:#ffd080;--soft:#1b2935}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}} main{{max-width:1160px;margin:auto;padding:42px 28px 80px}} h1{{font-size:38px;line-height:1.1;letter-spacing:-.03em;margin:0 0 12px;color:var(--navy)}} h2{{font-size:23px;margin:0 0 15px;color:var(--navy)}} h3{{font-size:16px;margin:0 0 8px}} .kicker{{text-transform:uppercase;letter-spacing:.16em;color:var(--cyan);font-weight:800;font-size:12px}} .lede{{max-width:850px;font-size:18px;color:var(--muted)}} .meta{{font-size:13px;color:var(--muted);margin-top:16px}} section{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:25px;margin-top:20px;box-shadow:0 8px 28px rgba(11,29,44,.05)}} .summary{{border-left:5px solid var(--cyan)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}} .card{{background:var(--soft);border-radius:12px;padding:16px}} .big{{font-size:25px;font-weight:800;color:var(--navy)}} .label{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}} .warning{{color:var(--amber);font-weight:800}} .good{{color:var(--green);font-weight:800}} .bad{{color:var(--red);font-weight:800}} .callout{{border:1px solid var(--line);border-radius:12px;padding:16px;background:var(--soft)}} ul{{padding-left:20px}} .table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}} table{{width:100%;border-collapse:collapse;min-width:760px;font-size:13px}} th{{text-align:left;background:var(--soft);color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-size:11px}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}} tbody tr:last-child td{{border-bottom:0}} .footnote,.empty{{color:var(--muted);font-size:13px}} .chart{{display:block;width:100%;height:auto;min-height:280px;background:var(--soft);border-radius:12px}} .zero{{stroke:var(--muted);stroke-width:1}} .bar.positive{{fill:var(--green)}} .bar.negative{{fill:var(--red)}} .axis-label,.value{{fill:var(--ink);font-size:11px}} code{{background:var(--soft);padding:2px 5px;border-radius:5px}} footer{{color:var(--muted);font-size:12px;margin-top:25px;text-align:center}}
@media(max-width:640px){{main{{padding:25px 13px 50px}}h1{{font-size:30px}}section{{padding:18px}}}}
</style>
</head>
<body><main>
<div class="kicker">CoinBacktest · Independent Research</div>
<h1>Macro Repricing Signals: Cross-Asset Edge Audit</h1>
<p class="lede">Does hawkish or dovish repricing in FedWatch, US2Y, US10Y and DXY precede a repeatable response in SOXX, SOXL, QQQ, ETH or gold?</p>
<p class="meta">Research cut-off: {html.escape(generated_at_utc.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S Beijing'))} · All calculations use UTC internally.</p>

<section class="summary">
<h2>Executive Summary</h2>
<ul>
<li><strong>The true FedWatch evidence is not yet an edge test.</strong> The local monitor contributes {hf_unique} thinned high-frequency event timestamps, so this track is graded <span class="warning">{high_frequency_grade}</span>.</li>
<li><strong>The new five-minute scheduled-release panel is also exploratory.</strong> It contains {scheduled_events} independent event anchors with a non-stable proxy signal and is graded <span class="warning">{scheduled_grade}</span>. It tests post-release continuation after a decision delay, not advance prediction.</li>
<li><strong>The long-history track is a proxy test.</strong> It finds {daily_unique} rate-history signals, of which {daily_overlap} overlap the tested 2019–2026 asset histories. It must not be interpreted as historical CME probability data.</li>
<li><strong>The predeclared candidate is US2Y-led dovish repricing followed by 5-session QQQ/SOXX strength.</strong> {html.escape(candidate_text)} This remains preliminary after cost and multiple-testing stress.</li>
<li><strong>SOXL is measured directly.</strong> Its results include actual leverage path and volatility decay; no SOXX result is multiplied by three.</li>
<li><strong>No trading rule is promoted here.</strong> A signal needs adequate sample size, stable direction across subperiods, cost tolerance and corrected significance before it can graduate from watchlist research.</li>
</ul>
</section>

<section><h2>Evidence Readiness</h2><div class="grid">
<div class="card"><div class="label">True monitor events</div><div class="big">{hf_unique}</div><div class="warning">{high_frequency_grade}</div></div>
<div class="card"><div class="label">Scheduled 5m events</div><div class="big">{scheduled_events}</div><div class="warning">{scheduled_grade}</div></div>
<div class="card"><div class="label">Asset-overlap proxy events</div><div class="big">{daily_overlap}</div><div class="{'good' if daily_overlap >= 50 else 'warning'}">{daily_grade}</div></div>
<div class="card"><div class="label">Entry convention</div><div class="big">Next bar</div><div class="footnote">Strictly after signal timestamp</div></div>
<div class="card"><div class="label">Multiple tests</div><div class="big">BH-FDR</div><div class="footnote">Applied to reported p-values</div></div>
</div></section>

<section><h2>Source And Freshness</h2>
<p>True monitor observations come from the local SQLite collector. Event-window listed-equity minutes come from Alpaca SIP; ETH/XAU short-window bars come from OKX. FRED is the daily rates source. Yahoo Finance supplies adjusted ETF daily histories plus DXY, `ZQ=F`, and `GC=F` proxy histories. Missing or stale inputs remain visible below.</p>
{_table(coverage_frame, [('source','Source',None),('dataset','Dataset',None),('rows','Rows',None),('start','Start',None),('end','End',None),('notes','Notes',None)])}
</section>

<section><h2>Five-Minute Scheduled-Release Study</h2>
<p>This iteration observes the first 5, 10, or 15 minutes after CPI, PPI, NFP, FOMC, Core PCE, ISM and Retail Sales anchors. A trade-response clock starts only after that observation window, and execution is stressed by another 0, 5, or 10 minutes. This removes event-time look-ahead but means the study tests continuation or reversal after repricing has already begun.</p>
<div class="callout"><strong>Source semantics:</strong> CNBC/Tradeweb rows are exact US2Y and US10Y yields but cover only the recent public intraday window. The 60-day track uses the October 2026 30-Day Fed Funds Futures contract as a post-September-FOMC implied-rate proxy, `ZT=F` as a 2Y Treasury futures <em>price</em> diagnostic, `^TNX` as a 10Y yield quote, and DXY. None of these rows is historical FedWatch probability.</div>
<h3>Non-stable proxy signals</h3>
{_table(scheduled_display, [('event_time_bjt_display','Beijing time',None),('event_type','Event',None),('signal_delay_minutes','Observed min',0),('regime','Proxy regime',None),('severity','Severity',0),('zq_post_fomc_implied_rate_change_bp','ZQ implied Δ bp',2),('zt_tightening_price_proxy_bp','ZT tightening proxy',2),('us2y_exact_change_bp','US2Y exact Δ bp',2),('us10y_exact_change_bp','US10Y exact Δ bp',2),('us10y_yahoo_change_bp','US10Y quote Δ bp',2),('dxy_change_pct','DXY Δ %',3),('drivers','Drivers',None)], limit=80)}
<p>The table below fixes the macro observation delay at 5 minutes and enters on the first subsequent asset bar. Means are conditional responses with event-id clustering; they are not causal estimates.</p>
{_mean_return_chart(scheduled_best)}
{_table(scheduled_best, [('asset','Asset',None),('regime','Proxy regime',None),('horizon_minutes','Forward min',0),('events','Rows',0),('independent_clusters','Events',0),('mean_return_pct','Gross mean %',3),('median_return_pct','Median %',3),('hit_rate_positive_pct','Positive %',1),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4),('evidence_grade','Grade',None)])}
</section>

<section><h2>Latency, Cost And Event Heterogeneity</h2>
<p>Latency and cost are part of the hypothesis, not an afterthought. The rows below hold the macro observation window at 5 minutes and compare 60-minute asset responses under additional execution delay and 0/5/10 bp round-trip stress.</p>
{_table(scheduled_cost_rows, [('asset','Asset',None),('regime','Proxy regime',None),('execution_delay_minutes','Exec delay min',0),('cost_stress_bp','Cost bp',0),('events','Rows',0),('independent_clusters','Events',0),('mean_return_pct','Mean after cost %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4)])}
<p>Event-type cuts are deliberately shown even when sparse. A result concentrated in one CPI or FOMC row is a case observation, not a reusable edge.</p>
{_table(scheduled_event_type_rows, [('event_type','Event type',None),('asset','Asset',None),('regime','Proxy regime',None),('events','Rows',0),('independent_clusters','Events',0),('mean_return_pct','Net mean %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('evidence_grade','Grade',None)], limit=80)}
</section>

<section><h2>Small-Sample Robustness</h2>
<p>Leave-one-event-out ranges test whether removing any one release flips the 5 bp cost-stressed 60-minute mean. Threshold sensitivity repeats the same read at 0.75×, 1.0× and 1.5× the predeclared proxy thresholds. With this sample, sign stability is only a screening condition.</p>
{_table(scheduled_loo_rows, [('asset','Asset',None),('regime','Proxy regime',None),('events','Events',0),('full_mean_pct','Full mean %',3),('loo_min_mean_pct','LOO min %',3),('loo_max_mean_pct','LOO max %',3),('same_sign_all_loo','Same sign',None)])}
{_table(threshold_rows, [('threshold_multiplier','Threshold ×',2),('asset','Asset',None),('regime','Proxy regime',None),('events','Rows',0),('independent_clusters','Events',0),('mean_return_pct','Net mean %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4)])}
</section>

<section><h2>Five-Minute Data Quality</h2>
<p>Coverage is checked at the raw instrument level before signal alignment. Weekend and exchange-session gaps are expected; duplicate timestamps or null closes require review. A five-minute share below 100% can reflect market closures rather than missing in-session bars, so it is a diagnostic rather than an automatic failure.</p>
{_table(scheduled_data_quality, [('dataset','Dataset',None),('rows','Rows',0),('start_utc','Start UTC',None),('end_utc','End UTC',None),('duplicate_timestamps','Duplicates',0),('null_close_pct','Null close %',2),('median_interval_minutes','Median min',1),('expected_interval_share_pct','5m share %',1),('status','Status',None)], limit=80)}
</section>

<section><h2>What The Long-History Proxy Currently Shows</h2>
<p>The chart reports conditional mean forward returns, not a causal estimate. A visually large bar with a small sample or a wide bootstrap interval is not evidence of a tradable edge.</p>
{_mean_return_chart(daily_proxy_summary)}
<h3>Best-looking rows after sorting by corrected p-value</h3>
{_table(best_rows, [('asset','Asset',None),('regime','Regime',None),('horizon_sessions','Sessions',0),('events','N',0),('independent_clusters','Weeks',0),('mean_return_pct','Mean %',3),('median_return_pct','Median %',3),('hit_rate_positive_pct','Positive %',1),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4),('evidence_grade','Grade',None)])}
</section>

<section><h2>Cost And Chronological Robustness</h2>
<p>Net results subtract a fixed round-trip stress of 5 bp for SOXX/QQQ, 10 bp for SOXL/ETH, and 8 bp for the gold-futures proxy. Confidence intervals and tests resample natural-week clusters rather than pretending adjacent daily events are independent.</p>
{_table(net_rows, [('asset','Asset',None),('regime','Regime',None),('horizon_sessions','Sessions',0),('events','N',0),('independent_clusters','Weeks',0),('mean_return_pct','Net mean %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4),('evidence_grade','Grade',None)])}
<p>The table below checks sign consistency across train (2019–2022), validation (2023–2024), and test (2025–2026). Same-sign persistence is a screen, not proof; its magnitude and confidence still have to survive.</p>
{_table(fold_consistency, [('asset','Asset',None),('regime','Regime',None),('horizon_sessions','Sessions',0),('folds','Folds',0),('positive_folds','Positive folds',0),('min_fold_mean_pct','Min net mean %',3),('max_fold_mean_pct','Max net mean %',3),('total_events','Total N',0),('same_sign_all_folds','Same sign',None)])}
</section>

<section><h2>Driver Ablation And Static Background</h2>
<p>The change-signal ablation separates US2Y-led days from the front Fed Funds Futures proxy and from rare same-direction confirmation. This prevents a largely US2Y result from being mislabelled as a FedWatch result.</p>
{_table(ablation_rows, [('signal_subset','Signal subset',None),('asset','Asset',None),('regime','Regime',None),('horizon_sessions','Sessions',0),('events','N',0),('mean_return_pct','Net mean %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4)])}
<p>The static background is a separate trailing-level classifier: at least two of US2Y versus its 1-year median, DXY versus its 200-day mean, and the front Fed Funds implied-rate proxy versus its 6-month median must agree. It is not historical FedWatch stance. It contributes {len(static_background_events):,} background dates before asset-range filtering.</p>
{_table(static_rows, [('asset','Asset',None),('regime','Background',None),('horizon_sessions','Sessions',0),('events','N',0),('independent_clusters','Weeks',0),('mean_return_pct','Net mean %',3),('bootstrap_95_low_pct','CI low',3),('bootstrap_95_high_pct','CI high',3),('bh_adjusted_p_value','BH p',4),('evidence_grade','Grade',None)])}
</section>

<section><h2>True FedWatch / Intraday Event Register</h2>
<p>This is a forensic register for real alerts and market response. It is useful for debugging timing and entries, but the current sample is too short for generalization.</p>
{_table(event_display, [('timestamp_bjt','Beijing time',None),('regime','Regime',None),('severity','Severity',0),('score','Score',2),('fedwatch_policy_bias_pct','Bias',1),('fedwatch_expected_rate_pct','Expected rate %',4),('us2y_yield_pct','US2Y %',3),('drivers','Drivers',None)], limit=60)}
<h3>Aligned intraday outcomes</h3>
{_table(intraday_summary, [('asset','Asset',None),('regime','Regime',None),('horizon_minutes','Minutes',0),('events','N',0),('mean_return_pct','Mean %',3),('median_return_pct','Median %',3),('mean_mae_pct','Mean MAE %',3),('mean_mfe_pct','Mean MFE %',3),('evidence_grade','Grade',None)])}
</section>

<section><h2>Transmission Map And Interpretation</h2>
<div class="grid">
<div class="callout"><h3>Hawkish repricing</h3><p>Higher expected policy rate / higher US2Y / firmer DXY → higher discount rate and tighter financial conditions → pressure on long-duration semiconductor and growth multiples. SOXL adds leverage-path and volatility-decay risk.</p></div>
<div class="callout"><h3>Dovish repricing</h3><p>Lower expected policy rate / lower US2Y / softer DXY → discount-rate relief → potential multiple support. Inflation or growth scares can break this channel, so the sign is conditional rather than automatic.</p></div>
<div class="callout"><h3>ETH and gold</h3><p>ETH can respond through liquidity and dollar channels but also through crypto-specific risk. Gold can benefit from lower real-rate expectations, yet headline inflation, safe-haven demand and positioning can dominate nominal-yield signals.</p></div>
</div>
</section>

<section><h2>What Would Change The View</h2>
<ul>
<li>At least 50 independent true-monitor events spanning multiple CPI, NFP, FOMC and Fed-speech days.</li>
<li>Same directional response in chronological out-of-sample folds, not only in the full sample.</li>
<li>Bootstrap interval excluding zero and BH-adjusted p ≤ 0.05 for a predeclared asset/horizon family.</li>
<li>Return and excursion advantage survives realistic spread, slippage and delayed-entry stress.</li>
<li>No single event day contributes a dominant share of the measured edge.</li>
</ul>
</section>

<section><h2>Strongest Counterargument</h2>
<p>Macro repricing and asset prices react to the same news at nearly the same time. A post-alert return may therefore be continuation, reversal, or noise rather than an exploitable lead. The next-bar convention removes obvious look-ahead, but only a much larger event sample, latency stress and walk-forward stability can distinguish a genuine monitor edge from contemporaneous correlation.</p>
</section>

<footer>Generated by research/macro_repricing_asset_edge · Research output, not a strategy or trading instruction.</footer>
</main></body></html>"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
